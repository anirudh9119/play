import ipdb
import numpy
import theano
import matplotlib
import sys
import os
import math
matplotlib.use('Agg')

from matplotlib import pyplot
from scipy.io import wavfile

from blocks.algorithms import (GradientDescent, Adam,
                               StepClipping, CompositeRule)
from blocks.bricks import (Tanh, MLP,
                        Rectifier, Activation, Identity)
from blocks.bricks.recurrent import GatedRecurrent, RecurrentStack
from blocks.bricks.sequence_generators import ( 
                        Readout, SequenceGenerator)

from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                        DataStreamMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model

from fuel.streams import ServerDataStream

from theano import tensor, config, function

from play.bricks.custom import GMMMLP, GMMEmitter, DeepTransitionFeedback
from play.bricks.recurrent import SimpleSequenceAttention
from play.datasets.blizzard import Blizzard
from play.extensions import SaveComputationGraph, Flush
from play.extensions.plot import Plot
from play.utils import GMM

###################
# Define parameters of the model
###################

lr = 10 ** (2*numpy.random.rand() - 5)
depth = numpy.random.randint(2,5)
size = numpy.random.randint(10,20)

batch_size = 64
frame_size = 128
k = 32
target_size = frame_size * k

depth_x = depth
hidden_size_mlp_x = 32*size

depth_transition = depth-1

depth_theta = depth
hidden_size_mlp_theta = 32*size
hidden_size_recurrent = 32*size*3

depth_context = depth
hidden_size_mlp_context = 32*size
context_size = 32*size


#print config.recursion_limit
floatX = theano.config.floatX

#job_id = 5557
job_id = int(sys.argv[1])

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'blizzard/', str(job_id) + "/")

experiment_name = 'deep_l32_{}_{}_{}_{}'.format(job_id, lr, depth, size)

train_stream = ServerDataStream(('upsampled', 'residual',), 
                  produces_examples = False,
                  port = job_id)

valid_stream = ServerDataStream(('upsampled', 'residual',), 
                  produces_examples = False,
                  port = job_id+50)
#################
# Model
#################

x = tensor.tensor3('residual')
context = tensor.tensor3('upsampled')

activations_x = [Rectifier()]*depth_x

dims_x = [frame_size] + [hidden_size_mlp_x]*(depth_x-1) + \
         [4*hidden_size_recurrent]

activations_theta = [Rectifier()]*depth_theta

dims_theta = [hidden_size_recurrent] + \
             [hidden_size_mlp_theta]*depth_theta

activations_context = [Rectifier()]*depth_context

dims_context = [frame_size] + [hidden_size_mlp_context]*(depth_context-1) + \
         [context_size]

mlp_x = MLP(activations = activations_x,
            dims = dims_x,
            name = "mlp_x")

feedback = DeepTransitionFeedback(mlp = mlp_x)

transition = [GatedRecurrent(dim=hidden_size_recurrent, 
                   use_bias = True,
                   name = "gru_{}".format(i) ) for i in range(depth_transition)]

transition = RecurrentStack( transition,
            name="transition", skip_connections = True)

mlp_theta = MLP( activations = activations_theta,
             dims = dims_theta,
             name = "mlp_theta")

mlp_gmm = GMMMLP(mlp = mlp_theta,
                  dim = target_size,
                  k = k,
                  const = 0.00001,
                  name = "gmm_wrap")

gmm_emitter = GMMEmitter(gmmmlp = mlp_gmm,
  output_size = frame_size, k = k)

source_names = [name for name in transition.apply.states if 'states' in name]

attention = SimpleSequenceAttention(
              state_names = source_names,
              state_dims = [hidden_size_recurrent],
              attended_dim = context_size,
              name = "attention")

#ipdb.set_trace()
# Verify source names
readout = Readout(
    readout_dim = hidden_size_recurrent,
    source_names =source_names + ['feedback'] + ['glimpses'],
    emitter=gmm_emitter,
    feedback_brick = feedback,
    name="readout")

generator = SequenceGenerator(readout=readout, 
                              transition=transition,
                              attention = attention,
                              name = "generator")

mlp_context = MLP(activations = activations_context,
                  dims = dims_context)

bricks = [mlp_context]

for brick in bricks:
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0.)
    brick.initialize()

generator.weights_init = IsotropicGaussian(0.01)
generator.biases_init = Constant(0.)
generator.push_initialization_config()

#generator.transition.biases_init = IsotropicGaussian(0.01,1)
#generator.transition.push_initialization_config()

generator.initialize()

##############
# Test model
##############

cost_matrix = generator.cost_matrix(x,
        attended = mlp_context.apply(context))
cost = cost_matrix.mean()
cost.name = "nll"

emit = generator.generate(
  attended = mlp_context.apply(context),
  n_steps = context.shape[0],
  batch_size = context.shape[1],
  iterate = True
  )[-4]

cg = ComputationGraph(cost)
model = Model(cost)

transition_matrix = VariableFilter(
            theano_name_regex = "state_to_state")(cg.parameters)
for matr in transition_matrix:
    matr.set_value(0.98*numpy.eye(hidden_size_recurrent, dtype = floatX))


#################
# Algorithm
#################

n_batches = 139#139*16

algorithm = GradientDescent(
    cost=cost, parameters=cg.parameters,
    step_rule=CompositeRule([StepClipping(10.0), Adam(lr)]))

train_monitor = TrainingDataMonitoring(
    variables=[cost],
    every_n_batches = n_batches,
    prefix="train")

valid_monitor = DataStreamMonitoring(
     [cost],
     valid_stream,
     after_epoch = True,
     every_n_batches = n_batches,
     prefix="valid")

def _is_nan(log):
    try:
      result = math.isnan(log.current_row['train_nll'])
      return result
    except:
      return False

extensions = extensions=[
    Timing(every_n_batches = n_batches),
    train_monitor,
    valid_monitor,
    TrackTheBest('valid_nll', after_epoch = True),
    Plot(save_dir+experiment_name+".png",
         [['train_nll',
           'valid_nll']],
         every_n_batches = 4*n_batches,
         email=False),
    Checkpoint(save_dir+experiment_name+".pkl",
               use_cpickle = True,
               every_n_batches = n_batches*8,
               after_epoch = True),
    Checkpoint(save_dir+"best_"+experiment_name+".pkl",
     after_epoch = True,
     use_cpickle = True
     ).add_condition(['after_epoch'],
          predicate=OnLogRecord('valid_nll_best_so_far')),
    Printing(every_n_batches = n_batches, after_epoch = True),
    FinishAfter(after_n_epochs=10)
    .add_condition(["after_batch"], _is_nan),
    SaveComputationGraph(emit),
    Flush(every_n_batches = n_batches, after_epoch = True)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()
