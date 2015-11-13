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

from play.datasets.blizzard import Blizzard
from play.extensions import SaveComputationGraph, Flush
from play.extensions.plot import Plot

from play.projects.pyramid.config import PyramidParameters

###################
# Define parameters of the model
###################

pl_params = PyramidParameters()

lr = pl_params.lr
depth = pl_params.depth
size = pl_params.size

batch_size = pl_params.batch_size
frame_size = pl_params.frame_size
k = pl_params.k
target_size = pl_params.target_size

depth_x = pl_params.depth_x
hidden_size_mlp_x = pl_params.hidden_size_mlp_x

depth_transition = pl_params.depth_transition

depth_theta = pl_params.depth_theta
hidden_size_mlp_theta = pl_params.hidden_size_mlp_theta
hidden_size_recurrent = pl_params.hidden_size_recurrent

depth_context = pl_params.depth_context
hidden_size_mlp_context = pl_params.hidden_size_mlp_context
context_size = pl_params.context_size

#print config.recursion_limit
floatX = theano.config.floatX

job_id = 5656
#job_id = int(sys.argv[1])

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

from play.projects.pyramid.model import PyramidLayer
pl= PyramidLayer(batch_size, frame_size, k, depth, size)

x = tensor.tensor3('residual')
context = tensor.tensor3('upsampled')

mlp_context = pl.mlp_context
generator = pl.generator

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

n_batches = 16#139#139*16

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
