import ipdb
import numpy
import theano
import matplotlib
import os
matplotlib.use('Agg')

from matplotlib import pyplot
from scipy.io import wavfile

from blocks.algorithms import (GradientDescent, Scale,
                               RMSProp, Adam,
                               StepClipping, CompositeRule)
from blocks.bricks import (Tanh, MLP,
                        Rectifier, Activation, Identity)

from blocks.bricks.sequence_generators import ( 
                        Readout, SequenceGenerator)
from blocks.bricks.recurrent import LSTM, RecurrentStack, GatedRecurrent
from blocks.extensions import FinishAfter, Printing, Timing, ProgressBar
from blocks.extensions.monitoring import (TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.utils import shared_floatx_zeros, shared_floatx

from fuel.transformers import (Mapping, FilterSources,
                        ForceFloatX, ScaleAndShift)
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream

from theano import tensor, config, function

from play.bricks.custom import (DeepTransitionFeedback, GMMEmitter,
                     GMMMLP)
from play.bricks.recurrent import SimpleSequenceAttention

from play.datasets.blizzard import Blizzard
from play.extensions import Flush, LearningRateSchedule, TimedFinish
from play.extensions.plot import Plot
from play.toy.segment_transformer import SegmentSequence

###################
# Define parameters of the model
###################

batch_size = 64 #for tpbtt
frame_size = 257
seq_size = 128
k = 20
target_size = frame_size * k

depth_x = 4
hidden_size_mlp_x = 2#000

depth_theta = 4
hidden_size_mlp_theta = 2#000
hidden_size_recurrent = 2#000

depth_context = 2
hidden_size_mlp_context = 5#00
context_size = 100

depth_recurrent = 3
lr = 5e-4
lr = shared_floatx(lr, "learning_rate")

floatX = theano.config.floatX

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'blizzard/')

experiment_name = "sp_conditional_f0_0"

#################
# Prepare dataset
#################

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

def _is_nonzero(data):
    return tuple([1.*(data[0]>0)])

def _zero_for_unvoiced(data):
    #Multiply by 0 the unvoiced components.
    return tuple([data[0]*data[2],data[1],data[2]])

data_dir = os.environ['FUEL_DATA_PATH']
data_dir = os.path.join(data_dir, 'blizzard/', 'sp_standardize.npz')

data_stats = numpy.load(data_dir)
sp_mean = data_stats['sp_mean']
sp_std = data_stats['sp_std']
f0_mean = data_stats['f0_mean']
f0_std = data_stats['f0_std']

dataset = Blizzard(which_sets = ('train',), filename = "sp_blizzard.hdf5")
data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            batch_size*(dataset.num_examples/batch_size), batch_size))
data_stream = Mapping(data_stream, _is_nonzero, add_sources = ('voiced',))
data_stream = ScaleAndShift(data_stream,
                            scale = 1/sp_std, 
                            shift = -sp_mean/sp_std,
                            which_sources = ('sp',))
data_stream = ScaleAndShift(data_stream,
                            scale = 1/f0_std, 
                            shift = -f0_mean/f0_std,
                            which_sources = ('f0',))
data_stream = Mapping(data_stream, _zero_for_unvoiced)
data_stream = Mapping(data_stream, _transpose)
data_stream = SegmentSequence(data_stream, seq_size, add_flag=True)
data_stream = ForceFloatX(data_stream)
train_stream = data_stream

num_valid_examples = 4*64
dataset = Blizzard(which_sets = ('valid',), filename = "sp_blizzard.hdf5")
data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            num_valid_examples, batch_size))
data_stream = Mapping(data_stream, _is_nonzero, add_sources = ('voiced',))
data_stream = ScaleAndShift(data_stream,
                            scale = 1/sp_std, 
                            shift = -sp_mean/sp_std,
                            which_sources = ('sp',))
data_stream = ScaleAndShift(data_stream,
                            scale = 1/f0_std, 
                            shift = -f0_mean/f0_std,
                            which_sources = ('f0',))
data_stream = Mapping(data_stream, _zero_for_unvoiced)
data_stream = Mapping(data_stream, _transpose)
data_stream = SegmentSequence(data_stream, 8*seq_size, add_flag=True)
data_stream = ForceFloatX(data_stream)
valid_stream = data_stream

#################
# Model
#################

start_flag = tensor.scalar('start_flag')
x = tensor.tensor3('sp')
#x = tensor.tensor3('features')

f0 = tensor.matrix('f0')
voiced = tensor.matrix('voiced')

f0s = f0.dimshuffle(0,1,'x')
voiceds = voiced.dimshuffle(0,1,'x')

context = tensor.concatenate([f0s, voiceds], 2)

activations_x = [Rectifier()]*depth_x

dims_x = [frame_size] + [hidden_size_mlp_x]*(depth_x-1) + \
         [hidden_size_recurrent]

activations_theta = [Rectifier()]*depth_theta

dims_theta = [hidden_size_recurrent] + \
             [hidden_size_mlp_theta]*depth_theta

activations_context = [Rectifier()]*depth_context

dims_context = [2] + [hidden_size_mlp_context]*(depth_context-1) + \
         [context_size]

mlp_x = MLP(activations = activations_x,
            dims = dims_x)

feedback = DeepTransitionFeedback(mlp = mlp_x)

transition = [GatedRecurrent(dim=hidden_size_recurrent, 
                   name = "gru_{}".format(i) ) for i in range(depth_recurrent)]

transition = RecurrentStack( transition,
            name="transition", skip_connections = True)

mlp_theta = MLP( activations = activations_theta,
             dims = dims_theta)

mlp_context = MLP(activations = activations_context,
                      dims = dims_context)

source_names = [name for name in transition.apply.states if 'states' in name]

attention = SimpleSequenceAttention(
              state_names = source_names,
              state_dims = [hidden_size_recurrent],
              attended_dim = context_size,
              name = "attention")

mlp_gmm = GMMMLP(mlp = mlp_theta,
                  dim = target_size,
                  k = k,
                  const = 0.00001)

emitter = GMMEmitter(gmmmlp = mlp_gmm,
                     output_size = frame_size,
                     k = k,
                     name = "emitter")

source_names = [name for name in transition.apply.states if 'states' in name]
readout = Readout(
    readout_dim = hidden_size_recurrent,
    source_names =source_names,
    emitter=emitter,
    feedback_brick = feedback,
    name="readout")

generator = SequenceGenerator(readout=readout, 
                              transition=transition,
                              attention = attention,
                              name = "generator")

generator.weights_init = IsotropicGaussian(0.01)
generator.biases_init = Constant(0.)
generator.push_initialization_config()

mlp_context.weights_init = IsotropicGaussian(0.001)
mlp_context.biases_init = Constant(0.)

generator.transition.biases_init = IsotropicGaussian(0.01,1)
generator.transition.push_initialization_config()

generator.initialize()

states = {}
states = generator.transition.apply.outputs

states = {name: shared_floatx_zeros((batch_size, hidden_size_recurrent))
        for name in states}

x_tr=next(data_stream.get_epoch_iterator())
#ipdb.set_trace()

print function([f0,voiced], mlp_context.apply(context))(x_tr[0],x_tr[2]).shape

cost_matrix = generator.cost_matrix(x, attended = mlp_context.apply(context))# , **states)

print function([f0,x,voiced], cost_matrix)(x_tr[0],x_tr[1],x_tr[2]).shape

cost = cost_matrix.mean() + 0.*start_flag
cost.name = "nll"

cg = ComputationGraph(cost)
model = Model(cost)

transition_matrix = VariableFilter(
            theano_name_regex="state_to_state")(cg.parameters)
for matr in transition_matrix:
    matr.set_value(0.98*numpy.eye(hidden_size_recurrent, dtype=floatX))

from play.utils import regex_final_value
extra_updates = []
# for name, var in states.items():
#   ipdb.set_trace()
#   update = tensor.switch(start_flag, 0.*var,
#               VariableFilter(theano_name_regex=regex_final_value(name)
#                   )(cg.auxiliary_variables)[0])
#   extra_updates.append((var, update))

#################
# Monitoring vars
#################

mean_data = x.mean(axis = (0,1)).copy(name="data_mean")
sigma_data = x.std(axis = (0,1)).copy(name="data_std")
max_data = x.max(axis = (0,1)).copy(name="data_max")
min_data = x.min(axis = (0,1)).copy(name="data_min")

monitoring_variables = [cost, lr]

data_monitoring = [mean_data, sigma_data,
                     max_data, min_data]

#################
# Algorithm
#################

n_batches = 200

algorithm = GradientDescent(
    cost=cost, parameters=cg.parameters,
    step_rule=CompositeRule([StepClipping(10.0), Adam(lr)]))
algorithm.add_updates(extra_updates)

train_monitor = TrainingDataMonitoring(
    variables=monitoring_variables + data_monitoring,
    every_n_batches=n_batches,
    prefix="train")

valid_monitor = DataStreamMonitoring(
     monitoring_variables,
     valid_stream,
     every_n_batches=n_batches,
     prefix="valid")

extensions=[
    ProgressBar(),
    Timing(every_n_batches=n_batches),
    train_monitor,
    valid_monitor,
    TrackTheBest('valid_nll', every_n_batches=n_batches),
    Plot(save_dir+ "progress/" +experiment_name+".png",
     [['train_nll',
       'valid_nll'], ['valid_learning_rate']],
     every_n_batches=n_batches,
     email=False),
    Checkpoint(
        save_dir+"pkl/best_"+experiment_name+".pkl",
        use_cpickle=True
    ).add_condition(
        ['after_batch'], predicate=OnLogRecord('valid_nll_best_so_far')),
    Printing(every_n_batches = n_batches),
    Flush(every_n_batches=n_batches,
          before_first_epoch = True),
    # LearningRateSchedule(lr,
    #   'valid_nll',
    #   #path = save_dir+"pkl/best_"+experiment_name+".pkl",
    #   states = [],#states.values(),
    #   every_n_batches = n_batches,
    #   before_first_epoch = True)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()