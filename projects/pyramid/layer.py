import numpy
import theano
import sys
import os
import math
import ipdb

from blocks.algorithms import (GradientDescent, Adam,
                               StepClipping, CompositeRule)

from blocks.extensions import FinishAfter, Printing, Timing, ProgressBar
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
from blocks.utils import shared_floatx

from fuel.streams import ServerDataStream

from theano import tensor

from play.extensions import Flush, LearningRateSchedule2, TimedFinish
from play.extensions.plot import Plot
from play.projects.pyramid.config import PyramidParameters
from play.projects.pyramid.model import PyramidLayer, SimplePyramidLayer
from play.toy.segment_transformer import SegmentSequence
from play.datasets.server.blizzard.server_stream import open_stream

###################
# Define parameters of the model
###################

save_dir = os.environ['RESULTS_DIR']
if 'blizzard' not in save_dir:
  save_dir = os.path.join(save_dir,'/blizzard/')

if len(sys.argv) > 1:
  num_job = int(sys.argv[1])
else:
  num_job = 0

pl_params = PyramidParameters(num_job)

tbptt_flag = pl_params.tbptt_flag

lr = pl_params.lr
lr = shared_floatx(lr, "learning_rate")

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

n_batches = pl_params.n_batches
seq_length = pl_params.seq_length

# print config.recursion_limit
floatX = theano.config.floatX

experiment_name = pl_params.experiment_name

stream_vars = ('upsampled', 'residual',)

train_stream = ServerDataStream(
                  stream_vars,
                  produces_examples=False,
                  port=pl_params.port)

valid_stream = ServerDataStream(
                  stream_vars,
                  produces_examples=False,
                  port=pl_params.port+50)

if tbptt_flag:
    train_stream = SegmentSequence(train_stream, seq_length, add_flag=True)
    valid_stream = SegmentSequence(valid_stream, seq_length, add_flag=True)

#x_tr = next(train_stream.get_epoch_iterator())

#################
# Model
#################

pl = PyramidLayer(batch_size, frame_size, k, depth, size)

x = tensor.tensor3('residual')
context = tensor.tensor3('upsampled')
start_flag = tensor.scalar('start_flag')

pl.weights_init = IsotropicGaussian(0.01)
pl.biases_init = Constant(0.)
pl.initialize()

states = {}
if tbptt_flag:
  states = pl.transition.apply.outputs

  from blocks.utils import shared_floatx_zeros
  states = {name: shared_floatx_zeros((batch_size, hidden_size_recurrent))
            for name in states}

##############
# Test model
##############

cost = pl.cost(x, context, **states)

if tbptt_flag:
    cost += 0.*start_flag

cost.name = "nll"

cg = ComputationGraph(cost)
model = Model(cost)

transition_matrix = VariableFilter(
            theano_name_regex="state_to_state")(cg.parameters)
for matr in transition_matrix:
    matr.set_value(0.98*numpy.eye(hidden_size_recurrent, dtype=floatX))

from play.utils import regex_final_value
extra_updates = []
if tbptt_flag:
  for name, var in states.items():
      update = tensor.switch(start_flag, 0.*var,
                  VariableFilter(theano_name_regex=regex_final_value(name)
                      )(pl.final_states + cg.auxiliary_variables)[0])
      extra_updates.append((var, update))

#################
# Monitoring vars
#################

mean_residual = x.mean().copy(name="residual_mean")
sigma_residual = x.std().copy(name="residual_std")
max_residual = x.max().copy(name="residual_max")
min_residual = x.min().copy(name="residual_min")

mean_context = context.mean().copy(name="upsampled_mean")
sigma_context = context.std().copy(name="upsampled_std")
max_context = context.max().copy(name="upsampled_max")
min_context = context.min().copy(name="upsampled_min")

monitoring_variables = [lr, cost, mean_residual, sigma_residual,
                        max_residual, min_residual, mean_context,
                        sigma_context, max_context, min_context]

monitoring_variables += pl.monitoring_vars(cg)

#################
# Algorithm
#################

algorithm = GradientDescent(
    cost=cost, parameters=cg.parameters,
    step_rule=CompositeRule([StepClipping(10.0), Adam(lr)]))
algorithm.add_updates(extra_updates)

train_monitor = TrainingDataMonitoring(
    variables=monitoring_variables,
    every_n_batches=n_batches,
    prefix="train")

valid_monitor = DataStreamMonitoring(
     monitoring_variables,
     valid_stream,
     after_epoch=True,
     every_n_batches=n_batches,
     prefix="valid")

extensions = [
    Timing(every_n_batches=n_batches),
    train_monitor,
    valid_monitor,
    TrackTheBest('valid_nll', every_n_batches=n_batches),
    Plot(save_dir+ "progress/" +experiment_name+".png",
         [['train_nll',
           'valid_nll']],
         every_n_batches=4*n_batches,
         email=False),
    Checkpoint(
      save_dir+"pkl/best_"+experiment_name+".pkl",
      use_cpickle=True
     ).add_condition(
        ['after_batch'], predicate=OnLogRecord('valid_nll_best_so_far')),
    #ProgressBar(),
    Printing(every_n_batches=n_batches, after_epoch=True),
    Flush(every_n_batches=n_batches,
          after_epoch=True,
          before_first_epoch = True),
    LearningRateSchedule(lr,
      'valid_nll',
      states = states.values(),
      path = save_dir+"pkl/best_"+experiment_name+".pkl",
      every_n_batches = n_batches,
      before_first_epoch = True),
    TimedFinish(60*60*22)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions=extensions)

main_loop.run()
# ipdb.set_trace()
