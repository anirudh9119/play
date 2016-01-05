import sys
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import copy
import time
import math
import ipdb

from blocks.filter import VariableFilter
from blocks.extensions import SimpleExtension
from blocks.graph import ComputationGraph
from blocks.roles import ALGORITHM_BUFFER
from blocks.serialization import load

class SaveComputationGraph(SimpleExtension):
    def __init__(self, variable, **kwargs):
        super(SaveComputationGraph, self).__init__(**kwargs)
        variable_graph = ComputationGraph(variable)
        self.theano_function = variable_graph.get_theano_function()

    def do(self, which_callback, *args):
        print "empty"

class Flush(SimpleExtension):
    def do(self, which_callback, *args):
        sys.stdout.flush()

class TimedFinish(SimpleExtension):
    def __init__(self, time_limit):
        super(TimedFinish, self).__init__(after_batch = True)
        self.time_limit = time_limit
        self.start_time = time.time()

    def do(self, which_callback, *args):
    	if time.time() - self.start_time > self.time_limit:
    		self.main_loop.log.current_row['training_finish_requested'] = True

class LearningRateSchedule(SimpleExtension):
    """ Control learning rate.
    """
    def __init__(self, lr, track_var, states = {}, path = None, **kwargs):
        self.lr = lr
        self.patience = 15 #3
        self.counter = 0
        self.best_value = numpy.inf
        self.track_var = track_var
        # self.iteration_state = None
        self.log = None
        self.parameter_values = None
        self.algorithm_buffers = None
        self.tolerance = 1e-13
        self.states = states
        self.epsilon = -1e-5

        if path is not None:
            loaded_main_loop = load(path)
            #Hardcoded
            ext = loaded_main_loop.extensions[-1]
            self.lr.set_value(2.*ext.lr.get_value())
            self.log = ext.log
            self.parameter_values = ext.parameter_values
            self.best_value = ext.best_value
            self.counter = self.patience

        super(LearningRateSchedule, self).__init__(**kwargs)

    def do(self, callback_name, *args):
        current_value = self.main_loop.log.current_row.get(self.track_var)
        if current_value is None:
            return

        if current_value < self.best_value - self.epsilon:
            self.best_value = current_value
            self.counter = 0
            # self.iteration_state = copy.deepcopy(self.main_loop.iteration_state)
            self.log = copy.deepcopy(self.main_loop.log)
            self.parameter_values = self.main_loop.model.get_parameter_values()
        else:
            self.counter += 1

        # If nan, skip steps to go back.
        if math.isnan(current_value):
            self.counter = self.patience + 1

        if self.algorithm_buffers is None:
            self.algorithm_buffers = [x for x,y in self.main_loop.algorithm.step_rule_updates]
            self.algorithm_buffers = VariableFilter(roles = [ALGORITHM_BUFFER])(self.algorithm_buffers)
            # self.algorithm_values = [x.get_value() for x in self.algorithm_buffers]

        if self.counter > self.patience:
            self.counter = 0
            # self.main_loop.iteration_state = self.iteration_state
            #self.main_loop.log = self.log
            self.main_loop.model.set_parameter_values(self.parameter_values)

            # Reset algorithm buffer
            for var in self.algorithm_buffers:
                var_value = var.get_value()
                var.set_value(numpy.zeros(var_value.shape, dtype = var_value.dtype))

            # Reset states
            for var in self.states:
                var_value = var.get_value()
                var.set_value(numpy.zeros(var_value.shape, dtype = var_value.dtype))

            self.lr.set_value(float(0.5*self.lr.get_value()))

            if self.lr.get_value() < self.tolerance:
                self.main_loop.log.current_row['training_finish_requested'] = True
