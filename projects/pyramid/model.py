from blocks.bricks.base import application
from blocks.bricks import (MLP, Rectifier, Activation, Identity, Initializable)
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, RecurrentStack
from blocks.bricks.sequence_generators import ( 
                        Readout, SequenceGenerator)

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.utils import dict_union

from theano import tensor

from play.bricks.custom import GMMMLP, GMMEmitter, DeepTransitionFeedback
from play.bricks.recurrent import SimpleSequenceAttention

class PyramidLayer(Initializable):
    """Basic unit for the pyramid model.

    """
    def __init__(self,
				 batch_size,
				 frame_size,
				 k,
				 depth,
				 size,
				  **kwargs):
		super(PyramidLayer, self).__init__(**kwargs)

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

		self.transition = transition

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

		self.generator = SequenceGenerator(readout=readout, 
		                              transition=transition,
		                              attention = attention,
		                              name = "generator")

		self.mlp_context = MLP(activations = activations_context,
		                  dims = dims_context)

		self.children = [self.generator, self.mlp_context]
		self.final_states = []
	

    def monitoring_vars(self, cg):

        readout = self.generator.readout
        readouts = VariableFilter( applications = [readout.readout],
            name_regex = "output")(cg.variables)[0]

        mu, sigma, coeff = readout.emitter.components(readouts)

        min_sigma = sigma.min().copy(name="sigma_min")
        mean_sigma = sigma.mean().copy(name="sigma_mean")
        max_sigma = sigma.max().copy(name="sigma_max")

        min_mu = mu.min().copy(name="mu_min")
        mean_mu = mu.mean().copy(name="mu_mean")
        max_mu = mu.max().copy(name="mu_max")

        monitoring_vars = [mean_sigma, min_sigma,
            min_mu, max_mu, mean_mu, max_sigma]

        return monitoring_vars

    @application
    def cost(self, x, context, **kwargs):
        cost_matrix = self.generator.cost_matrix(
                x, attended=self.mlp_context.apply(context),
                **kwargs)

        return cost_matrix.mean()

    @application
    def generate(context):
        return self.generator.generate(
          attended = self.mlp_context.apply(context),
          n_steps = context.shape[0],
          batch_size = context.shape[1],
          iterate = True)

class SimplePyramidLayer(Initializable):
    """Basic unit for the pyramid model.

    """
    def __init__(self,
				 batch_size,
				 frame_size,
				 k,
				 depth,
				 size,
				  **kwargs):
		super(SimplePyramidLayer, self).__init__(**kwargs)

		target_size = frame_size * k

		depth_x = depth
		hidden_size_mlp_x = 32*size

		depth_transition = depth-1

		depth_theta = depth
		hidden_size_mlp_theta = 32*size
		hidden_size_recurrent = 32*size*3

		activations_x = [Rectifier()]*depth_x

		dims_x = [frame_size] + [hidden_size_mlp_x]*(depth_x-1) + \
		         [4*hidden_size_recurrent]

		activations_theta = [Rectifier()]*depth_theta

		dims_theta = [hidden_size_recurrent] + \
		             [hidden_size_mlp_theta]*depth_theta

		self.mlp_x = MLP(activations = activations_x,
		            dims = dims_x,
		            name = "mlp_x")

		transition = [GatedRecurrent(dim=hidden_size_recurrent, 
		                   use_bias = True,
		                   name = "gru_{}".format(i) ) for i in range(depth_transition)]

		self.transition = RecurrentStack( transition,
		            name="transition", skip_connections = True)

		mlp_theta = MLP( activations = activations_theta,
		             dims = dims_theta,
		             name = "mlp_theta")

		mlp_gmm = GMMMLP(mlp = mlp_theta,
		                  dim = target_size,
		                  k = k,
		                  const = 0.00001,
		                  name = "gmm_wrap")

		self.gmm_emitter = GMMEmitter(gmmmlp = mlp_gmm,
		  output_size = frame_size, k = k)

		normal_inputs = [name for name in self.transition.apply.sequences
		                 if 'mask' not in name]

		self.fork = Fork(normal_inputs,
						 input_dim = 4*hidden_size_recurrent,
						 output_dims = self.transition.get_dims(normal_inputs))

		self.children = [self.mlp_x, self.transition,
		                 self.gmm_emitter, self.fork]

    def monitoring_vars(self, cg):

        mu, sigma, coeff = VariableFilter(
        	applications = [self.gmm_emitter.gmmmlp.apply],
        	name_regex = "output")(cg.variables)

        min_sigma = sigma.min().copy(name="sigma_min")
        mean_sigma = sigma.mean().copy(name="sigma_mean")
        max_sigma = sigma.max().copy(name="sigma_max")

        min_mu = mu.min().copy(name="mu_min")
        mean_mu = mu.mean().copy(name="mu_mean")
        max_mu = mu.max().copy(name="mu_max")

        monitoring_vars = [mean_sigma, min_sigma,
            min_mu, max_mu, mean_mu, max_sigma]

        return monitoring_vars

    @application
    def cost(self, x, context, **kwargs):
        x_g = self.mlp_x.apply(context)
        inputs = self.fork.apply(x_g, as_dict = True)
        h = self.transition.apply(**dict_union(inputs, kwargs))

        self.final_states = []
        for var in h:
        	self.final_states.append(var[-1].copy(name = var.name + "_final_value"))

        cost = self.gmm_emitter.cost(h[-1], x)
        return cost.mean()

    @application
    def generate(context):
        x_g = self.mlp_x.apply(context)
        inputs = self.fork.apply(x_g, as_dict = True)
        h = self.transition.apply(**dict_union(inputs, kwargs))
        return self.gmm_emitter.emit(h[-1])