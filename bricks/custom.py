import theano
import ipdb

from theano import tensor, config

from blocks.bricks import (Activation, Initializable, MLP, Random,
                        Identity, NDimensionalSoftmax)
from blocks.bricks.base import application
from blocks.bricks.sequence_generators import (AbstractEmitter, 
                        AbstractFeedback)

from cle.cle.cost import Gaussian
from cle.cle.utils import predict

from play.utils import GMM

floatX = config.floatX

import numpy

class SoftPlus(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.softplus(input_)

class DeepTransitionFeedback(AbstractFeedback, Initializable):
    def __init__(self, mlp, **kwargs):
        super(DeepTransitionFeedback, self).__init__(**kwargs)

        self.mlp = mlp
        self.feedback_dim = mlp.output_dim
        self.children = [self.mlp]

    @application(outputs=['feedback'])
    def feedback(self, outputs):
        return self.mlp.apply(outputs)

    def get_dim(self, name):
        if name == 'feedback':
            return self.feedback_dim
        return super(DeepTransitionFeedback, self).get_dim(name)

class GaussianMLP(Initializable):
    """An mlp brick that branchs out to output
    sigma and mu for Gaussian dist
    Parameters
    ----------
    mlp: MLP brick
        the main mlp to wrap around.
    dim:
        output dim
    """

    def __init__(self, mlp, dim, const=0., **kwargs):
        super(GaussianMLP, self).__init__(**kwargs)

        self.dim = dim
        self.const = const
        input_dim = mlp.output_dim
        self.mu = MLP(activations=[Identity()],
                      dims=[input_dim, dim],
                      weights_init=self.weights_init,
                      biases_init=self.biases_init,
                      name=self.name + "_mu")
        self.sigma = MLP(activations=[SoftPlus()],
                         dims=[input_dim, dim],
                         weights_init=self.weights_init,
                         biases_init=self.biases_init,
                         name=self.name + "_sigma")

        self.mlp = mlp
        self.children = [self.mlp, self.mu, self.sigma]
        self.children.extend(self.mlp.children)

    @application
    def apply(self, inputs):
        state = self.mlp.apply(inputs)
        mu = self.mu.apply(state)
        sigma = self.sigma.apply(state) + self.const

        return mu, sigma

    @property
    def output_dim(self):
        return self.dim

class GaussianEmitter(AbstractEmitter, Initializable, Random):
    """A Gaussian emitter for the case of real outputs.
    Parameters
    ----------
    initial_output :
        The initial output.
    """
    def __init__(self, gaussianmlp, output_size, **kwargs):
        super(GaussianEmitter, self).__init__(**kwargs)
        self.gaussianmlp = gaussianmlp
        self.children = [self.gaussianmlp]
        self.output_size = output_size

    def components(self, readouts):
        # Returns Mu and Sigma
        return self.gaussianmlp.apply(readouts)

    @application
    def emit(self, readouts):
        mu, sigma = self.components(readouts)

        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)

        return mu + sigma*epsilon

    @application
    def cost(self, readouts, outputs):
        mu, sigma = self.components(readouts)
        return Gaussian(outputs, mu, sigma)

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size, self.output_size), dtype=floatX)

    def get_dim(self, name):
        if name == 'outputs':
            return self.gaussianmlp.output_dim
        return super(GaussianEmitter, self).get_dim(name)


class GMMMLP(Initializable):
    """An mlp brick that branchs out to output
    sigma and mu for GMM
    Parameters
    ----------
    mlp: MLP brick
        the main mlp to wrap around.
    dim:
        output dim
    """
    def __init__(self, mlp, dim, k, const=1e-5, **kwargs):
        super(GMMMLP, self).__init__(**kwargs)

        self.dim = dim
        self.const = const
        self.k = k
        input_dim = mlp.output_dim
        self.mu = MLP(activations=[Identity()],
                      dims=[input_dim, dim],
                      name=self.name + "_mu")
        self.sigma = MLP(activations=[SoftPlus()],
                         dims=[input_dim, dim],
                         name=self.name + "_sigma")

        self.coeff = MLP(activations=[Identity()],
                         dims=[input_dim, k],
                         name=self.name + "_coeff")

        self.coeff2 = NDimensionalSoftmax()
        self.mlp = mlp
        self.children = [self.mlp, self.mu, 
                         self.sigma, self.coeff, self.coeff2]
        #self.children.extend(self.mlp.children)

    @application
    def apply(self, inputs):
        state = self.mlp.apply(inputs)
        mu = self.mu.apply(state)
        sigma = self.sigma.apply(state)+ self.const
        coeff = self.coeff2.apply(self.coeff.apply(state),
            extra_ndim=state.ndim - 2)
        return mu, sigma, coeff

    @property
    def output_dim(self):
        return self.dim


class GMMEmitter(AbstractEmitter, Initializable, Random):
    """A GMM emitter for the case of real outputs.
    Parameters
    ----------
    """
    def __init__(self, gmmmlp, output_size, k, **kwargs):
        super(GMMEmitter, self).__init__(**kwargs)
        self.gmmmlp = gmmmlp
        self.children = [self.gmmmlp]
        self.output_size = output_size
        self.k = k

    def components(self, readouts):
        # Returns Mu and Sigma
        return self.gmmmlp.apply(readouts)

    @application
    def emit(self, readouts):
        mu, sigma, coeff = self.gmmmlp.apply(readouts)
        
        frame_size = mu.shape[-1]/coeff.shape[-1]
        k = coeff.shape[-1]
        shape_result = coeff.shape
        shape_result = tensor.set_subtensor(shape_result[-1],frame_size)
        ndim_result = coeff.ndim

        mu = mu.reshape((-1, frame_size, k))
        sigma = sigma.reshape((-1, frame_size,k))
        coeff = coeff.reshape((-1, k))

        sample_coeff = self.theano_rng.multinomial(pvals = coeff, dtype=coeff.dtype)
        idx = predict(sample_coeff, axis = -1)

        mu = mu[tensor.arange(mu.shape[0]), :, idx]
        sigma = sigma[tensor.arange(sigma.shape[0]), :, idx]

        epsilon = self.theano_rng.normal(
            size=mu.shape,avg=0.,
            std=1.,
            dtype=mu.dtype)

        result = mu + sigma*epsilon

        return result.reshape(shape_result, ndim = ndim_result)

    @application
    def cost(self, readouts, outputs):
        mu, sigma, coeff = self.components(readouts)
        #ipdb.set_trace()
        return GMM(outputs, mu, sigma, coeff)

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size, self.output_size), dtype=floatX)

    def get_dim(self, name):
        if name == 'outputs':
            return self.gmmmlp.output_dim
        return super(GMMEmitter, self).get_dim(name)