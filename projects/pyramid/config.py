import ipdb
import numpy

class PyramidParameters:
    def __init__(self, seed=0):
        numpy.random.seed(seed = seed)
        self.lr = 10 ** (4*numpy.random.rand() - 5)
        self.depth = numpy.random.randint(2,5)
        self.size = numpy.random.randint(10,20)

        self.batch_size = 64
        self.frame_size = 128
        self.k = 32
        self.target_size = self.frame_size * self.k
        self.seq_length = 64

        self.depth_x = self.depth
        self.hidden_size_mlp_x = 32*self.size

        self.depth_transition = self.depth-1

        self.depth_theta = self.depth
        self.hidden_size_mlp_theta = 32*self.size
        self.hidden_size_recurrent = 32*self.size*3

        self.depth_context = self.depth
        self.hidden_size_mlp_context = 32*self.size
        self.context_size = 32*self.size
        self.n_batches = 139*6

        # datastream

        self.initial_scale = .25
        self.scale = .5
        self.port = 5800+seed

        # tbptt
        self.tbptt_flag = True
        self.experiment_name = 'exp_{}'.format(seed)


if __name__ == "__main__":
    pyramid_params = PyramidParameters()
    ipdb.set_trace()
