import sys
import numpy
import os

from multiprocessing import Process

from cle.cle.utils import segment_axis

from fuel.transformers import (Mapping, ForceFloatX, ScaleAndShift, Cast,
                               FilterSources, AgnosticSourcewiseTransformer)
from fuel.schemes import SequentialScheme
from fuel.server import start_server
from fuel.streams import DataStream

from play.datasets.blizzard import Blizzard
from play.toy.segment_transformer import SegmentSequence

from scikits.samplerate import resample

#################
# Prepare dataset
#################

def _transpose(data):
    return tuple(array.swapaxes(0, 1) for array in data)


def _copy(data):
    return data


class Resample(AgnosticSourcewiseTransformer):
    def __init__(self, data_stream, which_sources=None, scale=1., **kwargs):
        super(Resample, self).__init__(data_stream,
                                       data_stream.produces_examples,
                                       which_sources, **kwargs)
        self.scale = scale

    def transform_any_source(self, source_data, source_name):
        return numpy.array([resample(x, self.scale, 'sinc_best')
                            for x in source_data])


def _equalize_size(data):
    min_size = [min([len(x) for x in sequences]) for sequences in zip(*data)]
    x = tuple([numpy.array([x[:size] for x, size in zip(var, min_size)])
               for var in data])
    return x


def _get_residual(data):
    # The order is correct?
    ds = numpy.array([x[0]-x[1] for x in zip(*data)])
    return (ds,)

data_dir = os.environ['FUEL_DATA_PATH']
data_dir = os.path.join(data_dir, 'blizzard/', 'blizzard_standardize.npz')

data_stats = numpy.load(data_dir)
data_mean = data_stats['data_mean']
data_std = data_stats['data_std']

def define_stream(which_sets=('train',),
                initial_scale=1,
                scale=0.5,
                batch_size=64,
                seq_length=64,
                frame_size=128,
                tbptt_flag = True,
                num_examples=None):

    def _segment_axis(data):
        # Defined inside so that frame_size is available
        x = tuple([numpy.array([segment_axis(x, frame_size, 0) for x in var])
                   for var in data])
        return x

    scale = float(scale)

    dataset = Blizzard(which_sets=which_sets)

    if num_examples is None:
        num_examples = batch_size*(dataset.num_examples/batch_size)

    data_stream = DataStream.default_stream(
            dataset,
            iteration_scheme=SequentialScheme(num_examples, batch_size))

    data_stream = ScaleAndShift(data_stream,
                                scale=1/data_std,
                                shift=-data_mean/float(data_std))

    # Original sampling rate
    data_stream = Resample(data_stream, scale=initial_scale)
    data_stream = Mapping(data_stream, _copy, add_sources=('upsampled',))
    data_stream = Resample(data_stream, scale=scale, which_sources=('upsampled',))
    data_stream = Resample(data_stream, scale=1/scale, which_sources=('upsampled',))

    # data_stream = Mapping(data_stream, _downsample_and_upsample,
    #                       add_sources=('upsampled',))
    data_stream = Mapping(data_stream, _equalize_size)
    data_stream = Mapping(data_stream, _get_residual,
                          add_sources=('residual',))
    data_stream = FilterSources(data_stream,
                                sources=('upsampled', 'residual',))
    data_stream = Mapping(data_stream, _segment_axis)
    data_stream = Mapping(data_stream, _transpose)
    return data_stream

def open_stream(which_sets=('train',),
                initial_scale=1,
                scale=0.5,
                batch_size=64,
                seq_length=64,
                frame_size=128,
                port=5557,
                tbptt_flag = True,
                num_examples=None):

    data_stream = define_stream(which_sets, initial_scale, scale,
        batch_size, seq_length, frame_size, tbptt_flag, num_examples)
    start_server(data_stream, port=port)

if __name__ == "__main__":
    from play.projects.pyramid.config import PyramidParameters

    if len(sys.argv) > 1:
      num_job = int(sys.argv[1])
    else:
      num_job = 0

    pl_params = PyramidParameters(num_job)

    port = pl_params.port
    num_valid_examples = 64

    Process(target=open_stream, 
        args=(('train',),
             pl_params.initial_scale,
             pl_params.scale,
             pl_params.batch_size,
             pl_params.seq_length,
             pl_params.frame_size,
             pl_params.port,
             pl_params.tbptt_flag,
             None)).start()
    Process(target=open_stream, 
        args=(('valid',),
             pl_params.initial_scale,
             pl_params.scale,
             pl_params.batch_size,
             pl_params.seq_length,
             pl_params.frame_size,
             pl_params.port+50,
             pl_params.tbptt_flag,
             num_valid_examples)).start()
