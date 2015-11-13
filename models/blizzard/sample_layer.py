import numpy
import matplotlib
import os
import ipdb
matplotlib.use('Agg')
from matplotlib import pyplot

import numpy
import sys
from blocks.serialization import load
from cle.cle.utils import segment_axis
from fuel.transformers import (Mapping, Padding, 
                        ForceFloatX, ScaleAndShift,
                        FilterSources)
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from play.datasets.blizzard import Blizzard
from scikits.samplerate import resample
from scipy.io import wavfile

batch_size = 64
frame_size = 128
n_iter = 2
n_samples = 5

def _segment_axis(data):
        # Defined inside so that frame_size is available
        x = tuple([numpy.array([segment_axis(x, frame_size, 0) for x in var])
                   for var in data])
        return x

from play.datasets.server.blizzard.server_stream import (
    define_stream, Resample, _copy, _equalize_size, _transpose)

data_stream = define_stream(('test',))

data_dir = os.environ['FUEL_DATA_PATH']
data_dir = os.path.join(data_dir, 'blizzard/', 'blizzard_standardize.npz')

data_stats = numpy.load(data_dir)
data_mean = data_stats['data_mean']
data_std = data_stats['data_std']

which_sets= ('test',)

dataset = Blizzard(which_sets = which_sets)

data_stream = DataStream.default_stream(
        dataset, iteration_scheme=SequentialScheme(
        dataset.num_examples, batch_size))

epoch_iterator = data_stream.get_epoch_iterator()
raw_audio = next(epoch_iterator)[0]

for i in xrange(n_iter-1):
    x_tr = next(epoch_iterator)[0]
    raw_audio = numpy.hstack([raw_audio, x_tr])

save_dir = os.environ['RESULTS_DIR']
exp_path = os.path.join(save_dir,'blizzard/')

# for file_ in os.listdir(exp_path):
#     if file_.endswith(".pkl"):
#         exp_file = file_

exp_file = 'best_simple_0.pkl'
save_dir = os.path.join(save_dir,'blizzard/', "samples/")

def _is_nan(log):
    try:
        result = math.isnan(log.current_row['train_nll'])
        return result
    except:
        return False

main_loop = load(os.path.join(exp_path,exp_file))

rate = 16000

for i, sample in enumerate(raw_audio[:n_samples]):
    pyplot.plot(sample)
    pyplot.savefig(save_dir +"original_%i.png" % i)
    pyplot.close()

    wavfile.write(save_dir + "original_{}.wav".format(i),
        rate, sample)

data_stream = ScaleAndShift(data_stream,
                            scale=1/data_std,
                            shift=-data_mean/float(data_std))

initial_scale=1
scale=0.5
batch_size=64
seq_length=64
frame_size=128

data_stream = Resample(data_stream, scale=initial_scale)
data_stream = Mapping(data_stream, _copy, add_sources=('upsampled',))
data_stream = Resample(data_stream, scale=scale, which_sources=('upsampled',))
data_stream = Resample(data_stream, scale=1/scale, which_sources=('upsampled',))
data_stream = Mapping(data_stream, _equalize_size)

epoch_iterator = data_stream.get_epoch_iterator()

raw_audio_std, upsampled_audio = next(epoch_iterator)

for i in xrange(n_iter-1):
    x_tr,y_tr = next(epoch_iterator)
    raw_audio_std = numpy.hstack([raw_audio_std, x_tr])
    upsampled_audio = numpy.hstack([upsampled_audio, y_tr])

for i,(original_, upsampled_) in enumerate(
                                zip(raw_audio_std, upsampled_audio)[:n_samples]):

    f, (ax1, ax2) = pyplot.subplots(2, sharex=True, sharey=True)
    ax1.plot(original_)
    ax2.plot(upsampled_)
    f.subplots_adjust(hspace=0)
    f.savefig(save_dir + "comparison_upsample_%i.png" % i)
    pyplot.close()

real_residual = raw_audio_std - upsampled_audio
rate = 16000

upsampled_audio_std = upsampled_audio*data_std + data_mean

for i, sample in enumerate(upsampled_audio_std[:n_samples]):
    wavfile.write(save_dir + "upsampled_{}.wav".format(i),
        rate, sample.astype('int16'))

upsampled = _segment_axis((upsampled_audio,))[0]
upsampled = _transpose((upsampled,))[0]

# emit = generator.generate(
#   attended=mlp_context.apply(context),
#   n_steps=context.shape[0],
#   batch_size=context.shape[1],
#   iterate=True
#   )[-4]
ipdb.set_trace()

pl = main_loop.model.get_top_bricks()[0]
from theano import function, tensor

x = tensor.tensor3('residual')
context = tensor.tensor3('upsampled')

x_g = pl.mlp_x.apply(context)
inputs = pl.fork.apply(x_g, as_dict = True)
h = pl.transition.apply(**inputs)

predict = pl.gmm_emitter.emit(h[-1])
predict = function([context],predict)

#del main_loop
#ipdb.set_trace()

residuals = predict(upsampled[:,:n_samples,:])
residuals = _transpose((residuals,))[0]
residuals = numpy.array([x.flatten() for x in residuals])

for i,(real_x, predict_x) in enumerate(zip(real_residual, residuals)[:n_samples]):
    
    f, (ax1, ax2) = pyplot.subplots(2, sharex=True, sharey=True)
    ax1.plot(real_x)
    ax2.plot(predict_x)
    f.subplots_adjust(hspace=0)
    f.savefig(save_dir + "residuals_%i.png" % i)
    pyplot.close()

    audio = real_x * data_std

    wavfile.write(save_dir + "real_residual_{}.wav".format(i),
        rate, audio.astype('int16'))

    audio = predict_x * data_std

    wavfile.write(save_dir + "predicted_residual_{}.wav".format(i),
        rate, audio.astype('int16'))

residuals = predict(upsampled[:,:n_samples,:])
reconstructed = upsampled[:,:n_samples,:] + residuals
reconstructed = _transpose((reconstructed,))[0]
reconstructed = numpy.array([x.flatten() for x in reconstructed])
reconstructed_std = reconstructed*data_std + data_mean

for i, sample in enumerate(reconstructed_std[:n_samples]):
    pyplot.plot(sample)
    pyplot.savefig(save_dir + "reconstructed_with_l0_%i.png" % i)
    pyplot.close()

    wavfile.write(save_dir + "reconstructed_with_l0_{}.wav".format(i),
        rate, sample.astype('int16'))
