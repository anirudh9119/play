import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import os

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

from blocks.serialization import load
from blocks.graph import ComputationGraph

data_dir = os.environ['FUEL_DATA_PATH']
data_dir = os.path.join(data_dir, 'blizzard/', 'sp_standardize.npz')

data_stats = numpy.load(data_dir)
sp_mean = data_stats['sp_mean']
sp_std = data_stats['sp_std']

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'blizzard/')

experiment_name = "sp_only_0"

main_loop = load(save_dir+"pkl/best_"+experiment_name+".pkl")

generator = main_loop.model.get_top_bricks()[0]

steps = 2048
n_samples = 1

sample = ComputationGraph(generator.generate(n_steps=steps, 
    batch_size=n_samples, iterate=True))
sample_fn = sample.get_theano_function()

outputs = sample_fn()[-2]

outputs = outputs*sp_std + sp_mean
outputs = outputs.swapaxes(0,1)
outputs = outputs[0]

print outputs.max(), outputs.min()

pyplot.figure(figsize=(100,15))
pyplot.imshow(outputs.T)
pyplot.colorbar()
pyplot.gca().invert_yaxis()
pyplot.savefig(save_dir+"samples/best_"+experiment_name+"9.png")
pyplot.close()

#Scaling
outputs[outputs>11.866405] = 11.866405
outputs[outputs<-2.0992377] = -2.0992377

pyplot.figure(figsize=(100,15))
pyplot.imshow(outputs.T)
pyplot.colorbar()
pyplot.gca().invert_yaxis()
pyplot.savefig(save_dir+"samples/best_"+experiment_name+"9_scaled.png")
pyplot.close()


from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from play.datasets.blizzard import Blizzard

batch_size = 5

dataset = Blizzard(which_sets = ('test',), filename = "sp_blizzard.hdf5")
data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            batch_size*(dataset.num_examples/batch_size), batch_size))

f0, sp = next(data_stream.get_epoch_iterator())

sp = sp[0]
f0 = f0[0]
f, axarr = pyplot.subplots(2, sharex=True)
f.set_size_inches(100,35)
axarr[0].imshow(sp.T)
#axarr[0].colorbar()
axarr[0].invert_yaxis()
axarr[0].set_ylim(0,257)
axarr[0].set_xlim(0,2048)
axarr[1].plot(f0,linewidth=3)
axarr[0].set_adjustable('box-forced')
axarr[1].set_adjustable('box-forced')
pyplot.savefig(save_dir+"samples/data_sp.png")
pyplot.close()

# Sample wav file

from play.utils.mgc import mgcf02wav
import pysptk as SPTK
from scipy.io import wavfile

order = 34
alpha = 0.4
stage = 2
gamma = -1.0 / stage

mgc_sp = outputs
mgc_sp_test = numpy.hstack([mgc_sp,mgc_sp[:,::-1][:,1:-1]])
mgc_sp_test = mgc_sp_test.astype('float64').copy(order = 'C')

mgc_reconstruct = numpy.apply_along_axis(SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma, eps = 0.0012, etype = 1, itype = 2)

f0, sp = next(data_stream.get_epoch_iterator())

x_synth = mgcf02wav(mgc_reconstruct, f0[2])
x_synth = .95 * x_synth/max(abs(x_synth)) * 2**15
wavfile.write(save_dir+"samples/best_"+experiment_name+"9_scaled.wav", 16000, x_synth.astype('int16'))




# f0, sp = next(data_stream.get_epoch_iterator())
# sp = sp[0]
# f0 = f0[1]

# mgc_sp = sp # For true data
# mgc_sp_test = numpy.hstack([mgc_sp,mgc_sp[:,::-1][:,1:-1]])
# mgc_sp_test = mgc_sp_test.astype('float64').copy(order = 'C')

# mgc_reconstruct = numpy.apply_along_axis(SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma, eps = 0.0012, etype = 1, itype = 2)

# f0, sp = next(data_stream.get_epoch_iterator())

# f0[f0>0] = f0[f0>0].mean()

# x_synth = mgcf02wav(mgc_reconstruct, f0)
# x_synth = .95 * x_synth/max(abs(x_synth)) * 2**15
# wavfile.write(save_dir+"samples/data_sp_wrong_pitch.wav", 16000, x_synth.astype('int16'))
