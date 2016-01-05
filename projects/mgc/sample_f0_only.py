import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import os

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

def _is_nonzero(data):
    return [1.*(array>0) for array in data]

def _zero_for_unvoiced(data):
    #Multiply by 0 the unvoiced components.
    return tuple([data[0]*data[1],data[1]])

from blocks.serialization import load
from blocks.graph import ComputationGraph

data_dir = os.environ['FUEL_DATA_PATH']
data_dir = os.path.join(data_dir, 'blizzard/', 'sp_standardize.npz')

data_stats = numpy.load(data_dir)
f0_mean = data_stats['f0_mean']
f0_std = data_stats['f0_std']

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'blizzard/')

experiment_name = "f0_only_1"

main_loop = load(save_dir+"pkl/best_"+experiment_name+".pkl")

generator = main_loop.model.get_top_bricks()[0]

steps = 2048
n_samples = 1

sample = ComputationGraph(generator.generate(n_steps=steps, 
    batch_size=n_samples, iterate=True))
sample_fn = sample.get_theano_function()

outputs = sample_fn()[-2]

voiced = outputs[:,:,1]
outputs = outputs[:,:,0]
outputs = outputs*f0_std + f0_mean
outputs = outputs*voiced
outputs = outputs.swapaxes(0,1)

outputs = outputs[0]
pyplot.figure(figsize=(100,15))	
pyplot.plot(outputs,linewidth=3)
pyplot.gca().set_xlim(0,2048)
pyplot.savefig(save_dir+"samples/best_"+experiment_name+"3.png")
pyplot.close()

# from fuel.schemes import SequentialScheme
# from fuel.streams import DataStream
# from play.datasets.blizzard import Blizzard

# batch_size = 5

# dataset = Blizzard(which_sets = ('test',), filename = "sp_blizzard.hdf5")
# data_stream = DataStream.default_stream(
#             dataset, iteration_scheme=SequentialScheme(
#             batch_size*(dataset.num_examples/batch_size), batch_size))

# f0, sp = next(data_stream.get_epoch_iterator())

# sp = sp[0]
# f0 = f0[0]
# f, axarr = pyplot.subplots(2, sharex=True)
# f.set_size_inches(100,35)
# axarr[0].imshow(sp.T)
# #axarr[0].colorbar()
# axarr[0].invert_yaxis()
# axarr[0].set_ylim(0,257)
# axarr[0].set_xlim(0,2048)
# axarr[1].plot(f0,linewidth=3)
# axarr[0].set_adjustable('box-forced')
# axarr[1].set_adjustable('box-forced')
# pyplot.savefig(save_dir+"samples/data_sp.png")
# pyplot.close()
