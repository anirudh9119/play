import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import os
from play.utils.mgc import mgcf02wav
import pysptk as SPTK
from scipy.io import wavfile
from blocks.serialization import load
from blocks.graph import ComputationGraph
import ipdb

from fuel.transformers import (Mapping, FilterSources,
                        ForceFloatX, ScaleAndShift)
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from play.datasets.blizzard import Blizzard
from play.toy.segment_transformer import SegmentSequence

batch_size = 10

order = 34
alpha = 0.4
stage = 2
gamma = -1.0 / stage

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

dataset = Blizzard(which_sets = ('test',), filename = "sp_blizzard.hdf5")
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
data_stream = SegmentSequence(data_stream, 128*2, add_flag=True)
data_stream = ForceFloatX(data_stream)

x_tr = next(data_stream.get_epoch_iterator())

#f0, sp = next(data_stream.get_epoch_iterator())

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'blizzard/')

experiment_name = "sp_and_f0_2_wn"
num_sample = "06_less_var_f0_less_var_sp_conditioned"

main_loop = load(save_dir+"pkl/best_"+experiment_name+".pkl")

generator = main_loop.model.get_top_bricks()[0]

steps = 2048
n_samples = 10

from blocks.utils import shared_floatx_zeros, shared_floatx

hidden_size_recurrent = 2000

states = generator.transition.apply.outputs
states = {name: shared_floatx_zeros((batch_size, hidden_size_recurrent)) for name in states}

from theano import tensor, function
f0 = tensor.matrix('f0')
voiced = tensor.matrix('voiced')
start_flag = tensor.scalar('start_flag')
sp = tensor.tensor3('sp')

f0s = f0.dimshuffle(0,1,'x')
voiceds = voiced.dimshuffle(0,1,'x')
x = tensor.concatenate([sp, f0s, voiceds], 2)
cost_matrix = generator.cost_matrix(x)

cg = ComputationGraph(cost_matrix)

from blocks.filter import VariableFilter
from play.utils import regex_final_value
extra_updates = []
for name, var in states.items():
  update = VariableFilter(theano_name_regex=regex_final_value(name))(cg.auxiliary_variables)[0]
  extra_updates.append((var, update))

#print function([f0, sp, voiced], cost_matrix, updates = extra_updates)(x_tr[0],x_tr[1],x_tr[2])

#generator.generate(n_steps=steps, batch_size=n_samples, iterate=True, **states)

#states = {}
sample = ComputationGraph(generator.generate(n_steps=steps, 
    batch_size=n_samples, iterate=True, **states))
sample_fn = sample.get_theano_function()

outputs_bp = sample_fn()[-2]

for this_sample in range(n_samples):
	print "Iteration: ", this_sample
	outputs = outputs_bp

	sampled_f0 = outputs[:,:,-2]
	sampled_voiced = outputs[:,:,-1]

	print sampled_voiced.mean()
	print sampled_f0.max(), sampled_f0.min()

	outputs = outputs[:,:,:-2]
	outputs = outputs*sp_std + sp_mean
	outputs = outputs.swapaxes(0,1)
	outputs = outputs[this_sample]
	print outputs.max(), outputs.min()

	sampled_f0 = sampled_f0*f0_std + f0_mean
	sampled_f0 = sampled_f0*sampled_voiced
	sampled_f0 = sampled_f0.swapaxes(0,1)
	sampled_f0 = sampled_f0[this_sample]

	print sampled_f0.min(), sampled_f0.max()

	f, axarr = pyplot.subplots(2, sharex=True)
	f.set_size_inches(100,35)
	axarr[0].imshow(outputs.T)
	#axarr[0].colorbar()
	axarr[0].invert_yaxis()
	axarr[0].set_ylim(0,257)
	axarr[0].set_xlim(0,2048)
	axarr[1].plot(sampled_f0,linewidth=3)
	axarr[0].set_adjustable('box-forced')
	axarr[1].set_adjustable('box-forced')
	pyplot.savefig(save_dir+"samples/new/best_"+experiment_name+num_sample+str(this_sample)+".png")
	pyplot.close()

	sampled_f0_corrected = sampled_f0
	sampled_f0_corrected[sampled_f0_corrected<0] = 0.

	mgc_sp = outputs 
	mgc_sp_test = numpy.hstack([mgc_sp,mgc_sp[:,::-1][:,1:-1]])
	mgc_sp_test = mgc_sp_test.astype('float64').copy(order = 'C')

	mgc_reconstruct = numpy.apply_along_axis(SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma, eps = 0.0012, etype = 1, itype = 2)

	x_synth = mgcf02wav(mgc_reconstruct, sampled_f0_corrected)
	x_synth = .95 * x_synth/max(abs(x_synth)) * 2**15
	wavfile.write(save_dir+"samples/new/best_"+experiment_name+num_sample+str(this_sample)+ ".wav", 16000, x_synth.astype('int16'))

	#Scaling
	outputs[outputs>11.866405] = 11.866405
	outputs[outputs<-2.0992377] = -2.0992377

	f, axarr = pyplot.subplots(2, sharex=True)
	f.set_size_inches(100,35)
	axarr[0].imshow(outputs.T)
	#axarr[0].colorbar()
	axarr[0].invert_yaxis()
	axarr[0].set_ylim(0,257)
	axarr[0].set_xlim(0,2048)
	axarr[1].plot(sampled_f0,linewidth=3)
	axarr[0].set_adjustable('box-forced')
	axarr[1].set_adjustable('box-forced')
	pyplot.savefig(save_dir+"samples/new/best_"+experiment_name+num_sample+str(this_sample)+"_scaled.png")
	pyplot.close()

	mgc_sp = outputs 
	mgc_sp_test = numpy.hstack([mgc_sp,mgc_sp[:,::-1][:,1:-1]])
	mgc_sp_test = mgc_sp_test.astype('float64').copy(order = 'C')
	mgc_reconstruct = numpy.apply_along_axis(SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma, eps = 0.0012, etype = 1, itype = 2)
	x_synth = mgcf02wav(mgc_reconstruct, sampled_f0_corrected)
	x_synth = .95 * x_synth/max(abs(x_synth)) * 2**15
	wavfile.write(save_dir+"samples/new/best_"+experiment_name+num_sample+str(this_sample)+ "_scaled.wav", 16000, x_synth.astype('int16'))
