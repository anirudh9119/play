
import numpy as np
import pysptk as SPTK
from scipy.io import wavfile

fs, x = wavfile.read('test.wav')
assert fs == 16000

x = 1.*x #change to float64

from cle.cle.utils import segment_axis

frame_length = 1024
hopsize = 80
noverlap = frame_length - hopsize

frames = segment_axis(x,frame_length, noverlap).astype('float64').T
frames = xw*SPTK.blackman(frame_length).reshape((1024,1))

#frames = frames.T
#frames = frames.copy(order='C')
frames = frames.T

order = 20
alpha = 0.41
stage = 4
gamma = -1.0 / stage

mgc = np.apply_along_axis(SPTK.mgcep, 1, frames, order, alpha, gamma)
mgc_sp = np.apply_along_axis(SPTK.mgc2sp, 1, mgc, alpha, gamma, frame_length).real

mgc_sp_test = np.hstack([mgc_sp,mgc_sp[:,::-1][:,1:-1]])
mgc_sp_test = mgc_sp_test.copy(order = 'C')


# Check in original data, that the processing was good

import h5py
import fuel
from play.datasets.blizzard import Blizzard
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream

dataset_mgc = Blizzard(which_sets = ('train','valid','test'), filename = "mgc_blizzard.hdf5")
dataset = Blizzard(which_sets = ('train','valid','test'))

batch_size = 2

data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, batch_size))

data_stream_mgc = DataStream.default_stream(
            dataset_mgc, iteration_scheme=SequentialScheme(
            dataset_mgc.num_examples, batch_size))

raw = next(data_stream.get_epoch_iterator())[0]
f0, mgc = next(data_stream_mgc.get_epoch_iterator())

from play.utils.mgc import wav2mgcf0

raw = raw[0]
f0 = f0[0]
mgc = mgc[0]

mgc2, f02 = wav2mgcf0(raw)

order=34
frame_window=512
zerofill_width=1024
shift_window=64
pass_const=0.4
min_pitch=20
max_pitch=500
mgcep_gamma=2
e = 0.0012

#-a 0 -s 16


SPTK.rapt(raw.astype(np.float32), fs=16000,
	hopsize=shift_window, min=min_pitch, max=max_pitch, otype="pitch")

f0[1200:1300]


import pipes, os, subprocess, tempfile
import numpy as np

frame_cmd = 'frame -l {} -p {}'.format(frame_window, shift_window)

raw2 = raw.astype('float32')

#raw2 = np.arange(5000).astype('float32')

p = subprocess.Popen(frame_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
stdout, stderr = p.communicate(raw2.tobytes())

framed_raw = np.fromstring(stdout, dtype='float32').reshape((-1,512)).T
framed_raw

# Le tengo q sumar 4
# El algoritmo para hacer frames no es tan bueno, porque tiene algunos de sobre.
# Puede que sea mejor hacerlo con segment_axis

frames = segment_axis(raw,frame_window, frame_window-shift_window).astype('float64').T
frames = frames*SPTK.blackman(frame_window).reshape((frame_window,1))

frames = frames.T
frames = frames.copy(order='C')
# frames = frames.T

order = 34
alpha = 0.4
stage = 2
gamma = -1.0 / stage

first_frame = frames[:,0]
first_frame = frames[300,:]

new_mgc = np.apply_along_axis(SPTK.mgcep, 1, frames, order, alpha, gamma, eps = 0.0012, etype = 1)

test_mgc = new_mgc
test_mgc = mgc.astype('float64').copy(order = 'C')

mgc_sp = np.apply_along_axis(SPTK.mgc2sp, 1, test_mgc, alpha, gamma, frame_window).real

# here is where I can put some noise
# and this is the representation that will be used

mgc_sp_test = np.hstack([mgc_sp,mgc_sp[:,::-1][:,1:-1]])
mgc_sp_test = mgc_sp_test.copy(order = 'C')

mgc_reconstruct = np.apply_along_axis(SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma, eps = 0.0012, etype = 1, itype = 2)

# Check in transformed data



# Transform data









np.apply_along_axis(SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma, itype =2)