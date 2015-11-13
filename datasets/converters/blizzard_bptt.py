import h5py
import os
import numpy
from fuel.datasets.hdf5 import H5PYDataset

data_path = '/data/lisatmp4/sotelo/data/blizzard/'
file_name = "raw_blizzard.hdf5"

std_file = data_path + 'blizzard' + '_standardize.npz'
len_file = data_path + 'blizzard' + '_length.npy'
hdf5_path = os.path.join(data_path, file_name)

h5file = h5py.File(hdf5_path, mode='r')

big_array = numpy.array([])

no_files = 20
for i in xrange(20):
    x = h5file['features'][i]
    print (i, no_files)
    big_array = numpy.hstack([big_array, x])

data_mean = big_array.mean()
data_std = big_array.std()

# Save mean and std.
numpy.savez(std_file, data_mean=data_mean, data_std=data_std)

len_files = []
for i, x in enumerate(h5file['features']):
    print (i, len(h5file['features']))
    len_files.append(len(x))

len_files = numpy.array(len_files)
numpy.save(len_file, len_files)

file_name = "tbptt_blizzard_intermediate.hdf5"
hdf5_path = os.path.join(data_path, file_name)

intermediateh5file = h5py.File(hdf5_path, mode='w')

batch_size = 64
row_size = sum(len_files)/batch_size
reiterations = 1
seq_size = 8192*16  # 2**13
frame_size = 128
no_minibatches = row_size/seq_size
no_restarts = no_minibatches/reiterations
row_size = no_restarts*reiterations*seq_size
no_minibatches = row_size/seq_size

features = intermediateh5file.create_dataset(
           'features', (batch_size, row_size),
           dtype=numpy.dtype('int16'))

past_observations = 0
current_file = 0
leftovers = numpy.array([])

for row in xrange(batch_size):
    data_row = numpy.array([])
    data_row = numpy.hstack([data_row, leftovers])

    while len(data_row) < row_size:
        data_row = numpy.hstack([data_row, h5file['features'][current_file]])

        past_observations += len_files[current_file]
        current_file += 1

    leftovers = data_row[row_size:]
    data_row = data_row[:row_size]
    features[row] = data_row
    print (len(data_row), row_size)
    print "Leftovers: ", len(leftovers)

    print "Finished row: ", row
    print "Processed files: ", current_file, " out of: ", len(len_files)

intermediateh5file.flush()
intermediateh5file.close()

h5file.close()

intermediateh5file = h5py.File(hdf5_path, mode='r')

file_name = "tbptt_blizzard.hdf5"
hdf5_path = os.path.join(data_path, file_name)

finalh5file = h5py.File(hdf5_path, mode='w')
features = finalh5file.create_dataset(
           'features',
           (no_minibatches*batch_size, seq_size),
           dtype=numpy.dtype('int16'))

for minibatch in xrange(no_minibatches):
    print (minibatch, no_minibatches)
    idx1 = numpy.arange(seq_size*minibatch, seq_size*(minibatch + 1))
    idx2 = numpy.arange(batch_size*minibatch, batch_size*(minibatch + 1))
    features[idx2, :] = intermediateh5file['features'][:, idx1]

features.dims[0].label = 'batch'
features.dims[1].label = 'time'	

size_valid = int(no_restarts*0.05) * reiterations * batch_size
size_test = int(no_restarts*0.05) * reiterations * batch_size

end_train = no_minibatches*batch_size - size_test - size_valid
end_valid = no_minibatches*batch_size - size_test

split_dict = {
    'train': {'features': (0, end_train)},
    'valid': {'features': (end_train, end_valid)},
    'test': {'features': (end_valid, no_minibatches*batch_size)}
    }

finalh5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

finalh5file.flush()
finalh5file.close()
