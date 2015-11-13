from scipy.io import wavfile
import os
from multiprocessing import Process, Queue
from play.utils import chunkIt

import h5py
import fnmatch
import numpy

# import ipdb

from fuel.datasets.hdf5 import H5PYDataset

save_path = '/data/lisatmp4/sotelo/data/blizzard/'
data_path = '/data/lisatmp3/mehris/blizzard2013_train/train/unsegmented_wav/'
file_name = "raw_blizzard.hdf5"
hdf5_path = os.path.join(save_path, file_name)

h5file = h5py.File(hdf5_path, mode='w')

file_list = []
for root, dirnames, filenames in os.walk(data_path):
    for filename in fnmatch.filter(filenames, '*.wav'):
        file_list.append(os.path.join(root, filename))

file_list = sorted(file_list)


def read_data(q, data_files, i):
    # Reads and appends files to a list
    results = []
    for n, f in enumerate(data_files):
        if n % 10 == 0:
            print("Reading file %i of %i" % (n+1, len(data_files)))
        try:
            di = wavfile.read(f)[1]
            if len(di.shape) > 1:
                di = di[:, 0]
            results.append(di)
        except:
            pass
    return q.put((i, results))

n_times = 10
n_process = 8
indx_mp = chunkIt(file_list, n_times)

size_per_iteration = [len(x) for x in indx_mp]
indx_mp = [chunkIt(x, n_process) for x in indx_mp]

size_first_iteration = [len(x) for x in indx_mp[0]]

features = h5file.create_dataset(
            'features', (len(file_list),),
            dtype=h5py.special_dtype(vlen=numpy.dtype('int16')))

cont = 0
for time_step in xrange(n_times):
    print("Time step %i" % (time_step))
    q = Queue()

    process_list = []
    results_list = []

    for i_process in xrange(n_process):
        this_process = Process(
            target=read_data,
            args=(q, indx_mp[time_step][i_process], i_process))
        process_list.append(this_process)
        process_list[i_process].start()

    results_list = [q.get() for i in xrange(n_process)]
    results_list = sorted(results_list, key=lambda x: x[0])
    _, results_list = zip(*results_list)

    results_list = [x for small_list in results_list
                    for x in small_list]

    for result in results_list:
        features[cont] = result
        cont += 1

# print len(all_results)
# features[...] = all_results
features.dims[0].label = 'batch'

split_dict = {
    'all': {'features': (0, len(file_list))}
    }

h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

# print len(all_results)

h5file.flush()
h5file.close()
