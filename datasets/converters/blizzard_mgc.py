import h5py
import os
import sys
import numpy
from fuel.datasets.hdf5 import H5PYDataset
from multiprocessing import Process, Queue
from play.utils.mgc import wav2mgcf0
import multiprocessing
import time
from play.utils import chunkIt

#Total number of rows
total_rows = 138368

if len(sys.argv) > 1:
  num_chunk = int(sys.argv[1])
else:
  num_chunk = 0

n_times = 50
n_process = 7 # briaree
files_per_batch = 25
num_files = n_process*n_times*files_per_batch

total_chunks = numpy.ceil(total_rows/float(num_files))

const = num_files*num_chunk
indx_mp = range(num_files)
indx_mp = [x + const for x in indx_mp]
indx_mp = [x for x in indx_mp if x < total_rows]
indx_mp = chunkIt(indx_mp, n_times)
indx_mp = [chunkIt(x, n_process) for x in indx_mp]

data_path = os.environ['FUEL_DATA_PATH']
data_path = os.path.join(data_path,'blizzard/')
file_name = "tbptt_blizzard.hdf5"
save_name = "chunk_{}.hdf5".format(num_chunk)
hdf5_path = os.path.join(data_path, file_name)

save_dir = os.environ['RESULTS_DIR']
if 'blizzard' not in save_dir:
  save_dir = os.path.join(save_dir,'blizzard/')
save_path = os.path.join(save_dir, save_name)
resulth5 = h5py.File(save_path, mode='w')

h5file = h5py.File(hdf5_path, mode='r')
raw_data = h5file['features']

# Prepare output file

#Hardcoded values
mgc_h5 = resulth5.create_dataset(
            'mgc', (num_files, 2048, 35), dtype='float32')
f0_h5 = resulth5.create_dataset(
            'f0', (num_files, 2048), dtype='float32')

def process_batch(q, x, i):
    results = []
    for n, f in enumerate(x):
        if n % 10 == 0:
            print("Reading row %i of %i" % (n+1, len(x)))
        results.append(wav2mgcf0(f))

    return q.put((i, results))

total_time = time.time()
cont = 0
for time_step in xrange(n_times):
    print("Time step %i" % (time_step))
    q = Queue()

    process_list = []
    results_list = []

    for i_process in xrange(n_process):
        this_process = Process(
            target=process_batch,
            args=(q, raw_data[indx_mp[time_step][i_process]], i_process))
        process_list.append(this_process)
        process_list[i_process].start()

    results_list = [q.get() for i in xrange(n_process)]
    results_list = sorted(results_list, key=lambda x: x[0])
    _, results_list = zip(*results_list)

    results_list = [x for small_list in results_list
                      for x in small_list]

    #mgcc, f0 = zip(*results_list)

    # Add to hdf5 file
    for mgc, f0 in results_list:
        mgc_h5[cont] = mgc
        f0_h5[cont] = f0
        cont += 1

    print "total time: ", (time.time()-total_time)/60.
    sys.stdout.flush()

resulth5.flush()
resulth5.close()
h5file.close()
