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
import ipdb
import pysptk as SPTK

TOTAL_ROWS = 138368

def process_chunk(num_chunk):
    #Total number of rows
    TOTAL_ROWS = 138368
    n_times = 50
    n_process = 7 # briaree
    files_per_batch = 25
    num_files = n_process*n_times*files_per_batch

    #total_chunks = numpy.ceil(TOTAL_ROWS/float(num_files))

    const = num_files*num_chunk
    indx_mp = range(num_files)

    indx_mp = [x + const for x in indx_mp]
    indx_mp = [x for x in indx_mp if x < TOTAL_ROWS]
    num_files = len(indx_mp)
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


def convert_to_spectrum():
    #ipdb.set_trace()
    data_path = os.environ['FUEL_DATA_PATH']
    data_path = os.path.join(data_path,'blizzard/')
    data_name = "mgc_blizzard.hdf5"
    save_name = "sp_blizzard.hdf5"

    save_path = os.path.join(data_path, save_name)
    resulth5 = h5py.File(save_path, mode='w')

    sp_h5 = resulth5.create_dataset(
                'sp', (TOTAL_ROWS, 2048, 257), dtype='float32')
    f0_h5 = resulth5.create_dataset(
                'f0', (TOTAL_ROWS, 2048), dtype='float32')

    means = []
    stds  = []
    cont = 0

    hdf5_path = os.path.join(data_path, data_name)
    h5file = h5py.File(hdf5_path, mode='r')
    mgc = h5file['mgc']
    f0 = h5file['f0']

    #Check next line
    sp_h5[:] =  f0[:]

    import time

    alpha = 0.4
    stage = 2
    gamma = -1.0 / stage
    frame_window = 512

    #TOTAL_ROWS = 5000#138368
    n_times = 100
    n_process = 14 # marge
    #files_per_batch = 100
    #num_files = n_process*n_times*files_per_batch

    #total_chunks = numpy.ceil(TOTAL_ROWS/float(num_files))

    #const = num_files*num_chunk
    indx_mp = range(TOTAL_ROWS)
    indx_mp = chunkIt(indx_mp, n_times)
    indx_mp = [chunkIt(x, n_process) for x in indx_mp]

    def process_batch(q, x, i):
        results = []
        for n, f in enumerate(x):
            if n % 10 == 0:
                print("Reading row %i of %i" % (n+1, len(x)))
            # Check if same results
            f = f.astype('float64').copy(order = 'C')
            results.append(
                numpy.apply_along_axis(SPTK.mgc2sp, 1, f, alpha, gamma, frame_window).real)
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
                args=(q, mgc[indx_mp[time_step][i_process]], i_process))
            process_list.append(this_process)
            process_list[i_process].start()

        results_list = [q.get() for i in xrange(n_process)]
        results_list = sorted(results_list, key=lambda x: x[0])
        _, results_list = zip(*results_list)

        results_list = [x for small_list in results_list
                          for x in small_list]

        # Add to hdf5 file
        for sp in results_list:
            sp_h5[cont] = sp.astype('float32')
            cont += 1

        print "total time: ", (time.time()-total_time)/60.
        sys.stdout.flush()

    cont = TOTAL_ROWS
    end_train = int(.9*cont)
    end_valid = int(.95*cont)
    end_test = cont

    split_dict = {
        'train': {'sp': (0, end_train),
                  'f0': (0, end_train)},
        'valid': {'sp': (end_train, end_valid),
                  'f0': (end_train, end_valid)},
        'test': {'sp': (end_valid, end_test),
                 'f0': (end_valid, end_test)}
        }

    resulth5.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    resulth5.flush()
    resulth5.close()


def compute_std_sp():
    data_path = os.environ['FUEL_DATA_PATH']
    data_path = os.path.join(data_path,'blizzard/')
    data_name = "sp_blizzard.hdf5"

    save_path = os.path.join(data_path, data_name)
    h5 = h5py.File(save_path, mode='r')

    sp_means = []
    sp_stds  = []
    f0_means = []
    f0_stds  = []

    step = 100
    cont = 0
    num_steps = 1000
    for i in range(num_steps):
        print i, num_steps
        sp = h5['sp'][cont:(cont+step)]
        sp_means.append(sp.mean(axis = (0,1)))
        sp_stds.append(sp.std(axis = (0,1)))

        f0 = h5['f0'][cont:(cont+step)]
        f0 = f0[f0>0]

        f0_means.append(f0.mean())
        f0_stds.append(f0.std())

        cont += step

    std_file = os.path.join(data_path, 'sp_standardize.npz')
    sp_mean = numpy.array(sp_means).mean(axis = 0)
    sp_std = numpy.array(sp_stds).mean(axis = 0)

    f0_mean = numpy.array(f0_means).mean(axis = 0)
    f0_std = numpy.array(f0_stds).mean(axis = 0)

    numpy.savez(std_file,
        sp_mean = sp_mean,
        sp_std = sp_std,
        f0_mean = f0_mean,
        f0_std = f0_std)

def paste_chunks():
    data_path = os.environ['FUEL_DATA_PATH']
    data_path = os.path.join(data_path,'blizzard/')
    save_name = "mgc_blizzard.hdf5"
    save_path = os.path.join(data_path, save_name)
    resulth5 = h5py.File(save_path, mode='w')

    mgc_h5 = resulth5.create_dataset(
                'mgc', (TOTAL_ROWS, 2048, 35), dtype='float32')
    f0_h5 = resulth5.create_dataset(
                'f0', (TOTAL_ROWS, 2048), dtype='float32')

    #list_chunks = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    list_chunks = range(16)
    means = []
    stds  = []
    cont = 0
    for num_chunk in list_chunks:
        print "Processing chunk: ", num_chunk
        file_name = "chunk_{}.hdf5".format(num_chunk)
        hdf5_path = os.path.join(data_path, file_name)

        h5file = h5py.File(hdf5_path, mode='r')
        mgc = h5file['mgc']
        f0 = h5file['f0']

        #means.append((mgc[:,:,:].mean(axis=(0,1)), f0[:].mean()))
        #stds.append((mgc[:,:,:].std(axis=(0,1)), f0[:].std()))

        num_files = len(f0)

        if num_chunk == 15:
            num_files = 7118

        mgc_h5[cont:cont+num_files] = mgc[:num_files,:,:]
        f0_h5[cont:cont+num_files] = f0[:num_files,:]

        h5file.close()
        cont += num_files

    end_train = int(.9*cont)
    end_valid = int(.95*cont)
    end_test = cont

    split_dict = {
        'train': {'mgc': (0, end_train),
                  'f0': (0, end_train)},
        'valid': {'mgc': (end_train, end_valid),
                  'f0': (end_train, end_valid)},
        'test': {'mgc': (end_valid, end_test),
                 'f0': (end_valid, end_test)}
        }

    resulth5.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    resulth5.flush()
    resulth5.close()

    std_file = os.path.join(data_path, 'mgc_standardize.npz')
    data_mean = numpy.array(means).mean(axis = 0)
    data_std = numpy.array(stds).mean(axis = 0)

    mgc_mean = data_mean[0]
    mgc_std = data_std[0]

    f0_mean = data_mean[1]
    f0_std = data_std[1]

    numpy.savez(std_file,
        mgc_mean = mgc_mean,
        mgc_std = mgc_std,
        f0_mean = f0_mean,
        f0_std = f0_std)


if __name__ == "__main__":

    if len(sys.argv) > 1:
      num_chunk = int(sys.argv[1])
    else:
      num_chunk = 0

    #num_chunk = 1
    #process_chunk(num_chunk)
    #paste_chunks()
    convert_to_spectrum()
