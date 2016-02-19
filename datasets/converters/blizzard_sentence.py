# Save the segmented blizzard 80h in a hdf5 file.

from scipy.io import wavfile
import os
from multiprocessing import Process, Queue
from play.utils import chunkIt

import hdf55py
import fnmatch
import numpy

# import ipdb

from fuel.datasets.hdf5 import H5PYDataset

save_path = '/data/lisatmp4/sotelo/data/blizzard/'
#data_path = '/data/lisatmp3/mehris/blizzard2013_train/train/unsegmented_wav/'
file_name = "raw_blizzard_80h.hdf5"
hdf5_path = os.path.join(save_path, file_name)

h5file = h5py.File(hdf5_path, mode='w')

info_file = '/data/lisatmp3/mehris/blizzard2013_train/train/blizzard_80hr/kaldi-data/wav.scp'

with open(info_file) as f:
    file_list = f.readlines()
file_list = [x.strip().split(' ') for x in file_list]
name_list, file_list = zip(*file_list)
name_list = list(name_list)
file_list = list(file_list)

transcripts_file = '/data/lisatmp3/mehris/blizzard2013_train/train/blizzard_80hr/kaldi-data/text'

with open(transcripts_file) as f:
    transcripts_list = f.readlines()

transcripts_list = [x.strip().split(' ', 1)[1] for x in transcripts_list]

all_chars = ([chr(ord('A') + i) for i in range(26)] + [' ', '<UNK>'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}
unk_char = '<UNK>'

transcripts_list = [numpy.array([char2code.get(x, char2code[unk_char])
                    for x in transcript]) for transcript in transcripts_list]

phonemes_dir = "/data/lisatmp3/mehris/blizzard2013_train/train/forced_alignments/blizzard_80h/fine_tuned_80hr"
phonemes_files = [phonemes_dir+"/"+str(i+1)+'.ctm' for i in range(10)]

all_phonemes = []
for phon_file in phonemes_files:
    with open(phon_file) as f:
        all_phonemes.extend(f.readlines())

raw_data = [x.strip().split(" ") for x in all_phonemes]
raw_data = [[file_name, int(16000*float(duration)), int(phoneme)] 
            for file_name, _1, start_time, duration, phoneme in raw_data]

audio_files = []
all_phonemes = []
all_durations = []

for file_name, duration, phoneme in raw_data:
    if len(audio_files) == 0:
        audio_files.append(file_name)
        phonemes = []
        durations = []
    if file_name == audio_files[-1]:
        phonemes.append(phoneme)
        durations.append(duration)
    else:
        audio_files.append(file_name)
        all_phonemes.append(numpy.array(phonemes))
        all_durations.append(numpy.array(durations))
        phonemes = [phoneme]
        durations = [duration]
all_phonemes.append(numpy.array(phonemes))
all_durations.append(numpy.array(durations))

#compare order with audio_files (from phonemes) against the one
#coming from the wav files

#names = [x.split(' ')[0] for x in all_phonemes]
#There are 200 out of 36412 files missing :(  ... We have to verify at some point that
#this is not causing some problems when stitching the files together
file_list = [file_list[i] for i,x in enumerate(name_list) if x in audio_files]
transcripts_list = [transcripts_list[i] for i,x in enumerate(name_list) if x in audio_files]

# file_list = []
# for root, dirnames, filenames in os.walk(data_path):
#     for filename in fnmatch.filter(filenames, '*.wav'):
#         file_list.append(os.path.join(root, filename))

# file_list = sorted(file_list)

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

phonemes = h5file.create_dataset(
            'phonemes', (len(file_list),),
            dtype=h5py.special_dtype(vlen=numpy.dtype('int16')))

transcripts = h5file.create_dataset(
            'transcripts', (len(file_list),),
            dtype=h5py.special_dtype(vlen=numpy.dtype('int16')))

transcripts[...] = transcripts_list

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
        all_durations[cont][-1] += result.size - all_durations[cont].sum()
        this_phon = [[phon]*dur for dur,phon in zip(all_durations[cont],all_phonemes[cont])]
        this_phon = [x for y in this_phon for x in y]
        phonemes[cont] = numpy.array(this_phon, dtype = 'int16')

        assert len(result) == len(this_phon)
        cont += 1

# print len(all_results)
# features[...] = all_results
features.dims[0].label = 'batch'
phonemes.dims[0].label = 'batch'
transcripts.dims[0].label = 'batch'

split_dict = {
    'all': {'features': (0, len(file_list)),
            'phonemes': (0, len(file_list)),
            'transcripts': (0, len(file_list))}
    }

h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

# print len(all_results)

h5file.flush()
h5file.close()
