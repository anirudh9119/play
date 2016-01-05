from play.datasets.blizzard import Blizzard
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

from fuel.schemes import SequentialScheme
from fuel.streams import DataStream


dataset = Blizzard(which_sets = ('test',),
	filename = "mgc_blizzard.hdf5")

batch_size = 1

data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, batch_size))

f0, mgc = next(data_stream.get_epoch_iterator())

pyplot.plot(f0[0][:500])
pyplot.savefig('plot_f0.png')
pyplot.close()

pyplot.plot(mgc[0,:500,3])
pyplot.savefig('plot_mgc.png')
pyplot.close()

pyplot.hist(f0[0,f0[0]>0])
pyplot.savefig('plot_hist_cond.png')
pyplot.close()

pyplot.hist(mgc.reshape(-1))
pyplot.savefig('plot_mgc_hist.png')
pyplot.close()

