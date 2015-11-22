import os
import ipdb

from fuel import config
from fuel.datasets import H5PYDataset

class Blizzard(H5PYDataset):
    def __init__(self, which_sets, filename = 'tbptt_blizzard.hdf5', **kwargs):
    	self.filename = filename
        super(Blizzard, self).__init__(self.data_path, which_sets, **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path[0], 'blizzard', self.filename)