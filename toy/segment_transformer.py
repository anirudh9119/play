from fuel.transformers import Transformer
import numpy

class SegmentSequence(Transformer):
    """Segments the sequences in a batch.
    This transformer is useful to do tbptt. All the sequences to segment
    should have the time dimension as their first dimension.
    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    seq_size : int
        maximum size of the resulting sequences.
    which_sources : tuple of str, optional
        sequences to segment
    add_flag : bool, optional
        add a flag indicating the beginning of a new sequence.
    flag_name : str, optional
        name of the source for the flag
    """
    def __init__(self, data_stream, seq_size=100, which_sources=None,
                 add_flag=False, flag_name = None, min_size = 10, **kwargs):
        super(SegmentSequence, self).__init__(data_stream=data_stream,
            produces_examples=data_stream.produces_examples,**kwargs)

        if which_sources is None:
            which_sources = data_stream.sources
        self.which_sources = which_sources

        self.seq_size = seq_size
        self.step = 0
        self.data = None
        self.len_data = None
        self.add_flag = add_flag
        self.min_size = min_size

        if flag_name is None:
            flag_name = u"start_flag"

        self.flag_name = flag_name

    @property
    def sources(self):
        return self.data_stream.sources + ((self.flag_name,)
                                           if self.add_flag else ())

    def get_data(self, request = None):
        flag = 0

        if self.data is None:
            self.data = next(self.child_epoch_iterator)
            idx = self.sources.index(self.which_sources[0])
            self.len_data = self.data[idx].shape[0]
            #flag is one in the first cut of sequence

        segmented_data = list(self.data)

        for source in self.which_sources:
            idx = self.sources.index(source)
            # Segment data:
            segmented_data[idx] = self.data[idx][
                            self.step:(self.step+self.seq_size)]

        self.step += self.seq_size
        
        if self.step + self.min_size >= self.len_data:
            self.data = None
            self.len_data = None
            self.step = 1
            flag = 1

        if self.add_flag:
            segmented_data.append(flag)

        return tuple(segmented_data)

if __name__ == "__main__":
    print "hello"
    import ipdb
    from scribe.datasets.handwriting import Handwriting
    from fuel.transformers import Mapping, Padding, FilterSources, ForceFloatX
    from fuel.schemes import SequentialScheme
    from fuel.streams import DataStream

    def _transpose(data):
        return tuple(array.swapaxes(0,1) for array in data)

    batch_size = 10

    dataset = Handwriting(('train',))
    data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            50, batch_size))
    data_stream = FilterSources(data_stream, 
                          sources = ('features',))
    data_stream = Padding(data_stream)
    data_stream = Mapping(data_stream, _transpose)

    epoch = data_stream.get_epoch_iterator()
    for batch in epoch:
        print batch[0].shape

    print "Segmented:"
    data_stream = SegmentSequence(data_stream, add_flag = True)

    epoch = data_stream.get_epoch_iterator()
    for batch in epoch:
        print batch[0].shape, batch[2]

    #ipdb.set_trace()









