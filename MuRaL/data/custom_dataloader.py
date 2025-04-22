import sys
import numpy as np
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from torch.utils.data import DataLoader, Dataset,_DatasetKind
from torch.utils.data.dataloader import  _SingleProcessDataLoaderIter
from torch.utils.data import RandomSampler,BatchSampler, SequentialSampler
from torch.utils.data import _utils

class CombinedDatasetBatch(Dataset):
    """
    Dataset class combining batches of data.
    """
    def __init__(self, sampled_segments,sampled_segments2=0):
        """  
        Initialize the dataset with batched data.
        Args:
            sampled_segments (list): List of batches containing data.
            sampled_segments2 (list, optional): Additional batch of data to concatenate.
        """
        
        self.y = np.concatenate([batch[0] for batch in sampled_segments])
        self.cont_X = np.asarray([0.])

        
        self.cat_X = np.concatenate([batch[2] for batch in sampled_segments])
        self.batch_distal = np.concatenate([batch[3] for batch in sampled_segments])
        if sampled_segments2:
            self.y = np.concatenate([sampled_segments2[0].numpy(), self.y])
            self.cat_X = np.concatenate([sampled_segments2[2].numpy(), self.cat_X])
            self.batch_distal = np.concatenate([sampled_segments2[3].numpy(), self.batch_distal])
        
        self.n = self.y.shape[0]
            
    def __len__(self):
        """ Denote the total number of samples. """
        return self.n

    def __getitem__(self, index):
        """ Generate one batch of data. """
        assert index < self.n
        
        return self.y[index], self.cont_X, self.cat_X[index], self.batch_distal[index]
    

class MyDataLoader(DataLoader):
    def __init__(self, dataset, 
                 batch_size, batch_size2,
                 shuffle = None,shuffle2=None,
                 sampler = None,
                 batch_sampler = None,
                 num_workers = 0,
                 collate_fn = None,
                 pin_memory = False, drop_last1 = False,
                 drop_last2 = False,timeout = 0,
                 worker_init_fn = None,
                 multiprocessing_context = None,
                 generator = None):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, 
                         num_workers, collate_fn, pin_memory, drop_last1, 
                         timeout, worker_init_fn, multiprocessing_context, 
                         generator)
        self.batch_size2 = batch_size2
        self.drop_last2 = drop_last2
        self.shuffle2 = shuffle2

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _MySingleProcessDataLoaderIter(self)
        else:
            print("Error: num_workers greater than 1 is not supported.\
                   please set --per_trial_cpu to 1 or not use --custom_dataloader! ")
            sys.exit()
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

class _MySingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):
    def __init__(self, loader):
        """
        Initialize the custom single-process DataLoader iterator.
        """
        super().__init__(loader)
        self.sampleIter = False
        self._auto_collation = loader._auto_collation
        self.batch_size2 = loader.batch_size2
        self.drop_last2 = loader.drop_last2
        self.sample_drop_last=False 
        self.data = None
        self.shuffle = loader.shuffle2
        if self.shuffle:
            self.Sampler = RandomSampler
        else:
            self.Sampler = SequentialSampler

        self.batch_end_check = False
        if self.drop_last2 == False:
            self.batch_end_check = True 
        
    def _segment_fetch(self, index_sampled_segments):
        """
        Fetch segments based on the provided indices.

        Args:
            possibly_batched_index (list): List of indices to fetch data for.

        Returns:
            list: List of segment samples.
        """
        data = [self._dataset[idx] for idx in index_sampled_segments]
        return data
        
    def _next_index2(self):
        return next(self._sampler_iter2)
    
    @property
    def _index_sampler2(self):
        """
        Get the secondary index sampler.

        Returns:
            BatchSampler or Sampler: The appropriate sampler based on auto collation.
        """
        if self._auto_collation:
            return self.batch_sampler2
        else:
            return self.sampler2


    def get_sampler2(self, sampled_segments): 
        """
        Initialize the sampler for extracting samples from segments.
        Args:
            sampled_segments (list): List of segments used for sampling.
        """
        sampler2 = self.Sampler(sampled_segments)
        batch_sampler2 = BatchSampler(sampler2, self.batch_size2, self.sample_drop_last)
        
        self.sampler2 = sampler2
        self.batch_sampler2 = batch_sampler2
        self._sampler_iter2 = iter(self._index_sampler2)
        self._dataset_fetcher2= _DatasetKind.create_fetcher(
            self._dataset_kind, sampled_segments, self._auto_collation, self._collate_fn, self.sample_drop_last)#dataset is iterable need self.drop_last2
    
    def prepare_next_data(self):
        # init
        if not self.sampleIter:
            index_sampled_segments = self._next_index()
            sampled_segments = self._segment_fetch(index_sampled_segments)
            sampled_segments = CombinedDatasetBatch(sampled_segments)
            self.get_sampler2(sampled_segments)
            sample_index = self._next_index2()
            self.sampleIter = True
        else:
            try:
                sample_index = self._next_index2()
            except StopIteration:  
                # batch drop end
                if self.batch_end_check:
                    try:
                        index_sampled_segments = self._next_index()
                    except:
                        self.batch_end_check = False
                        if not self.data:
                            return self.prepare_next_data()
                        return self.data
                            
                else: 
                    index_sampled_segments = self._next_index()
                
                sampled_segments = self._segment_fetch(index_sampled_segments)
                sampled_segments = CombinedDatasetBatch(sampled_segments, self.data)
                self.data = None
                self.get_sampler2(sampled_segments)
                sample_index = self._next_index2()
              #  print(sample_index)
                
        data = self._dataset_fetcher2.fetch(sample_index)
        if data[0].shape[0] < self.batch_size2:
            self.data = data
            return self.prepare_next_data()
        else:
            return data
    
    def _next_data(self):
        data = self.prepare_next_data()
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data
