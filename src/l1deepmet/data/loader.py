import logging
from typing import Dict, List, Tuple, Generator
import tensorflow as tf # type: ignore 
import numpy as np # type: ignore
from data import DataConfig, SampleInfo #TODO: write these classes

logger = logging.getLogger(__name__)

class H5FileManager:
    '''Context manager for handling HDF5 file operations.'''

    def __init__(self, config: DataConfig):
        self.config = config
        self.samples: Dict[str, SampleInfo] = {}
        self._file_info_cache= {}
        self._discover_samples()

    def _discover_samples(self):
        '''Discover all h5 files matching the sample names.'''
        for data_dir in self.config.data_dirs:
            if not data_dir.exists():
                logger.warning(f"Data directory not found: {data_dir}")
                continue
            
            for sample_name in self.config.sample_names:
                pattern = f"*{sample_name}*.h5"
                files = list(data_dir.glob(pattern))

                for file_path in files:
                    sample_key = f"{sample_name}_{file_path.stem}"
                    self.samples[sample_key] = SampleInfo(
                        name=sample_key,
                        path=file_path,
                    )
        logger.info(f"Discovered {len(self.samples)} samples.")
    
    def get_total_events(self) -> int:
        total_events = 0
        for sample in self.samples.values():
            info = self.get_file_info(sample.path)
            total_events += info['n_events']

        logger.info(f"Total events across all samples: {total_events}")
        return total_events
    
    def create_global_index(self) -> List[Tuple[str, int]]:
        global_index = []

        for sample_key, sample in self.samples.items():
            info = self.get_file_info(sample.path)
            n_events = info['n_events']

            for local_idx in range(n_events):
                global_index.append((sample_key, local_idx))
        
        return global_index
    
    # TODO: implement get_file_info, open_file, 


class CachedDataStreamer:
    '''stream data with caching and lazy loading'''

    def __init__(self, 
                 file_manager: H5FileManager,
                 indices: List[Tuple[str, int]],
                 config: DataConfig,
                 batch_size: int = 512,
                 ):
        self.file_manager = file_manager
        self.indices = indices
        self.config = config
        self.batch_size = batch_size
        self._cache = {} #TODO: implement cache

    def create_dataset(self, shuffle: bool = True) -> tf.data.Dataset:
        '''Create a TensorFlow dataset from global indices'''
        
        def data_generator() -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
            indices_to_use = list(self.indices)
            if shuffle:
                np.random.shuffle(indices_to_use)
            
            for file_key, local_idx in indices_to_use:
                with self.file_manager.open_file(file_key) as h5_file:

                    data = h5_file['data'][local_idx].astype(np.float32)
                    labels = h5_file['labels'][local_idx].astype(np.float32)
                    yield data, labels #TODO: check for match with file structure
        
        first_file_key = list(self.file_manager.samples.keys())[0]
        with self.file_manager.open_file(first_file_key) as f:
            data_shape = f['data'].shape[1:]  # remove batch dimension
            label_shape = f[self.config.target].shape[1:] if len(f[self.config.target].shape) > 1 else () #TODO: check for match with file structure

        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=data_shape, dtype=tf.float32),
                tf.TensorSpec(shape=label_shape, dtype=tf.float32)
            )
        )

        dataset = dataset.batch(self.batch_size) #TODO: check diff batch in load and train

        if self.config.normalize:
            dataset = dataset.map(
                lambda x, y: (tf.nn.l2_normalize(x, axis=-1), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
    
        #TODO: create unit tests for both classes 
        