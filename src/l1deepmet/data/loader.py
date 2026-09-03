import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import h5py # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore

logger = logging.getLogger(__name__)


def split_preprocessed_features(
    X: np.ndarray,
    continuous_slots: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split preprocessed H5 features into model input format.

    Two layouts are supported, chosen by ``continuous_slots``:

    - **Legacy** (``continuous_slots is None``): the 9-feature layout
      produced by ``load_samples_to_numpy`` / ``preprocess_data``.
      Continuous = slots 0-4 (pt, eta, phi, puppi_weight, hcal_depth);
      momentum = slots 5-6 (px, py); categoricals = slots 7, 8
      (encoded_pdgId, encoded_charge).

    - **Extended** (``continuous_slots`` given): the 26-feature layout
      produced by ``load_samples_to_numpy_extended``. The caller passes
      an explicit ordered list of slot indices to use as continuous
      inputs (e.g. ``[0, 1, 2, 3, 8, 9, ...]``). Momentum is taken from
      slots 4-5 (px, py) and categoricals from slots 6, 7 by convention,
      reflecting the params.yaml ``feature_layout_extended`` ordering.

    Args:
        X: preprocessed features array (n_events, max_pf, n_features)
        continuous_slots: optional list of column indices for the
            continuous input. When None, the legacy slicing applies.

    Returns:
        inputs:      (n_events, max_pf, len(continuous_slots) or 5)
        pxpy:        (n_events, max_pf, 2)
        inputs_cat0: (n_events, max_pf) — encoded pdgId
        inputs_cat1: (n_events, max_pf) — encoded charge
    """
    if continuous_slots is None:
        inputs = X[:, :, 0:5]      # pt, eta, phi, puppi, hcal_depth
        pxpy = X[:, :, 5:7]        # px, py
        inputs_cat0 = X[:, :, 7]   # encoded_pdgId
        inputs_cat1 = X[:, :, 8]   # encoded_charge
    else:
        # Extended layout — explicit continuous slot list; momentum and
        # categoricals at the fixed extended positions.
        inputs = X[:, :, list(continuous_slots)]
        pxpy = X[:, :, 4:6]
        inputs_cat0 = X[:, :, 6]
        inputs_cat1 = X[:, :, 7]
    return inputs, pxpy, inputs_cat0, inputs_cat1


class H5DataLoader:

    
    def __init__(self, data_dir: Union[str, Path]):
       
        self.data_dir = Path(data_dir)
        self.metadata = {}
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata from train.h5 file."""
        train_file = self.data_dir / "train.h5"
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        
        with h5py.File(train_file, 'r') as f:
            # Load metadata attributes
            for key in f.attrs.keys():
                self.metadata[key] = f.attrs[key]
        
        logger.info(f"Loaded metadata: {self.metadata}")
    
    def load_data(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features and targets for specified split.
        
        Args:
            split: One of 'train', 'val', 'test'
            
        Returns:
            features: (n_events, max_pf, n_features) preprocessed features
            targets: (n_events, 2) target px, py values
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split must be one of ['train', 'val', 'test'], got {split}")
        
        file_path = self.data_dir / f"{split}.h5"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading {split} data from {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            features = f['features'][:]
            targets = f['targets'][:]
        
        logger.info(f"Loaded {split}: features {features.shape}, targets {targets.shape}")
        return features, targets
    
    def load_split_data(
        self,
        split: str = "train",
        continuous_slots: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and split features into model input format.

        Args:
            split: One of 'train', 'val', 'test'
            continuous_slots: optional explicit slot list for the
                extended layout — see ``split_preprocessed_features``.

        Returns:
            inputs: (n_events, max_pf, K) continuous features
                    (K=5 for legacy, K=len(continuous_slots) for extended)
            pxpy: (n_events, max_pf, 2) momentum components
            inputs_cat0: (n_events, max_pf) encoded pdgId
            inputs_cat1: (n_events, max_pf) encoded charge
            targets: (n_events, 2) target px, py values
        """
        features, targets = self.load_data(split)
        inputs, pxpy, inputs_cat0, inputs_cat1 = split_preprocessed_features(
            features, continuous_slots=continuous_slots
        )
        return inputs, pxpy, inputs_cat0, inputs_cat1, targets

    def create_tf_dataset(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        split_features: bool = True,
        normfac: float = 1.0,
        continuous_slots: Optional[list] = None,
    ) -> tf.data.Dataset:
        """
        Args:
            split: One of 'train', 'val', 'test'
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            split_features: If True, split features into model inputs
            normfac: Divide targets by this value to normalize MET scale (e.g. 100.0).
                     Predictions from the model will be in units of normfac GeV.

        Returns:
            tf.data.Dataset ready for training
        """
        if split_features:
            inputs, pxpy, inputs_cat0, inputs_cat1, targets = self.load_split_data(
                split, continuous_slots=continuous_slots
            )

            if normfac != 1.0:
                targets = targets / normfac

            # Create dataset with multiple inputs
            dataset = tf.data.Dataset.from_tensor_slices({
                'continuous_inputs': inputs,
                'momentum_inputs': pxpy,
                'pdgid_inputs': inputs_cat0,
                'charge_inputs': inputs_cat1,
                'targets': targets
            })

            # Map to (inputs, targets) format for training
            # num_parallel_calls=1 to avoid spawning extra threads
            dataset = dataset.map(
                lambda x: (
                    {
                        'continuous_inputs': x['continuous_inputs'],
                        'momentum_inputs': x['momentum_inputs'],
                        'pdgid_inputs': x['pdgid_inputs'],
                        'charge_inputs': x['charge_inputs']
                    },
                    x['targets']
                ),
                num_parallel_calls=1,
            )
        else:
            features, targets = self.load_data(split)
            if normfac != 1.0:
                targets = targets / normfac
            dataset = tf.data.Dataset.from_tensor_slices((features, targets))
        
        if shuffle and split == "train":
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(batch_size)
        # Use prefetch(1) instead of AUTOTUNE to limit thread creation
        # on thread-constrained shared systems.
        # Also set threading options to minimize private threadpool size.
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options)
        dataset = dataset.prefetch(1)
        
        logger.info(f"Created TensorFlow dataset for {split}: batch_size={batch_size}, shuffle={shuffle}")
        return dataset
    
    def get_data_info(self) -> Dict:
        info = {
            'metadata': self.metadata,
            'data_dir': str(self.data_dir),
            'files': {}
        }
        
        for split in ['train', 'val', 'test']:
            file_path = self.data_dir / f"{split}.h5"
            if file_path.exists():
                with h5py.File(file_path, 'r') as f:
                    info['files'][split] = {
                        'features_shape': f['features'].shape,
                        'targets_shape': f['targets'].shape,
                        'file_size_mb': file_path.stat().st_size / 1024**2
                    }
        
        return info


# Convenience function for quick loading
def load_data(data_dir: Union[str, Path], split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        data_dir: Directory containing H5 files
        split: One of 'train', 'val', 'test'
        
    Returns:
        features: (n_events, max_pf, 9) preprocessed features
        targets: (n_events, 2) target px, py values
    """
    loader = H5DataLoader(data_dir)
    return loader.load_data(split) 
        