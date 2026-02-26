import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import h5py # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore

logger = logging.getLogger(__name__)


def split_preprocessed_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split preprocessed H5 features into model input format.
    
    Args:
        X: preprocessed features array (n_events, max_pf, n_features)
        
    Returns:
        inputs: (n_events, max_pf, 5) - continuous features [pt, eta, phi, puppi, hcal_depth]
        pxpy: (n_events, max_pf, 2) - momentum components [px, py]
        inputs_cat0: (n_events, max_pf) - encoded pdgId
        inputs_cat1: (n_events, max_pf) - encoded charge
    """
    inputs = X[:, :, 0:5]      # pt, eta, phi, puppi, hcal_depth
    pxpy = X[:, :, 5:7]        # px, py
    inputs_cat0 = X[:, :, 7]   # encoded_pdgId
    inputs_cat1 = X[:, :, 8]   # encoded_charge
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
    
    def load_split_data(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split features into model input format.
        
        Args:
            split: One of 'train', 'val', 'test'
            
        Returns:
            inputs: (n_events, max_pf, 5) continuous features
            pxpy: (n_events, max_pf, 2) momentum components
            inputs_cat0: (n_events, max_pf) encoded pdgId
            inputs_cat1: (n_events, max_pf) encoded charge
            targets: (n_events, 2) target px, py values
        """
        features, targets = self.load_data(split)
        inputs, pxpy, inputs_cat0, inputs_cat1 = split_preprocessed_features(features)
        return inputs, pxpy, inputs_cat0, inputs_cat1, targets
    
    def create_tf_dataset(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        split_features: bool = True,
        normfac: float = 1.0,
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
            inputs, pxpy, inputs_cat0, inputs_cat1, targets = self.load_split_data(split)

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
            dataset = dataset.map(
                lambda x: (
                    {
                        'continuous_inputs': x['continuous_inputs'],
                        'momentum_inputs': x['momentum_inputs'],
                        'pdgid_inputs': x['pdgid_inputs'],
                        'charge_inputs': x['charge_inputs']
                    },
                    x['targets']
                )
            )
        else:
            features, targets = self.load_data(split)
            if normfac != 1.0:
                targets = targets / normfac
            dataset = tf.data.Dataset.from_tensor_slices((features, targets))
        
        if shuffle and split == "train":
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
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
        