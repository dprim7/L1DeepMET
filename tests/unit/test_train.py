"""Unit tests for training utilities."""

import pytest # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
import h5py # type: ignore
from pathlib import Path

from l1deepmet.train import (
    load_h5_data,
    split_preprocessed_features,
    create_callbacks
)
from l1deepmet.config import Config


class TestTrainUtilities:
    """Tests for training utility functions."""
    
    def test_load_and_split_h5_data(self, tmp_path):
        """Test loading and splitting H5 data."""
        # Create test H5 file
        h5_path = tmp_path / "test_data.h5"
        X_test = np.random.randn(100, 128, 9).astype(np.float32)
        Y_test = np.random.randn(100, 2).astype(np.float32)
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('X', data=X_test)
            f.create_dataset('Y', data=Y_test)
        
        # Load and split
        X, Y = load_h5_data(str(h5_path))
        result = split_preprocessed_features(X)
        
        # Verify loading
        assert X.shape == X_test.shape
        assert Y.shape == Y_test.shape
        
        # Verify splitting
        assert 'continuous_inputs' in result
        assert 'momentum_inputs' in result
        assert result['continuous_inputs'].shape == (100, 128, 5)
        assert result['momentum_inputs'].shape == (100, 128, 2)
        np.testing.assert_array_equal(result['continuous_inputs'], X[:, :, 0:5])
        np.testing.assert_array_equal(result['momentum_inputs'], X[:, :, 5:7])
    
    def test_create_callbacks(self, tmp_path):
        """Test callback creation with configuration."""
        config = Config({
            "callbacks": {
                "early_stopping": {"monitor": "val_loss", "patience": 10},
                "reduce_lr": {"monitor": "val_loss", "factor": 0.5, "patience": 5}
            }
        })
        
        callbacks = create_callbacks(tmp_path, config)
        
        # Verify callbacks exist
        assert len(callbacks) == 6  # EarlyStopping, ReduceLR, TerminateOnNaN, CSV, Checkpoint, TensorBoard
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'EarlyStopping' in callback_types
        assert 'ReduceLROnPlateau' in callback_types
        assert tmp_path.exists()

