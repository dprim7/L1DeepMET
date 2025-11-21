"""Unit tests for training utilities."""

import pytest # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
import h5py # type: ignore
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from l1deepmet.train import (
    load_h5_data,
    split_preprocessed_features,
    create_callbacks
)
from l1deepmet.config import Config


class TestLoadH5Data:
    """Tests for load_h5_data function."""
    
    def test_load_h5_data(self, tmp_path):
        """Test loading data from H5 file."""
        # Create test H5 file
        h5_path = tmp_path / "test_data.h5"
        X_test = np.random.randn(100, 128, 9).astype(np.float32)
        Y_test = np.random.randn(100, 2).astype(np.float32)
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('X', data=X_test)
            f.create_dataset('Y', data=Y_test)
        
        # Load data
        X, Y = load_h5_data(str(h5_path))
        
        # Check shapes and values
        assert X.shape == X_test.shape
        assert Y.shape == Y_test.shape
        np.testing.assert_array_almost_equal(X, X_test)
        np.testing.assert_array_almost_equal(Y, Y_test)


class TestSplitPreprocessedFeatures:
    """Tests for split_preprocessed_features function."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature array."""
        batch_size = 32
        n_candidates = 128
        n_features = 9
        return np.random.randn(batch_size, n_candidates, n_features).astype(np.float32)
    
    def test_split_features(self, sample_features):
        """Test feature splitting."""
        result = split_preprocessed_features(sample_features)
        
        # Check keys
        assert 'continuous_inputs' in result
        assert 'momentum_inputs' in result
        assert 'cat0' in result
        assert 'cat1' in result
        
        # Check shapes
        batch_size, n_candidates, _ = sample_features.shape
        assert result['continuous_inputs'].shape == (batch_size, n_candidates, 5)
        assert result['momentum_inputs'].shape == (batch_size, n_candidates, 2)
        assert result['cat0'].shape == (batch_size, n_candidates)
        assert result['cat1'].shape == (batch_size, n_candidates)
    
    def test_split_features_values(self, sample_features):
        """Test that split preserves correct values."""
        result = split_preprocessed_features(sample_features)
        
        # Check that continuous features are correctly extracted
        np.testing.assert_array_equal(
            result['continuous_inputs'],
            sample_features[:, :, 0:5]
        )
        
        # Check that momentum features are correctly extracted
        np.testing.assert_array_equal(
            result['momentum_inputs'],
            sample_features[:, :, 5:7]
        )
        
        # Check categorical features
        np.testing.assert_array_equal(
            result['cat0'],
            sample_features[:, :, 7]
        )
        np.testing.assert_array_equal(
            result['cat1'],
            sample_features[:, :, 8]
        )


class TestCreateCallbacks:
    """Tests for create_callbacks function."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config_dict = {
            "callbacks": {
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": 10
                },
                "reduce_lr": {
                    "monitor": "val_loss",
                    "factor": 0.5,
                    "patience": 5,
                    "min_delta": 0.0001,
                    "min_lr": 0.000001
                }
            }
        }
        return Config(config_dict)
    
    def test_create_callbacks(self, config, tmp_path):
        """Test callback creation."""
        callbacks = create_callbacks(tmp_path, config)
        
        # Check that we have callbacks
        assert len(callbacks) > 0
        
        # Check for expected callback types
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'EarlyStopping' in callback_types
        assert 'ReduceLROnPlateau' in callback_types
        assert 'TerminateOnNaN' in callback_types
        assert 'CSVLogger' in callback_types
        assert 'ModelCheckpoint' in callback_types
        assert 'TensorBoard' in callback_types
    
    def test_callback_output_dir(self, config, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "test_output"
        callbacks = create_callbacks(output_dir, config)
        
        # Check that directory was created
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_early_stopping_config(self, config, tmp_path):
        """Test early stopping callback configuration."""
        callbacks = create_callbacks(tmp_path, config)
        
        # Find early stopping callback
        early_stopping = None
        for cb in callbacks:
            if isinstance(cb, tf.keras.callbacks.EarlyStopping):
                early_stopping = cb
                break
        
        assert early_stopping is not None
        assert early_stopping.monitor == "val_loss"
        assert early_stopping.patience == 10
    
    def test_reduce_lr_config(self, config, tmp_path):
        """Test reduce LR callback configuration."""
        callbacks = create_callbacks(tmp_path, config)
        
        # Find reduce LR callback
        reduce_lr = None
        for cb in callbacks:
            if isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau):
                reduce_lr = cb
                break
        
        assert reduce_lr is not None
        assert reduce_lr.monitor == "val_loss"
        assert reduce_lr.factor == 0.5
        assert reduce_lr.patience == 5
