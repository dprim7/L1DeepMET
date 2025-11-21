"""Unit tests for model architectures."""

import pytest # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore

from l1deepmet.models.dense import Dense


class TestDenseModel:
    """Tests for Dense model."""
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample input data."""
        batch_size = 8
        n_candidates = 128
        continuous = np.random.randn(batch_size, n_candidates, 5).astype(np.float32)
        momentum = np.random.randn(batch_size, n_candidates, 2).astype(np.float32)
        return {'continuous_inputs': continuous, 'momentum_inputs': momentum}
    
    @pytest.mark.parametrize("mode,units,activation", [
        (0, [32], "tanh"),
        (1, [64, 32], "relu"),
        (2, [128, 64, 32], "elu"),
    ])
    def test_model_modes_and_architectures(self, sample_inputs, mode, units, activation):
        """Test different training modes and architectures."""
        config = {
            "model": {"units": units, "activation": activation},
            "training": {"mode": mode},
            "data": {"maxNPF": 128}
        }
        model = Dense(config=config)
        output = model(sample_inputs, training=False)
        
        # Verify output
        assert output.shape == (sample_inputs['continuous_inputs'].shape[0], 2)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
    
    def test_model_training_and_inference(self, sample_inputs):
        """Test model in training and inference modes."""
        config = {
            "model": {"units": [32, 16], "activation": "tanh"},
            "training": {"mode": 1},
            "data": {"maxNPF": 128}
        }
        model = Dense(config=config)
        
        output_train = model(sample_inputs, training=True)
        output_test = model(sample_inputs, training=False)
        
        assert output_train.shape == output_test.shape == (8, 2)
        assert not tf.reduce_any(tf.math.is_nan(output_train))
    
    def test_model_serialization(self, sample_inputs):
        """Test model config and serialization."""
        config = {"model": {"units": [32]}, "training": {"mode": 1}, "data": {"maxNPF": 128}}
        model = Dense(config=config)
        
        # Test get_config
        model_config = model.get_config()
        assert 'config' in model_config
        
        # Test from_config
        restored_model = Dense.from_config(model_config)
        assert restored_model.units == model.units
