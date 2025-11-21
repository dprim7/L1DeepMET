"""Unit tests for model architectures."""

import pytest # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from typing import Dict, Any

from l1deepmet.models.dense import Dense
from l1deepmet.config import Config


class TestDenseModel:
    """Tests for Dense model."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration for Dense model."""
        config_dict = {
            "model": {
                "type": "dense",
                "units": [64, 32, 16],
                "activation": "tanh",
                "with_bias": False
            },
            "training": {
                "mode": 1  # Per-particle weight mode
            },
            "data": {
                "maxNPF": 128
            }
        }
        return config_dict
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample input data."""
        batch_size = 8
        n_candidates = 128
        
        continuous = np.random.randn(batch_size, n_candidates, 5).astype(np.float32)
        momentum = np.random.randn(batch_size, n_candidates, 2).astype(np.float32)
        cat0 = np.random.randint(0, 6, (batch_size, n_candidates)).astype(np.float32)
        cat1 = np.random.randint(0, 4, (batch_size, n_candidates)).astype(np.float32)
        
        return {
            'continuous_inputs': continuous,
            'momentum_inputs': momentum,
            'cat0': cat0,
            'cat1': cat1
        }
    
    def test_model_initialization(self, config):
        """Test that model initializes correctly."""
        model = Dense(config=config)
        assert model is not None
        assert hasattr(model, 'units')
        assert model.units == [64, 32, 16]
    
    def test_model_call_mode1(self, config, sample_inputs):
        """Test model forward pass in mode 1."""
        config["training"]["mode"] = 1
        model = Dense(config=config)
        
        output = model(sample_inputs, training=False)
        
        # Check output shape (batch_size, 2) for MET px, py
        assert output.shape == (sample_inputs['continuous_inputs'].shape[0], 2)
        
        # Check no NaN or Inf values
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
    
    def test_model_call_mode0(self, config, sample_inputs):
        """Test model forward pass in mode 0."""
        config["training"]["mode"] = 0
        model = Dense(config=config)
        
        output = model(sample_inputs, training=False)
        
        # Check output shape
        assert output.shape == (sample_inputs['continuous_inputs'].shape[0], 2)
        assert not tf.reduce_any(tf.math.is_nan(output))
    
    def test_model_call_mode2(self, config, sample_inputs):
        """Test model forward pass in mode 2."""
        config["training"]["mode"] = 2
        model = Dense(config=config)
        
        output = model(sample_inputs, training=False)
        
        # Check output shape
        assert output.shape == (sample_inputs['continuous_inputs'].shape[0], 2)
        assert not tf.reduce_any(tf.math.is_nan(output))
    
    def test_model_training_mode(self, config, sample_inputs):
        """Test model behavior in training mode."""
        model = Dense(config=config)
        
        output_train = model(sample_inputs, training=True)
        output_test = model(sample_inputs, training=False)
        
        # Both should produce valid outputs
        assert output_train.shape == output_test.shape
        assert not tf.reduce_any(tf.math.is_nan(output_train))
        assert not tf.reduce_any(tf.math.is_nan(output_test))
    
    def test_model_get_config(self, config):
        """Test get_config method."""
        model = Dense(config=config)
        model_config = model.get_config()
        
        assert isinstance(model_config, dict)
        assert 'name' in model_config
        assert model_config['name'] == 'DenseModel'
    
    def test_model_with_bias(self, config, sample_inputs):
        """Test model with bias enabled."""
        config["model"]["with_bias"] = True
        model = Dense(config=config)
        
        output = model(sample_inputs, training=False)
        
        assert output.shape == (sample_inputs['continuous_inputs'].shape[0], 2)
        assert not tf.reduce_any(tf.math.is_nan(output))
    
    def test_model_different_units(self, sample_inputs):
        """Test model with different layer configurations."""
        configs = [
            {"model": {"units": [32]}, "training": {"mode": 1}, "data": {"maxNPF": 128}},
            {"model": {"units": [128, 64, 32]}, "training": {"mode": 1}, "data": {"maxNPF": 128}},
            {"model": {"units": [256, 128, 64, 32]}, "training": {"mode": 1}, "data": {"maxNPF": 128}},
        ]
        
        for cfg in configs:
            model = Dense(config=cfg)
            output = model(sample_inputs, training=False)
            assert output.shape == (sample_inputs['continuous_inputs'].shape[0], 2)
    
    def test_model_different_activations(self, sample_inputs):
        """Test model with different activation functions."""
        activations = ["tanh", "relu", "elu"]
        
        for act in activations:
            config = {
                "model": {"units": [64, 32], "activation": act},
                "training": {"mode": 1},
                "data": {"maxNPF": 128}
            }
            model = Dense(config=config)
            output = model(sample_inputs, training=False)
            assert output.shape == (sample_inputs['continuous_inputs'].shape[0], 2)
            assert not tf.reduce_any(tf.math.is_nan(output))
