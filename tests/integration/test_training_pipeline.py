"""Integration test for training pipeline."""

import pytest # type: ignore
import numpy as np # type: ignore
import h5py # type: ignore
import tempfile
from pathlib import Path
import json

from l1deepmet.config import Config
from l1deepmet.train import load_h5_data, split_preprocessed_features, create_callbacks
from l1deepmet.models.dense import Dense
from l1deepmet.losses.corrected import CorrectedCompositeLoss


class TestTrainingPipeline:
    """Integration tests for complete training pipeline."""
    
    @pytest.fixture
    def synthetic_data(self, tmp_path):
        """Create synthetic training and validation data."""
        n_train = 100
        n_val = 50
        n_candidates = 128
        n_features = 9
        
        # Generate random features
        X_train = np.random.randn(n_train, n_candidates, n_features).astype(np.float32)
        X_val = np.random.randn(n_val, n_candidates, n_features).astype(np.float32)
        
        # Generate random targets (px, py)
        Y_train = np.random.randn(n_train, 2).astype(np.float32) * 100
        Y_val = np.random.randn(n_val, 2).astype(np.float32) * 100
        
        # Save to H5 files
        train_path = tmp_path / "train.h5"
        val_path = tmp_path / "val.h5"
        
        with h5py.File(train_path, 'w') as f:
            f.create_dataset('X', data=X_train)
            f.create_dataset('Y', data=Y_train)
        
        with h5py.File(val_path, 'w') as f:
            f.create_dataset('X', data=X_val)
            f.create_dataset('Y', data=Y_val)
        
        return train_path, val_path
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config_dict = {
            "model": {
                "type": "dense",
                "units": [32, 16],
                "activation": "tanh",
                "with_bias": False
            },
            "training": {
                "mode": 1,
                "epochs": 2,
                "batch_size": 32
            },
            "loss": {
                "mae_weight": 1.0,
                "mse_weight": 1.0,
                "binned_weight": 100.0
            },
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001
            },
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
            },
            "data": {
                "maxNPF": 128
            }
        }
        return Config(config_dict)
    
    def test_end_to_end_training(self, synthetic_data, config, tmp_path):
        """Test complete training pipeline from data loading to model training."""
        train_path, val_path = synthetic_data
        output_dir = tmp_path / "output"
        
        # Load data
        X_train, Y_train = load_h5_data(str(train_path))
        X_val, Y_val = load_h5_data(str(val_path))
        
        assert X_train.shape[0] == 100
        assert X_val.shape[0] == 50
        
        # Split features
        train_inputs = split_preprocessed_features(X_train)
        val_inputs = split_preprocessed_features(X_val)
        
        # Build model
        model = Dense(config=config.to_dict())
        
        # Compile model
        loss = CorrectedCompositeLoss(
            mae_weight=1.0,
            mse_weight=1.0,
            binned_weight=100.0
        )
        
        import tensorflow as tf
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['mae', 'mse']
        )
        
        # Create callbacks
        callbacks = create_callbacks(output_dir, config)
        
        # Train model
        history = model.fit(
            train_inputs,
            Y_train,
            validation_data=(val_inputs, Y_val),
            epochs=2,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Check that training completed
        assert len(history.history['loss']) == 2
        assert len(history.history['val_loss']) == 2
        
        # Check that loss decreased or stayed reasonable
        assert history.history['loss'][0] > 0
        assert history.history['val_loss'][0] > 0
        
        # Check that model can make predictions
        predictions = model.predict(val_inputs, verbose=0)
        assert predictions.shape == Y_val.shape
        assert not np.any(np.isnan(predictions))
        
        # Check that output files were created
        assert (output_dir / "best_model.keras").exists()
        assert (output_dir / "loss_history.csv").exists()
    
    def test_model_save_and_load(self, synthetic_data, config, tmp_path):
        """Test that model can be saved and loaded."""
        train_path, val_path = synthetic_data
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Load data
        X_train, Y_train = load_h5_data(str(train_path))
        train_inputs = split_preprocessed_features(X_train)
        
        # Build and compile model
        model = Dense(config=config.to_dict())
        
        loss = CorrectedCompositeLoss()
        
        import tensorflow as tf
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss
        )
        
        # Train briefly
        model.fit(
            train_inputs,
            Y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        # Save model
        model_path = output_dir / "test_model.keras"
        model.save(model_path)
        
        # Load model
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
        
        # Test predictions match
        pred_original = model.predict(train_inputs, verbose=0)
        pred_loaded = loaded_model.predict(train_inputs, verbose=0)
        
        np.testing.assert_array_almost_equal(pred_original, pred_loaded, decimal=5)
