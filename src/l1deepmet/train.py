"""Training utilities for L1DeepMET models."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import tensorflow as tf # type: ignore
import keras # type: ignore
import h5py # type: ignore
import numpy as np # type: ignore

from l1deepmet.config import Config


def load_h5_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and targets from H5 file."""
    with h5py.File(filepath, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
    return X, Y


def split_preprocessed_features(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Split preprocessed H5 features into model input format.
    
    Args:
        X: Input array of shape (batch, n_candidates, n_features)
           Features are: [pt, eta, phi, puppi, hcal_depth, px, py, encoded_pdgId, encoded_charge]
    
    Returns:
        Dictionary with keys: 'continuous_inputs', 'momentum_inputs', 'cat0', 'cat1'
    """
    continuous = X[:, :, 0:5]   # pt, eta, phi, puppi, hcal_depth
    momentum = X[:, :, 5:7]     # px, py
    cat0 = X[:, :, 7]           # encoded_pdgId
    cat1 = X[:, :, 8]           # encoded_charge
    
    return {
        'continuous_inputs': continuous,
        'momentum_inputs': momentum,
        'cat0': cat0,
        'cat1': cat1
    }


def create_callbacks(
        output_dir: Path,
        config: Config,
) -> list:
    """Create training callbacks based on configuration."""
    callbacks = []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping
    early_stopping_config = config.get("callbacks.early_stopping", {})
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=early_stopping_config.get("monitor", "val_loss"),
        patience=early_stopping_config.get("patience", 40),
        verbose=1,
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)

    # Learning rate reduction on plateau
    reduce_lr_config = config.get("callbacks.reduce_lr", {})
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=reduce_lr_config.get("monitor", "val_loss"),
        factor=reduce_lr_config.get("factor", 0.5),
        patience=reduce_lr_config.get("patience", 10),
        verbose=1,
        mode="auto",
        min_delta=reduce_lr_config.get("min_delta", 0.0001),
        cooldown=0,
        min_lr=reduce_lr_config.get("min_lr", 0.000001),
    )
    callbacks.append(reduce_lr)

    # Terminate on NaN
    stop_on_nan = tf.keras.callbacks.TerminateOnNaN()
    callbacks.append(stop_on_nan)

    # CSV Logger
    csv_logger = tf.keras.callbacks.CSVLogger(
        str(output_dir / "loss_history.csv")
    )
    callbacks.append(csv_logger)

    # Model Checkpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(output_dir / "best_model.keras"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="auto",
    )
    callbacks.append(model_checkpoint)

    # TensorBoard
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(output_dir / "tensorboard_logs"),
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch",
        profile_batch=0,
    )
    callbacks.append(tensorboard)

    return callbacks
