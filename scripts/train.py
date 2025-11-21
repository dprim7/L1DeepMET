#!/usr/bin/env python3
"""
L1DeepMET Training Script

This script runs the complete training pipeline:
1. Load configuration
2. Load preprocessed data from H5 files
3. Build model
4. Train model with callbacks
5. Save trained model and metrics

Usage:
    # With DVC (recommended):
    dvc repro train
    
    # Direct execution:
    python scripts/train.py --config params.yaml --tag 25Jul8_140X_v0
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np # type: ignore
import tensorflow as tf # type: ignore

from l1deepmet.config import load_config
from l1deepmet.train import load_h5_data, split_preprocessed_features, create_callbacks
from l1deepmet.models.dense import Dense
from l1deepmet.losses.corrected import CorrectedCompositeLoss


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def build_model(config: Dict[str, Any]) -> tf.keras.Model:
    """Build model from configuration."""
    model_type = config.get("model", {}).get("type", "dense")
    
    if model_type == "dense":
        model = Dense(config=config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def compile_model(model: tf.keras.Model, config: Dict[str, Any]) -> None:
    """Compile model with optimizer and loss."""
    # Get training config
    training_cfg = config.get("training", {})
    optimizer_cfg = config.get("optimizer", {})
    loss_cfg = config.get("loss", {})
    
    # Create optimizer
    learning_rate = optimizer_cfg.get("learning_rate", 0.001)
    optimizer_type = optimizer_cfg.get("type", "adam")
    
    if optimizer_type == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Create loss
    loss = CorrectedCompositeLoss(
        mae_weight=loss_cfg.get("mae_weight", 1.0),
        mse_weight=loss_cfg.get("mse_weight", 1.0),
        binned_weight=loss_cfg.get("binned_weight", 200.0),
    )
    
    # Compile
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mae', 'mse']
    )


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="L1DeepMET Training")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--tag", type=str, required=True,
                       help="Data tag (e.g., 25Jul8_140X_v0)")
    parser.add_argument("--data-root", type=str, default="data/preprocessed",
                       help="Root directory containing preprocessed data")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory for model outputs")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting L1DeepMET training pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data tag: {args.tag}")
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    cfg_obj = load_config(str(config_path))
    cfg = cfg_obj.to_dict()
    
    # Override with CLI arguments
    if args.epochs:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    
    # Get training parameters
    training_cfg = cfg.get("train", cfg.get("training", {}))
    epochs = training_cfg.get("epochs", 100)
    batch_size = training_cfg.get("batch_size", 1024)
    
    logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
    
    # Setup paths
    data_dir = Path(args.data_root) / args.tag
    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading preprocessed data...")
    train_path = data_dir / "train.h5"
    val_path = data_dir / "val.h5"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    X_train, Y_train = load_h5_data(str(train_path))
    X_val, Y_val = load_h5_data(str(val_path))
    
    logger.info(f"Training data shape: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"Validation data shape: X={X_val.shape}, Y={Y_val.shape}")
    
    # Split features
    logger.info("Preparing model inputs...")
    train_inputs = split_preprocessed_features(X_train)
    val_inputs = split_preprocessed_features(X_val)
    
    # Build model
    logger.info("Building model...")
    model = build_model(cfg)
    
    # Compile model
    logger.info("Compiling model...")
    compile_model(model, cfg)
    
    # Create callbacks
    logger.info("Setting up callbacks...")
    callbacks = create_callbacks(output_dir, cfg_obj)
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_inputs,
        Y_train,
        validation_data=(val_inputs, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    logger.info("Saving final model...")
    model.save(output_dir / "final_model.keras")
    
    # Save training history
    logger.info("Saving training metrics...")
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    
    # Summary
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
