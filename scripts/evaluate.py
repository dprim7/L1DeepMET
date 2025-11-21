"""
L1DeepMET Evaluation Script

This script evaluates a trained model:
1. Load trained model
2. Load test data
3. Generate predictions
4. Calculate metrics
5. Save results

Usage:
    # With DVC (recommended):
    dvc repro evaluate
    
    # Direct execution:
    python scripts/evaluate.py --config params.yaml --tag 25Jul8_140X_v0 --model-dir models/25Jul8_140X_v0
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np # type: ignore
import tensorflow as tf # type: ignore

from l1deepmet.config import load_config
from l1deepmet.train import load_h5_data, split_preprocessed_features
from l1deepmet.constants import EPSILON


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics.
    
    Args:
        y_true: True MET values (px, py) of shape (N, 2)
        y_pred: Predicted MET values (px, py) of shape (N, 2)
    
    Returns:
        Dictionary of metrics
    """
    # Component-wise metrics
    mae_px = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
    mae_py = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
    mse_px = np.mean((y_true[:, 0] - y_pred[:, 0]) ** 2)
    mse_py = np.mean((y_true[:, 1] - y_pred[:, 1]) ** 2)
    
    # Calculate MET magnitude
    met_true = np.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2)
    met_pred = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)
    
    # MET magnitude metrics
    mae_met = np.mean(np.abs(met_true - met_pred))
    mse_met = np.mean((met_true - met_pred) ** 2)
    rmse_met = np.sqrt(mse_met)
    
    # Relative metrics
    relative_error = np.abs(met_true - met_pred) / (met_true + EPSILON)
    mean_relative_error = np.mean(relative_error)
    
    # Calculate phi (angle)
    phi_true = np.arctan2(y_true[:, 1], y_true[:, 0])
    phi_pred = np.arctan2(y_pred[:, 1], y_pred[:, 0])
    
    # Phi difference (handle wraparound)
    phi_diff = phi_pred - phi_true
    phi_diff = np.arctan2(np.sin(phi_diff), np.cos(phi_diff))
    mae_phi = np.mean(np.abs(phi_diff))
    
    # Resolution
    resolution = np.std(met_true - met_pred)
    
    metrics = {
        "mae_px": float(mae_px),
        "mae_py": float(mae_py),
        "mse_px": float(mse_px),
        "mse_py": float(mse_py),
        "mae_met": float(mae_met),
        "mse_met": float(mse_met),
        "rmse_met": float(rmse_met),
        "mae_phi": float(mae_phi),
        "mean_relative_error": float(mean_relative_error),
        "resolution": float(resolution),
        "mean_met_true": float(np.mean(met_true)),
        "mean_met_pred": float(np.mean(met_pred)),
    }
    
    return metrics


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="L1DeepMET Evaluation")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--tag", type=str, required=True,
                       help="Data tag (e.g., 25Jul8_140X_v0)")
    parser.add_argument("--data-root", type=str, default="data/preprocessed",
                       help="Root directory containing preprocessed data")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Directory containing trained model")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation",
                       help="Directory for evaluation outputs")
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting L1DeepMET evaluation pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data tag: {args.tag}")
    logger.info(f"Model directory: {args.model_dir}")
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    cfg_obj = load_config(str(config_path))
    cfg = cfg_obj.to_dict()
    
    # Setup paths
    data_dir = Path(args.data_root) / args.tag
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    logger.info("Loading test data...")
    test_path = data_dir / "test.h5"
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    X_test, Y_test = load_h5_data(str(test_path))
    logger.info(f"Test data shape: X={X_test.shape}, Y={Y_test.shape}")
    
    # Split features
    logger.info("Preparing model inputs...")
    test_inputs = split_preprocessed_features(X_test)
    
    # Load model
    logger.info("Loading model...")
    model_path = model_dir / "best_model.keras"
    if not model_path.exists():
        # Try alternative name
        model_path = model_dir / "final_model.keras"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found in {model_dir}")
    
    model = tf.keras.models.load_model(model_path, compile=False)
    logger.info(f"Model loaded from: {model_path}")
    
    # Generate predictions
    logger.info("Generating predictions...")
    Y_pred = model.predict(test_inputs, batch_size=args.batch_size, verbose=1)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(Y_test, Y_pred)
    
    # Print metrics
    logger.info("\n" + "="*50)
    logger.info("EVALUATION METRICS")
    logger.info("="*50)
    for key, value in metrics.items():
        logger.info(f"{key:30s}: {value:.6f}")
    logger.info("="*50 + "\n")
    
    # Save metrics
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Save predictions
    logger.info("Saving predictions...")
    predictions_path = output_dir / "predictions.npz"
    np.savez(
        predictions_path,
        Y_true=Y_test,
        Y_pred=Y_pred,
        X_test=X_test
    )
    logger.info(f"Predictions saved to: {predictions_path}")
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
