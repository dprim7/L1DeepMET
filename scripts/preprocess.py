#!/usr/bin/env python3
"""
L1DeepMET Data Preprocessing Script

This script runs the complete preprocessing pipeline:
1. Load configuration
2. Load raw ROOT data
3. Select desired number of events per sample
4. Apply preprocessing (outlier removal, sanitization)
5. Combine, shuffle, and split into train/val/test
6. Save to H5 files
7. Generate control plots

Usage:
    # With DVC (recommended):
    dvc repro preprocess
    
    # Direct execution:
    python scripts/preprocess.py --config params.yaml --tag 25Jul8_140X_v0
    python scripts/preprocess.py --config params.yaml --tag 25Jul8_140X_v0 --data-root /path/to/data
"""

import argparse
import logging
import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any
import numpy as np # type: ignore

from l1deepmet.data.preprocessing import (
    load_samples_to_numpy,
    select_events, 
    preprocess_data,
    combine_shuffle_split,
    save_h5_files,
    coerce_encoding
)
from l1deepmet.plotting import control_plots


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load preprocessing configuration from YAML file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    return cfg


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="L1DeepMET Data Preprocessing")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to preprocessing configuration YAML file")
    parser.add_argument("--tag", type=str, required=True,
                       help="Data tag for output directory (e.g., 25Jul8_140X_v0)")
    parser.add_argument("--data-root", type=str, default=None,
                       help="Root directory containing raw data (overrides config)")
    parser.add_argument("--output-root", type=str, default="data/preprocessed",
                       help="Root directory for preprocessed outputs")
    parser.add_argument("--plot-dir", type=str, default="outputs/preprocessing_plots",
                       help="Directory for control plots")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting L1DeepMET preprocessing pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data tag: {args.tag}")
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    cfg = load_config(config_path)
    
   
    # DVC params.yaml format
    preprocess_cfg = cfg["preprocess"]
    samples = preprocess_cfg["samples"]
    var_list = preprocess_cfg["var_list"]
    var_list_mc = preprocess_cfg.get("var_list_mc", [])
    max_pf = int(preprocess_cfg["max_pf_candidates"])
    encoding_raw = preprocess_cfg["discrete_encoding"]
    default_data_root = preprocess_cfg.get("data_root", "/home/users/dprimosc/L1DeepMET/data/25Jul8_140X_v0")
    split_config = preprocess_cfg["train_val_test_split"]
    # Convert DVC format to old format for compatibility
    data_cfg = {
        "train-val-test-split": {
            "train": split_config["train"],
            "val": split_config["val"], 
            "test": split_config["test"]
        }
    }
    
    logger.info(f"Sample configuration: {samples}")
    
    # Get other configuration parameters
    sample_names = list(samples.keys())
    encoding = coerce_encoding(encoding_raw)
    
    # Set data root (command line overrides config)
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(default_data_root)
    
    logger.info(f"Data root: {data_root}")
    
    include_mc = len(var_list_mc) > 0
    
    # Set random seed
    np.random.seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Load raw ROOT data
    logger.info("Loading raw ROOT data...")
    
    results = load_samples_to_numpy(
        data_root=data_root,
        sample_names=sample_names,
        var_list=var_list,
        var_list_mc=var_list_mc,
        max_pf=max_pf,
        encoding=encoding,
        include_mc=include_mc,
        step_size="100 MB",
        dtype=np.float32,
    )
    
    # Select events per sample
    logger.info("Selecting events per sample...")
    
    selected_results = select_events(results, samples)
    
    # Apply preprocessing
    logger.info("Applying preprocessing...")
    
    processed_results = preprocess_data(selected_results)
    
    # Combine and split data
    logger.info("Combining and splitting data...")
    
    X_train, X_val, X_test, Y_train, Y_val, Y_test = combine_shuffle_split(processed_results, data_cfg)
    
    # Save H5 files
    logger.info("Saving H5 files...")
    
    output_dir = Path(args.output_root) / args.tag
    save_h5_files(X_train, X_val, X_test, Y_train, Y_val, Y_test, output_dir, samples)
    
    # Generate control plots
    logger.info("Generating control plots...")
    
    plot_dir = Path(args.plot_dir) / args.tag
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        control_plots(processed_results, plot_dir)
        logger.info(f"Control plots saved to {plot_dir}")
    except Exception as e:
        logger.warning(f"Control plots failed: {e}")
        logger.warning("Continuing without plots...")
    
    # Summary
    logger.info("Preprocessing complete")
    
    logger.info(f"Preprocessed data saved to: {output_dir}")
    logger.info(f"Control plots saved to: {plot_dir}")
    logger.info(f"Total events processed: {sum(samples.values())}")
    logger.info(f"Train/Val/Test: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
