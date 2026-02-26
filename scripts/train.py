"""Training entry point for L1DeepMET."""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tensorflow as tf  # type: ignore
from tensorflow.keras import optimizers  # type: ignore

from l1deepmet.config import create_default_config, load_config
from l1deepmet.data.loader import H5DataLoader
from l1deepmet.losses.corrected import CorrectedCompositeLoss
from l1deepmet.metrics.binned import BinnedDeviation
from l1deepmet.models.factory import build_dense
from l1deepmet.train import train_full_precision


def parse_args():
    p = argparse.ArgumentParser(description="Train L1DeepMET model")
    p.add_argument("--data-dir", required=True, help="Directory with train/val/test .h5 files")
    p.add_argument("--output-dir", default="outputs/models/run_0", help="Where to save model and logs")
    p.add_argument("--config", default=None, help="Path to YAML config (uses default if omitted)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--normfac", type=float, default=100.0,
                   help="Divide MET targets by this value to normalize scale (default: 100)")
    p.add_argument("--mode", type=int, choices=[0, 1, 2], default=None,
                   help="Training mode: 0=direct, 1=per-particle weight, 2=event weight")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("train")

    # Config
    cfg = load_config(args.config) if args.config else create_default_config()
    if args.epochs is not None:
        cfg.set("training.epochs", args.epochs)
    if args.batch_size is not None:
        cfg.set("training.batch_size", args.batch_size)
    if args.lr is not None:
        cfg.set("optimizer.learning_rate", args.lr)
    if args.mode is not None:
        cfg.set("training.mode", args.mode)

    batch_size = cfg.get("training.batch_size") or cfg.get("training", {}).get("batch_size", 256)
    normfac = args.normfac

    # Data
    logger.info(f"Loading data from {args.data_dir} (normfac={normfac})")
    loader = H5DataLoader(args.data_dir)
    train_ds = loader.create_tf_dataset("train", batch_size=batch_size, shuffle=True, normfac=normfac)
    val_ds   = loader.create_tf_dataset("val",   batch_size=batch_size, shuffle=False, normfac=normfac)

    # Model
    model = build_dense(cfg.to_dict())
    model.summary()

    # Optimizer
    opt_cfg = cfg.get("optimizer") or {}
    optimizer = optimizers.AdamW(
        learning_rate=opt_cfg.get("learning_rate", 1e-3),
        clipnorm=opt_cfg.get("clipnorm", 1.0),
    )

    # Loss & metrics (pt_bins must match the normalized target scale)
    loss = CorrectedCompositeLoss(normfac=normfac)
    import numpy as np
    pt_bins_normalized = np.array([50., 100., 200., 300., 400., np.inf]) / normfac
    metrics = [
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanSquaredError(name="mse"),
        BinnedDeviation(pt_bins=pt_bins_normalized),
    ]

    # Train
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Starting training — output dir: {args.output_dir}")
    history, model = train_full_precision(
        model=model,
        config=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        batch_size=batch_size,
        path_out=args.output_dir,
    )

    # Save final model
    model_path = os.path.join(args.output_dir, "model_final.keras")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
