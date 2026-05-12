#!/usr/bin/env python3
"""Train a single HGQ2 (QAT) model on the preprocessed L1DeepMET dataset.

Companion to ``scripts/ablation.py`` (which trains full-precision models).
Keeps the same defaults — corrected loss (MAE only, no xy_balance, no
BinnedDeviation), AdamW, EarlyStopping — and produces the same
per-run artifacts (best_model.keras, history.csv, result.json).

The trained model is **directly synthesizable** by hls4ml (see
``src/l1deepmet/synthesis/``) — no ``export_for_hls`` rewrite needed.

Usage:

    python scripts/train_hgq2.py \
        --data-dir /home/users/dprimosc/L1DeepMET/preprocessed/25Jul8_140X_v0 \
        --output-dir outputs/hgq2_run \
        --width 64 --depth 3 --epochs 30 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

# Allow ``l1deepmet`` imports.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # type: ignore  # noqa: E402
import tensorflow as tf  # type: ignore  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(1)

from l1deepmet.losses.corrected import CorrectedCompositeLoss  # noqa: E402
from l1deepmet.metrics.binned import BinnedDeviation  # noqa: E402
from l1deepmet.quantization import (  # noqa: E402
    build_hgq2_model,
    encode_split_for_hgq2,
    make_hgq2_tf_dataset_from_h5,
)
from l1deepmet.metrics.physics import (  # noqa: E402
    compute_puppi_baseline,
    full_physics_card,
)

NORMFAC_DEFAULT = 100.0

logger = logging.getLogger(__name__)


# ─── Loss + optimizer builders (testable in isolation) ──────────────────────

def build_loss(
    *,
    mae_weight: float = 1.0,
    mse_weight: float = 0.0,
    huber_weight: float = 0.0,
    huber_delta: float = 0.5,
    binned_weight: float = 0.0,
    phi_weight: float = 0.0,
    xy_balance_weight: float = 0.0,
    normfac: float = NORMFAC_DEFAULT,
) -> CorrectedCompositeLoss:
    """Build the composite loss with QAT-friendly defaults.

    Defaults match the loss-ablation winner from the great-ishizaka branch:
    MAE only, no xy_balance, no BinnedDeviation. Override anything explicitly
    via CLI to reproduce the older 'broken' recipe.
    """
    return CorrectedCompositeLoss(
        mae_weight=mae_weight,
        mse_weight=mse_weight,
        huber_weight=huber_weight,
        huber_delta=huber_delta,
        binned_weight=binned_weight,
        phi_weight=phi_weight,
        xy_balance_weight=xy_balance_weight,
        normfac=normfac,
    )


def build_optimizer(
    *,
    learning_rate: float = 1e-3,
    clipnorm: float = 1.0,
) -> tf.keras.optimizers.Optimizer:
    """AdamW with gradient-norm clipping (matches the full-precision runner)."""
    return tf.keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=clipnorm)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model: tf.keras.Model, X: np.ndarray, Y: np.ndarray,
                   normfac: float) -> dict:
    """Run inference + compute the full physics card vs PUPPI."""
    inputs, _ = encode_split_for_hgq2(X, Y, normfac=1.0)
    pred = model.predict(inputs, batch_size=512, verbose=0) * normfac
    puppi = compute_puppi_baseline(X)
    return full_physics_card(gen_xy=Y, reco_xy=pred, puppi_xy=puppi)


# ─── Main training loop ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",
                   default="/home/users/dprimosc/L1DeepMET/preprocessed/25Jul8_140X_v0",
                   type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--name", default="hgq2_model",
                   help="Run identifier embedded in saved-file names.")

    # Architecture
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--use-embeddings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bounded-weight", action=argparse.BooleanOptionalAction, default=True)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normfac", type=float, default=NORMFAC_DEFAULT)

    # Loss recipe (defaults to current best: MAE only)
    p.add_argument("--mae-weight", type=float, default=1.0)
    p.add_argument("--mse-weight", type=float, default=0.0)
    p.add_argument("--xy-balance-weight", type=float, default=0.0)
    p.add_argument("--binned-weight", type=float, default=0.0)

    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    log = logging.getLogger("train_hgq2")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic-ish run.
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    log.info("Loading datasets from %s", args.data_dir)
    train_ds = make_hgq2_tf_dataset_from_h5(
        args.data_dir, split="train",
        batch_size=args.batch_size, normfac=args.normfac, seed=args.seed,
    )
    val_ds = make_hgq2_tf_dataset_from_h5(
        args.data_dir, split="val",
        batch_size=args.batch_size, shuffle=False, normfac=args.normfac,
    )

    log.info("Building HGQ2 model (w=%d d=%d emb=%s bounded=%s)",
             args.width, args.depth, args.use_embeddings, args.bounded_weight)
    model = build_hgq2_model(
        width=args.width, depth=args.depth,
        use_embeddings=args.use_embeddings,
        bounded_weight=args.bounded_weight,
        normfac=args.normfac,
    )

    loss = build_loss(
        mae_weight=args.mae_weight,
        mse_weight=args.mse_weight,
        binned_weight=args.binned_weight,
        xy_balance_weight=args.xy_balance_weight,
        normfac=args.normfac,
    )
    optimizer = build_optimizer(learning_rate=args.learning_rate)

    pt_bins_normalized = np.array([50.0, 100.0, 200.0, 300.0, 400.0, np.inf]) / args.normfac
    metrics = [
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanSquaredError(name="mse"),
        BinnedDeviation(pt_bins=pt_bins_normalized),
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(str(args.output_dir / "history.csv")),
    ]

    log.info("Training for up to %d epochs", args.epochs)
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
    )
    elapsed = time.time() - t0
    n_epochs = len(history.history["val_loss"])
    log.info("Training done in %.0fs (%d epochs)", elapsed, n_epochs)

    # Final evaluation on test set.
    import h5py  # local
    with h5py.File(str(args.data_dir / "test.h5"), "r") as f:
        X_test = f["features"][:]
        Y_test = f["targets"][:]

    card = evaluate_model(model, X_test, Y_test, normfac=args.normfac)
    log.info("Test resolution: X = %.2f, Y = %.2f, pT = %.2f  |  AUC = %.4f",
             card["met_x_resolution"], card["met_y_resolution"],
             card["met_pt_resolution"], card["auc"])

    # Save model + result.json.
    model_path = args.output_dir / "best_model.keras"
    model.save(model_path)
    log.info("Saved model to %s", model_path)

    result = {
        "name": args.name,
        "seed": args.seed,
        "n_params": int(model.count_params()),
        "epochs_trained": n_epochs,
        "train_time_s": elapsed,
        "best_val_loss": float(min(history.history["val_loss"])),
        "model_path": str(model_path.resolve()),
        "arch": {
            "width": args.width, "depth": args.depth,
            "use_embeddings": args.use_embeddings,
            "bounded_weight": args.bounded_weight,
            "quantization": "hgq2",
        },
        "loss_recipe": {
            "mae_weight": args.mae_weight,
            "mse_weight": args.mse_weight,
            "xy_balance_weight": args.xy_balance_weight,
            "binned_weight": args.binned_weight,
        },
        "test_physics_card": card,
    }
    with open(args.output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print()
    print("=" * 70)
    print(f"HGQ2 training summary  ({args.name}, seed {args.seed})")
    print("=" * 70)
    print(f"  X IQR/2  : {card['met_x_resolution']:6.2f} GeV")
    print(f"  Y IQR/2  : {card['met_y_resolution']:6.2f} GeV")
    print(f"  pT IQR/2 : {card['met_pt_resolution']:6.2f} GeV")
    print(f"  φ res    : {card['phi_resolution']:6.4f} rad")
    print(f"  AUC      : {card['auc']:6.4f}")
    print(f"  Δ vs PUPPI X : {card.get('delta_met_x_resolution', float('nan')):+6.2f} GeV")
    print(f"  Δ vs PUPPI pT: {card.get('delta_met_pt_resolution', float('nan')):+6.2f} GeV")
    print(f"  Δ vs PUPPI AUC: {card.get('delta_auc', float('nan')):+6.4f}")
    print(f"  Model     : {model_path}")
    print(f"  Result    : {args.output_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
