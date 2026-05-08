"""Architecture search for L1DeepMET Dense models.

Sweeps over width, depth, embeddings, training mode, activation, and loss
weight to find the best physics-informed architecture within FPGA resource
constraints.  Results are logged to a CSV and a summary table is printed.

Usage:
    python scripts/arch_search.py \
        --data-dir preprocessed/25Jul8_140X_v0 \
        --output-dir outputs/arch_search
"""

import argparse
import csv
import itertools
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Limit TF threading to avoid pthread_create failures on constrained systems
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)
from tensorflow.keras.layers import (  # type: ignore
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    Multiply,
)
from tensorflow.keras.models import Model  # type: ignore

from l1deepmet.data.loader import H5DataLoader
from l1deepmet.losses.corrected import CorrectedCompositeLoss
from l1deepmet.metrics.binned import BinnedDeviation

logger = logging.getLogger("arch_search")

# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------

@dataclass
class ArchConfig:
    """Single architecture configuration."""
    name: str
    width: int                    # hidden units per layer
    depth: int                    # number of hidden layers
    mode: int                     # 0 = direct regression, 1 = per-particle weight
    activation: str               # "relu" or "elu"
    use_embeddings: bool          # whether to embed pdgId and charge
    binned_weight: float          # weight for BinnedDeviation in composite loss
                                  # WARNING: > 0 degrades resolution. See
                                  # reports/loss_diagnosis_apr2026/. Keep at 0.
    embed_pdgid_dim: int = 4      # pdgId embedding output dim (vocab=6)
    embed_charge_dim: int = 2     # charge embedding output dim (vocab=4)
    with_bias: bool = False       # per-particle bias (b_ix, b_iy) correction
    weight_minus_one: bool = False # initialize weight at -1 (start from PF MET)
    use_sum: bool = False         # use reduce_sum instead of GlobalAveragePooling1D
    use_2d_weights: bool = False  # separate w_x, w_y per particle (vs scalar w)
    learning_rate: float = 1e-3   # optimizer learning rate
    use_cosine_decay: bool = False # cosine decay LR schedule
    warmup_epochs: int = 0        # linear warmup epochs before cosine decay
    phi_loss_weight: float = 0.0  # weight for phi-aware loss term (1-cos(dphi))
    xy_balance_weight: float = 0.0 # weight for X/Y symmetry loss (|MAE_x - MAE_y|)


def generate_search_configs() -> List[ArchConfig]:
    """Generate the architecture search space.

    Primary axes (full grid):
      - embeddings: [False, True]
      - width: [32, 64]
      - depth: [2, 3, 4]
      - mode: [0, 1]

    Secondary axes (sampled):
      - activation: default relu, test elu on a subset
      - binned_weight: default 200, test 100 on a subset

    This produces ~24 configs total.
    """
    configs: List[ArchConfig] = []

    # Primary grid: embeddings x width x depth x mode
    # Use relu + binned_weight=200 as defaults for the full grid.
    # Skip mode 0 for w32/no-embedding (least informative) to keep total ~24.
    for use_emb, width, depth, mode in itertools.product(
        [False, True],   # embeddings
        [32, 64],         # width
        [2, 3, 4],        # depth
        [0, 1],           # mode
    ):
        # Skip small no-embedding direct-regression configs
        if not use_emb and width == 32 and mode == 0:
            continue

        name = (
            f"{'emb' if use_emb else 'noemb'}"
            f"_w{width}_d{depth}"
            f"_m{mode}_relu_bw200"
        )
        configs.append(ArchConfig(
            name=name,
            width=width,
            depth=depth,
            mode=mode,
            activation="relu",
            use_embeddings=use_emb,
            binned_weight=200.0,
        ))

    # Secondary: ELU activation — test on mode-1 with embeddings, width=64, depths 2+4
    for depth in [2, 4]:
        name = f"emb_w64_d{depth}_m1_elu_bw200"
        configs.append(ArchConfig(
            name=name,
            width=64,
            depth=depth,
            mode=1,
            activation="elu",
            use_embeddings=True,
            binned_weight=200.0,
        ))

    # Secondary: lower binned_weight=100 — test on mode-1 with embeddings, width=64, depths 2+4
    for depth in [2, 4]:
        name = f"emb_w64_d{depth}_m1_relu_bw100"
        configs.append(ArchConfig(
            name=name,
            width=64,
            depth=depth,
            mode=1,
            activation="relu",
            use_embeddings=True,
            binned_weight=100.0,
        ))

    return configs


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

N_PARTICLES = 128
PDGID_VOCAB = 6   # 0=invalid, 1-5 = particle types
CHARGE_VOCAB = 4   # 0=invalid, 1=neg, 2=neutral, 3=pos


class CastToInt(tf.keras.layers.Layer):
    """Cast input to int32 — Keras 3 compatible."""
    def call(self, x):
        return tf.cast(x, tf.int32)


class ZeroReduce(tf.keras.layers.Layer):
    """Reduce input to zeros of shape (B, dim) — dummy connection for unused inputs."""
    def __init__(self, output_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.zeros((batch_size, self.output_dim), dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config["output_dim"] = self.output_dim
        return config


class ShiftByConstant(tf.keras.layers.Layer):
    """Add a fixed constant to input. Used for weight-minus-one trick.

    When initialized with shift=-1.0, the per-particle weight starts near -1
    (since Dense init produces values near 0), so the initial MET prediction
    is approximately -sum(-1 * px_i) = sum(px_i), which is the raw PF MET.
    The network then learns small corrections from this baseline.
    """
    def __init__(self, shift=-1.0, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift

    def call(self, x):
        return x + self.shift

    def get_config(self):
        config = super().get_config()
        config["shift"] = self.shift
        return config


class SumOverParticles(tf.keras.layers.Layer):
    """Sum over the particle axis (axis=1). Drop-in replacement for
    GlobalAveragePooling1D that doesn't divide by N."""
    def call(self, x):
        return tf.reduce_sum(x, axis=1)

    def get_config(self):
        return super().get_config()


def build_model(cfg: ArchConfig) -> Model:
    """Build a Keras functional model from an ArchConfig."""

    # --- Inputs (match data loader dict keys) ---
    x_cont = Input(shape=(N_PARTICLES, 5), name="continuous_inputs")
    x_pxpy = Input(shape=(N_PARTICLES, 2), name="momentum_inputs")
    x_pdgid = Input(shape=(N_PARTICLES,), name="pdgid_inputs")
    x_charge = Input(shape=(N_PARTICLES,), name="charge_inputs")

    # --- Feature assembly ---
    if cfg.use_embeddings:
        pdgid_int = CastToInt(name="cast_pdgid")(x_pdgid)
        charge_int = CastToInt(name="cast_charge")(x_charge)

        emb_pdgid = Embedding(
            input_dim=PDGID_VOCAB,
            output_dim=cfg.embed_pdgid_dim,
            name="emb_pdgid",
        )(pdgid_int)  # (B, 128, embed_pdgid_dim)

        emb_charge = Embedding(
            input_dim=CHARGE_VOCAB,
            output_dim=cfg.embed_charge_dim,
            name="emb_charge",
        )(charge_int)  # (B, 128, embed_charge_dim)

        features = Concatenate(name="concat_features")(
            [x_cont, emb_pdgid, emb_charge]
        )  # (B, 128, 5 + 4 + 2 = 11)
    else:
        features = x_cont  # (B, 128, 5)

    # --- Dense body ---
    x = features
    for i in range(cfg.depth):
        x = Dense(
            cfg.width,
            activation=None,
            kernel_initializer="lecun_uniform",
            name=f"dense_{i}",
        )(x)
        x = BatchNormalization(momentum=0.95, name=f"bn_{i}")(x)
        x = Activation(cfg.activation, name=f"act_{i}")(x)

    # --- Output head ---
    if cfg.mode == 1:
        if cfg.use_2d_weights:
            # 2D weights: predict (w_x, w_y) per particle for independent px/py correction.
            # This allows the model to correct both magnitude AND direction per particle,
            # unlike scalar weight which can only scale magnitude uniformly.
            w_init = "zeros" if cfg.weight_minus_one else "lecun_uniform"
            b_init = "zeros"
            w2d = Dense(2, activation="linear", name="met_weight_2d",
                        kernel_initializer=w_init, bias_initializer=b_init)(x)  # (B, 128, 2)

            if cfg.weight_minus_one:
                if cfg.use_sum:
                    shift_val = -1.0 / NORMFAC
                else:
                    shift_val = -float(N_PARTICLES) / NORMFAC
                w2d = ShiftByConstant(shift=shift_val, name="weight_minus_one")(w2d)

            # Element-wise multiply: w_x * px, w_y * py
            weighted = Multiply(name="weight_pxpy")([w2d, x_pxpy])  # (B, 128, 2)

            # Per-particle bias still supported on top of 2D weights
            if cfg.with_bias:
                bias_k_init = "zeros" if cfg.weight_minus_one else "lecun_uniform"
                bias = Dense(2, activation="linear", name="met_bias",
                             kernel_initializer=bias_k_init, bias_initializer="zeros")(x)
                weighted = tf.keras.layers.Add(name="add_bias")([weighted, bias])

            if cfg.use_sum:
                out = SumOverParticles(name="output")(weighted)
            else:
                out = GlobalAveragePooling1D(name="output")(weighted)
        else:
            # Per-particle scalar weight (original mode 1).
            # When weight_minus_one is enabled, init weight Dense with zeros so the
            # pre-shift output is exactly 0, giving a clean -shift starting point.
            w_init = "zeros" if cfg.weight_minus_one else "lecun_uniform"
            b_init = "zeros"
            w = Dense(1, activation="linear", name="met_weight",
                      kernel_initializer=w_init, bias_initializer=b_init)(x)  # (B, 128, 1)

            # Weight-minus-one: shift so initial weight starts from PF MET baseline.
            # With use_sum, output = sum(w * pxpy). Targets are in GeV/normfac.
            # So shift = -1/normfac makes initial output ≈ PF_MET/normfac ≈ targets.
            # Without use_sum (GAP), output = mean(w * pxpy), so shift = -N/normfac
            # to compensate for the 1/N averaging.
            if cfg.weight_minus_one:
                if cfg.use_sum:
                    shift_val = -1.0 / NORMFAC
                else:
                    shift_val = -float(N_PARTICLES) / NORMFAC
                w = ShiftByConstant(shift=shift_val, name="weight_minus_one")(w)

            # Weighted momentum
            weighted = Multiply(name="weight_pxpy")([w, x_pxpy])     # (B, 128, 2)

            # Per-particle bias: additive momentum correction (b_ix, b_iy)
            if cfg.with_bias:
                bias_k_init = "zeros" if cfg.weight_minus_one else "lecun_uniform"
                bias = Dense(2, activation="linear", name="met_bias",
                             kernel_initializer=bias_k_init, bias_initializer="zeros")(x)  # (B, 128, 2)
                weighted = tf.keras.layers.Add(name="add_bias")([weighted, bias])

            # Sum over particles (not average — original DeepMET uses reduce_sum)
            if cfg.use_sum:
                out = SumOverParticles(name="output")(weighted)       # (B, 2)
            else:
                out = GlobalAveragePooling1D(name="output")(weighted) # (B, 2)
    else:
        # Mode 0: global pool features, then regress MET directly
        pooled = GlobalAveragePooling1D(name="pool")(x)           # (B, width)
        out = Dense(2, activation="linear", name="output")(pooled)  # (B, 2)

    # All four inputs are always declared so the model accepts the full
    # dataset dict. Unused inputs get a zero-contribution dummy path.
    if not cfg.use_embeddings:
        _cat_zero = ZeroReduce(output_dim=2, name="cat_dummy")(
            Concatenate(name="cat_stack")([
                tf.keras.layers.Reshape((N_PARTICLES, 1), name="pdgid_reshape")(x_pdgid),
                tf.keras.layers.Reshape((N_PARTICLES, 1), name="charge_reshape")(x_charge),
            ])
        )
        out = tf.keras.layers.Add(name="add_cat_dummy")([out, _cat_zero])

    if cfg.mode == 0:
        _pxpy_zero = ZeroReduce(output_dim=2, name="pxpy_dummy")(x_pxpy)
        out = tf.keras.layers.Add(name="add_pxpy_dummy")([out, _pxpy_zero])

    model = Model(
        inputs={
            "continuous_inputs": x_cont,
            "momentum_inputs": x_pxpy,
            "pdgid_inputs": x_pdgid,
            "charge_inputs": x_charge,
        },
        outputs=out,
    )
    return model


# ---------------------------------------------------------------------------
# Training loop for a single config
# ---------------------------------------------------------------------------

NORMFAC = 100.0


def train_config(
    cfg: ArchConfig,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    output_dir: str,
) -> dict:
    """Train one architecture config and return results dict."""

    run_dir = os.path.join(output_dir, cfg.name)
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"=== Building model: {cfg.name} ===")
    model = build_model(cfg)
    param_count = model.count_params()
    logger.info(f"  Parameters: {param_count:,}")

    # Loss
    loss = CorrectedCompositeLoss(
        binned_weight=cfg.binned_weight,
        phi_weight=cfg.phi_loss_weight,
        xy_balance_weight=cfg.xy_balance_weight,
        normfac=NORMFAC,
    )

    # Metrics
    pt_bins_normalized = np.array([50.0, 100.0, 200.0, 300.0, 400.0, np.inf]) / NORMFAC
    metrics = [
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanSquaredError(name="mse"),
        BinnedDeviation(pt_bins=pt_bins_normalized),
    ]

    # Optimizer — support configurable LR and cosine decay
    lr = cfg.learning_rate
    if cfg.use_cosine_decay:
        # Cosine decay with optional warmup
        # Approximate steps: ~118k train events / 256 batch_size ≈ 462 steps/epoch
        steps_per_epoch = 462
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * cfg.warmup_epochs
        if warmup_steps > 0:
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=total_steps - warmup_steps,
                alpha=1e-6,  # minimum LR
                warmup_target=lr,
                warmup_steps=warmup_steps,
            )
        else:
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=total_steps,
                alpha=1e-6,
            )
        lr = lr_schedule

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr,
        clipnorm=1.0,
    )

    # Callbacks — lightweight for search: early stopping + CSV logger
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv")),
    ]

    # Compile and train
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    t0 = time.time()
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2,
        )
        elapsed = time.time() - t0
        trained = True

        # Extract best val metrics (from the epoch with best val_loss)
        val_losses = history.history.get("val_loss", [])
        if val_losses:
            best_epoch = int(np.argmin(val_losses))
            result = {
                "config_name": cfg.name,
                "width": cfg.width,
                "depth": cfg.depth,
                "mode": cfg.mode,
                "activation": cfg.activation,
                "use_embeddings": cfg.use_embeddings,
                "binned_weight": cfg.binned_weight,
                "param_count": param_count,
                "best_epoch": best_epoch + 1,
                "epochs_trained": len(val_losses),
                "val_loss": val_losses[best_epoch],
                "val_mae": history.history.get("val_mae", [np.nan])[best_epoch],
                "val_mse": history.history.get("val_mse", [np.nan])[best_epoch],
                "val_binned_deviation": history.history.get("val_binned_deviation", [np.nan])[best_epoch],
                "train_time_s": elapsed,
                "status": "ok",
            }
        else:
            result = _error_result(cfg, param_count, "no_val_loss")

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  FAILED: {e}")
        result = _error_result(cfg, param_count, str(e)[:100])
        result["train_time_s"] = elapsed

    # Delete model to free memory (avoid clear_session which leaks threads)
    del model
    import gc
    gc.collect()

    return result


def _error_result(cfg: ArchConfig, param_count: int, status: str) -> dict:
    return {
        "config_name": cfg.name,
        "width": cfg.width,
        "depth": cfg.depth,
        "mode": cfg.mode,
        "activation": cfg.activation,
        "use_embeddings": cfg.use_embeddings,
        "binned_weight": cfg.binned_weight,
        "param_count": param_count,
        "best_epoch": -1,
        "epochs_trained": 0,
        "val_loss": float("nan"),
        "val_mae": float("nan"),
        "val_mse": float("nan"),
        "val_binned_deviation": float("nan"),
        "train_time_s": 0.0,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: List[dict]) -> None:
    """Print a formatted summary table sorted by val_loss."""
    # Filter successful runs
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    ok.sort(key=lambda r: r["val_loss"])

    print("\n" + "=" * 120)
    print("ARCHITECTURE SEARCH RESULTS — sorted by val_loss (best first)")
    print("=" * 120)
    header = (
        f"{'Rank':<5} {'Config':<35} {'Mode':<5} {'Emb':<5} "
        f"{'W':<4} {'D':<3} {'Act':<5} {'BW':<5} "
        f"{'Params':>8} {'Epoch':>6} "
        f"{'val_loss':>10} {'val_mae':>10} {'val_mse':>10} {'val_bd':>10} "
        f"{'Time(s)':>8}"
    )
    print(header)
    print("-" * 120)
    for i, r in enumerate(ok, 1):
        print(
            f"{i:<5} {r['config_name']:<35} {r['mode']:<5} "
            f"{'Y' if r['use_embeddings'] else 'N':<5} "
            f"{r['width']:<4} {r['depth']:<3} {r['activation']:<5} "
            f"{r['binned_weight']:<5.0f} "
            f"{r['param_count']:>8,} {r['best_epoch']:>6} "
            f"{r['val_loss']:>10.6f} {r['val_mae']:>10.6f} "
            f"{r['val_mse']:>10.6f} {r['val_binned_deviation']:>10.6f} "
            f"{r['train_time_s']:>8.1f}"
        )

    if failed:
        print(f"\n--- {len(failed)} FAILED configs ---")
        for r in failed:
            print(f"  {r['config_name']}: {r['status']}")

    print("=" * 120)
    print(f"Total configs: {len(results)} | Successful: {len(ok)} | Failed: {len(failed)}")

    if ok:
        best = ok[0]
        print(f"\nBest config: {best['config_name']}")
        print(f"  val_loss={best['val_loss']:.6f}  val_mae={best['val_mae']:.6f}  "
              f"val_mse={best['val_mse']:.6f}  val_bd={best['val_binned_deviation']:.6f}")
        print(f"  params={best['param_count']:,}  best_epoch={best['best_epoch']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="L1DeepMET architecture search")
    p.add_argument("--data-dir", required=True,
                   help="Directory with train.h5, val.h5, test.h5")
    p.add_argument("--output-dir", default="outputs/arch_search",
                   help="Root output directory for search results")
    p.add_argument("--epochs", type=int, default=30,
                   help="Max epochs per config (default: 30)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size (default: 256)")
    p.add_argument("--normfac", type=float, default=100.0,
                   help="MET normalization factor (default: 100)")
    p.add_argument("--resume", action="store_true",
                   help="Skip configs whose results already appear in the CSV")
    return p.parse_args()


def main():
    args = parse_args()
    global NORMFAC
    NORMFAC = args.normfac

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "search_results.csv")

    # Load data once — shared across all configs
    logger.info(f"Loading data from {args.data_dir} (normfac={NORMFAC})")
    loader = H5DataLoader(args.data_dir)
    train_ds = loader.create_tf_dataset(
        "train", batch_size=args.batch_size, shuffle=True, normfac=NORMFAC
    )
    val_ds = loader.create_tf_dataset(
        "val", batch_size=args.batch_size, shuffle=False, normfac=NORMFAC
    )

    # Generate configs
    configs = generate_search_configs()
    logger.info(f"Search space: {len(configs)} configurations")

    # Check for resume
    completed_names = set()
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_names.add(row["config_name"])
        logger.info(f"Resuming: {len(completed_names)} configs already completed")

    # CSV header
    fieldnames = [
        "config_name", "width", "depth", "mode", "activation",
        "use_embeddings", "binned_weight", "param_count", "best_epoch",
        "epochs_trained", "val_loss", "val_mae", "val_mse",
        "val_binned_deviation", "train_time_s", "status",
    ]

    write_header = not os.path.exists(csv_path) or not args.resume
    csv_file = open(csv_path, "a" if args.resume else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    all_results: List[dict] = []

    # If resuming, load existing results for the summary
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields back
                for key in ["width", "depth", "mode", "param_count", "best_epoch", "epochs_trained"]:
                    row[key] = int(row[key]) if row[key] not in ("", "nan") else 0
                for key in ["binned_weight", "val_loss", "val_mae", "val_mse",
                            "val_binned_deviation", "train_time_s"]:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        row[key] = float("nan")
                row["use_embeddings"] = row["use_embeddings"] in ("True", "true", "1")
                all_results.append(row)

    # Train each config sequentially
    total = len(configs)
    for idx, cfg in enumerate(configs, 1):
        if cfg.name in completed_names:
            logger.info(f"[{idx}/{total}] Skipping {cfg.name} (already completed)")
            continue

        logger.info(f"\n[{idx}/{total}] Training: {cfg.name}")
        result = train_config(cfg, train_ds, val_ds, args.epochs, args.output_dir)

        writer.writerow(result)
        csv_file.flush()
        all_results.append(result)

        logger.info(
            f"  Result: val_loss={result['val_loss']:.6f}  "
            f"val_mae={result['val_mae']:.6f}  "
            f"params={result['param_count']:,}  "
            f"status={result['status']}"
        )

    csv_file.close()
    logger.info(f"\nResults saved to {csv_path}")

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
