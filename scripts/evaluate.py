"""Comprehensive evaluation of L1DeepMET architectures on the test set.

Retrains the top 5 architecture candidates from the arch search (3 seeds each),
evaluates them on the test set with physics metrics, compares against PUPPI
baseline, and generates publication-quality plots.

Usage:
    python scripts/evaluate.py \
        --data-dir preprocessed/25Jul8_140X_v0 \
        --output-dir outputs/evaluation \
        --epochs 50
"""

import argparse
import csv
import gc
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Limit TF threading to avoid pthread_create failures on constrained systems
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_DATA_EXPERIMENTAL_SLACK"] = "0"

import numpy as np  # type: ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore

import tensorflow as tf  # type: ignore

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from arch_search import ArchConfig, build_model
from l1deepmet.data.loader import H5DataLoader
from l1deepmet.losses.corrected import CorrectedCompositeLoss
from l1deepmet.metrics.binned import BinnedDeviation
from l1deepmet.plotting import (
    MakePlots,
    convertXY2PtPhi,
    phidiff,
    plot_roc_curve,
    plot_trigger_rates,
    plot_turn_on_curves,
    resolqt,
)

logger = logging.getLogger("evaluate")

NORMFAC = 100.0

# ============================================================================
# Top 5 architecture configs to evaluate
# ============================================================================

TOP_CONFIGS: List[ArchConfig] = [
    # ================================================================
    # BASELINE: scalar weight (reference)
    # ================================================================
    ArchConfig(
        name="baseline_wm1_w64_d3",
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=0.0,
        with_bias=False, weight_minus_one=True, use_sum=True,
    ),

    # ================================================================
    # ROUND 2: Fix the X/Y asymmetry in 2D weights
    # Root cause: binned_deviation (weight=200) only penalizes pT errors,
    # so the model sacrifices X/Y symmetry for pT improvement.
    # ================================================================

    # 2D weights + XY balance loss — prevents axis asymmetry
    ArchConfig(
        name="w2d_xybal10_emb_w64_d3",
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=0.0,
        with_bias=False, weight_minus_one=True, use_sum=True,
        use_2d_weights=True,
        xy_balance_weight=10.0,
    ),

    # 2D weights + reduced binned_weight (rebalance pT vs X/Y)
    ArchConfig(
        name="w2d_bw50_emb_w64_d3",
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=50.0,
        with_bias=False, weight_minus_one=True, use_sum=True,
        use_2d_weights=True,
    ),

    # 2D weights + small phi loss (2.0 — previous 50.0 was too strong)
    ArchConfig(
        name="w2d_phi2_emb_w64_d3",
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=0.0,
        with_bias=False, weight_minus_one=True, use_sum=True,
        use_2d_weights=True,
        phi_loss_weight=2.0,
    ),

    # 2D weights + XY balance + phi loss (combined)
    ArchConfig(
        name="w2d_xybal_phi2_emb_w64_d3",
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=0.0,
        with_bias=False, weight_minus_one=True, use_sum=True,
        use_2d_weights=True,
        phi_loss_weight=2.0,
        xy_balance_weight=10.0,
    ),

    # 2D weights + reduced binned + XY balance + phi (full rebalanced recipe)
    ArchConfig(
        name="w2d_bw50_xybal_phi2_emb_w64_d3",
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=50.0,
        with_bias=False, weight_minus_one=True, use_sum=True,
        use_2d_weights=True,
        phi_loss_weight=2.0,
        xy_balance_weight=10.0,
    ),

    # 2D weights + larger XY balance weight
    ArchConfig(
        name="w2d_xybal50_emb_w64_d3",
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=0.0,
        with_bias=False, weight_minus_one=True, use_sum=True,
        use_2d_weights=True,
        xy_balance_weight=50.0,
    ),

    # Scalar weight + XY balance (does XY balance help even scalar models?)
    ArchConfig(
        name="scalar_xybal10_emb_w64_d3",
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=0.0,
        with_bias=False, weight_minus_one=True, use_sum=True,
        xy_balance_weight=10.0,
    ),
]

SEEDS = [42, 123, 456]

# ============================================================================
# Phase 1: Retrain top architectures with multiple seeds
# ============================================================================


def retrain_model(
    cfg: ArchConfig,
    seed: int,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    output_dir: str,
) -> Optional[str]:
    """Train one model config with a given seed, save best weights.

    Returns the path to the saved model directory, or None on failure.
    """
    run_name = f"{cfg.name}_seed{seed}"
    run_dir = os.path.join(output_dir, "models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "best_model.keras")

    # Skip if already trained
    if os.path.exists(model_path):
        logger.info(f"  Model already exists at {model_path}, skipping training")
        return model_path

    logger.info(f"=== Training {run_name} (seed={seed}) ===")

    # Set seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_model(cfg)
    param_count = model.count_params()
    logger.info(f"  Parameters: {param_count:,}")

    loss = CorrectedCompositeLoss(
        binned_weight=cfg.binned_weight,
        phi_weight=getattr(cfg, 'phi_loss_weight', 0.0),
        xy_balance_weight=getattr(cfg, 'xy_balance_weight', 0.0),
        normfac=NORMFAC,
    )

    pt_bins_normalized = np.array([50.0, 100.0, 200.0, 300.0, 400.0, np.inf]) / NORMFAC
    metrics = [
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanSquaredError(name="mse"),
        BinnedDeviation(pt_bins=pt_bins_normalized),
    ]

    # Optimizer — support configurable LR and cosine decay from config
    lr = getattr(cfg, 'learning_rate', 1e-3)
    use_cosine = getattr(cfg, 'use_cosine_decay', False)
    warmup_ep = getattr(cfg, 'warmup_epochs', 0)

    if use_cosine:
        steps_per_epoch = 462  # ~118k events / 256 batch
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * warmup_ep
        if warmup_steps > 0:
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=total_steps - warmup_steps,
                alpha=1e-6,
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

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, clipnorm=1.0)

    # Adjust patience based on epoch count — longer training gets more patience
    patience_es = max(10, epochs // 10)
    patience_lr = max(5, epochs // 20)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_es, verbose=1, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=patience_lr, verbose=1, min_lr=1e-6,
        ) if not use_cosine else tf.keras.callbacks.TerminateOnNaN(),  # Don't ReduceLR when using cosine
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv")),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    t0 = time.time()
    try:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2,
        )
        elapsed = time.time() - t0
        logger.info(f"  Training complete in {elapsed:.1f}s")
    except Exception as e:
        logger.error(f"  Training FAILED for {run_name}: {e}")
        traceback.print_exc()
        del model
        gc.collect()
        return None

    del model
    gc.collect()

    if os.path.exists(model_path):
        return model_path
    return None


# ============================================================================
# Phase 2: Test-set evaluation with physics metrics
# ============================================================================


def compute_puppi_met(features: np.ndarray) -> np.ndarray:
    """Compute PUPPI MET as the negative sum of candidate (px, py).

    L1PuppiCands are the *outputs* of the PUPPI algorithm — pileup mitigation
    has already been applied at ntuple-write time, so the stored ``pt`` is
    ``puppi_weight × pt_input``. Summing the candidates therefore reproduces
    the official ``L1PuppiMet`` branch to floating-point precision (verified
    on source ROOT files: max |Δ| = 0.0004 GeV across 5000 events).

    Do NOT multiply by ``features[..., 3]`` (puppi_weight) before summing —
    the weight is already folded into pt; applying it again is double-weighting.
    The ``puppi_weight`` field is preserved in the features as a diagnostic
    record of what factor was applied, useful as a per-particle feature for
    the model.

    Features layout: [..., 5] = px, [..., 6] = py per particle.
    """
    puppi_px = -np.sum(features[:, :, 5], axis=1)
    puppi_py = -np.sum(features[:, :, 6], axis=1)
    return np.stack([puppi_px, puppi_py], axis=1)


def run_inference(model_path: str, cfg: ArchConfig, test_features: np.ndarray) -> np.ndarray:
    """Load a saved model and run inference on test features.

    Returns predicted MET in GeV as (N, 2) array.
    """
    from l1deepmet.data.loader import split_preprocessed_features

    # Load model with custom objects
    from arch_search import CastToInt, ZeroReduce, ShiftByConstant, SumOverParticles

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "CastToInt": CastToInt,
            "ZeroReduce": ZeroReduce,
            "ShiftByConstant": ShiftByConstant,
            "SumOverParticles": SumOverParticles,
            "CorrectedCompositeLoss": CorrectedCompositeLoss,
            "BinnedDeviation": BinnedDeviation,
        },
    )

    inputs, pxpy, cat0, cat1 = split_preprocessed_features(test_features)

    input_dict = {
        "continuous_inputs": inputs,
        "momentum_inputs": pxpy,
        "pdgid_inputs": cat0,
        "charge_inputs": cat1,
    }

    # Predict in batches to avoid OOM
    predictions = model.predict(input_dict, batch_size=1024, verbose=0)

    # Convert back to GeV
    predictions_gev = predictions * NORMFAC

    del model
    gc.collect()

    return predictions_gev


def compute_metrics(
    gen_xy: np.ndarray, reco_xy: np.ndarray, label: str = ""
) -> Dict[str, float]:
    """Compute physics metrics for a single model.

    Args:
        gen_xy: True MET (px, py) in GeV, shape (N, 2)
        reco_xy: Reconstructed MET (px, py) in GeV, shape (N, 2)

    Returns:
        Dictionary of metric names to values.
    """
    gen_pt_phi = convertXY2PtPhi(gen_xy)
    reco_pt_phi = convertXY2PtPhi(reco_xy)

    gen_pt = gen_pt_phi[:, 0]
    gen_phi = gen_pt_phi[:, 1]
    reco_pt = reco_pt_phi[:, 0]
    reco_phi = reco_pt_phi[:, 1]

    # --- Overall MET pT resolution (response-corrected) ---
    # Response correction: in bins of gen_pt, correct reco_pt by mean response
    # Start at 50 GeV to avoid dividing by near-zero gen_pt
    pt_bins_resp = [50, 100, 200, 300, 400, np.inf]
    responses = []
    for lo, hi in zip(pt_bins_resp[:-1], pt_bins_resp[1:]):
        mask = (gen_pt >= lo) & (gen_pt < hi)
        if np.sum(mask) > 10:
            resp = np.mean(reco_pt[mask] / gen_pt[mask])
            responses.append((lo, hi, resp, np.sum(mask)))
        else:
            responses.append((lo, hi, 1.0, np.sum(mask)))

    # Apply response correction (only for events with gen_pt >= 50 GeV)
    high_pt_mask = gen_pt >= 50
    reco_pt_corrected = np.copy(reco_pt)
    for lo, hi, resp, _ in responses:
        mask = (gen_pt >= lo) & (gen_pt < hi)
        if resp > 0.01:
            reco_pt_corrected[mask] = reco_pt[mask] / resp

    # Resolution = (p84 - p16) / 2 of (gen_pt - reco_pt_corrected)
    # Compute on events with gen_pt >= 50 to avoid noise-dominated low-MET region
    pt_residuals_all = gen_pt - reco_pt_corrected
    met_pt_resolution = resolqt(pt_residuals_all[high_pt_mask])

    # --- MET X, Y resolution ---
    met_x_resolution = resolqt(gen_xy[:, 0] - reco_xy[:, 0])
    met_y_resolution = resolqt(gen_xy[:, 1] - reco_xy[:, 1])

    # --- Mean response across bins ---
    mean_response = np.mean([r[2] for r in responses if r[3] > 10])

    # --- Phi resolution ---
    dphi = phidiff(gen_phi, reco_phi)
    phi_resolution = resolqt(dphi)

    # --- Response per bin ---
    metrics = {
        "met_pt_resolution": met_pt_resolution,
        "met_x_resolution": met_x_resolution,
        "met_y_resolution": met_y_resolution,
        "mean_response": mean_response,
        "phi_resolution": phi_resolution,
    }

    # Add per-bin response
    for lo, hi, resp, count in responses:
        hi_str = "inf" if np.isinf(hi) else f"{hi:.0f}"
        metrics[f"response_{lo:.0f}_{hi_str}"] = resp

    return metrics


def compute_trigger_metrics(
    gen_xy: np.ndarray, reco_pt: np.ndarray, label: str = ""
) -> Dict[str, float]:
    """Compute trigger-related metrics (ROC/AUC).

    Signal: events with genMET > 200 GeV.
    Background: events with genMET < 50 GeV.

    Returns dict with 'auc', plus arrays for ROC plotting stored under
    special keys.
    """
    gen_pt = np.sqrt(gen_xy[:, 0] ** 2 + gen_xy[:, 1] ** 2)

    signal_mask = gen_pt > 200.0
    background_mask = gen_pt < 50.0

    n_signal = np.sum(signal_mask)
    n_background = np.sum(background_mask)

    if n_signal < 10 or n_background < 10:
        logger.warning(f"Too few signal ({n_signal}) or background ({n_background}) events")
        return {"auc": float("nan"), "roc_thresholds": np.array([]), "roc_sig_eff": np.array([]), "roc_bg_eff": np.array([])}

    reco_pt_signal = reco_pt[signal_mask]
    reco_pt_background = reco_pt[background_mask]

    thresholds = np.arange(0, 501, 1.0)
    sig_eff = np.array([np.mean(reco_pt_signal > t) for t in thresholds])
    bg_eff = np.array([np.mean(reco_pt_background > t) for t in thresholds])

    # AUC via trapezoidal rule (integrate sig_eff vs bg_eff)
    # Sort by bg_eff ascending for proper integration
    sort_idx = np.argsort(bg_eff)
    auc = np.trapz(sig_eff[sort_idx], bg_eff[sort_idx])

    return {
        "auc": auc,
        "roc_thresholds": thresholds,
        "roc_sig_eff": sig_eff,
        "roc_bg_eff": bg_eff,
    }


def compute_turn_on(
    gen_xy: np.ndarray, reco_pt: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute turn-on curve: signal efficiency vs gen MET at a fixed threshold.

    Returns (bin_centers, efficiencies, errors).
    """
    gen_pt = np.sqrt(gen_xy[:, 0] ** 2 + gen_xy[:, 1] ** 2)

    bins = np.arange(0, 601, 20)
    centers = 0.5 * (bins[:-1] + bins[1:])
    efficiencies = np.zeros(len(centers))
    errors = np.zeros(len(centers))

    for i in range(len(centers)):
        mask = (gen_pt >= bins[i]) & (gen_pt < bins[i + 1])
        n = np.sum(mask)
        if n > 0:
            eff = np.mean(reco_pt[mask] > threshold)
            efficiencies[i] = eff
            # Binomial error
            errors[i] = np.sqrt(eff * (1 - eff) / n) if n > 1 else 0
        else:
            efficiencies[i] = 0
            errors[i] = 0

    return centers, efficiencies, errors


def find_threshold_for_rate(bg_reco_pt: np.ndarray, target_rate_fraction: float) -> float:
    """Find the MET threshold that gives a target background rate fraction.

    For 30 kHz at 40 MHz L1 rate, target_rate_fraction = 30e3 / 40e6 = 7.5e-4.
    """
    thresholds = np.arange(0, 501, 0.5)
    for t in thresholds:
        rate = np.mean(bg_reco_pt > t)
        if rate <= target_rate_fraction:
            return t
    return thresholds[-1]


# ============================================================================
# Phase 3: Plotting
# ============================================================================


def plot_roc_multi(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_dir: str,
) -> None:
    """Plot ROC curves for multiple models + PUPPI on the same figure.

    roc_data maps label -> (sig_eff, bg_eff, auc).
    """
    plt.figure(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (label, (sig_eff, bg_eff, auc_val)) in enumerate(roc_data.items()):
        if label == "PUPPI":
            plt.plot(sig_eff, bg_eff, label=f"{label} (AUC={auc_val:.4f})",
                     linewidth=2.5, color="red", linestyle="--")
        else:
            plt.plot(sig_eff, bg_eff, label=f"{label} (AUC={auc_val:.4f})",
                     linewidth=1.5, color=colors[i])

    plt.xlabel("Signal Efficiency (genMET > 200 GeV)", fontsize=14)
    plt.ylabel("Background Efficiency (genMET < 50 GeV)", fontsize=14)
    plt.title("ROC Curves: Model Architectures vs PUPPI", fontsize=16)
    plt.yscale("log")
    plt.xlim(0, 1.05)
    plt.ylim(1e-4, 1.1)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "roc_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Multi-ROC plot saved to {out_path}")


def plot_resolution_vs_params(
    summary_data: List[Dict], output_dir: str
) -> None:
    """Plot MET pT resolution (with error bars from seeds) vs parameter count."""
    plt.figure(figsize=(10, 7))

    configs_seen = {}
    for row in summary_data:
        name = row["config"]
        if name not in configs_seen:
            configs_seen[name] = {"params": [], "res": []}
        configs_seen[name]["params"].append(row["params"])
        configs_seen[name]["res"].append(row["met_pt_resolution"])

    names = []
    param_counts = []
    mean_res = []
    std_res = []

    for name, data in configs_seen.items():
        names.append(name)
        param_counts.append(np.mean(data["params"]))
        mean_res.append(np.mean(data["res"]))
        std_res.append(np.std(data["res"]))

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, (n, p, m, s) in enumerate(zip(names, param_counts, mean_res, std_res)):
        plt.errorbar(p, m, yerr=s, fmt="o", markersize=10, capsize=5,
                     color=colors[i], label=n, linewidth=2)

    plt.xlabel("Parameter Count", fontsize=14)
    plt.ylabel("MET pT Resolution [GeV]", fontsize=14)
    plt.title("Resolution vs Model Size (3 seeds per config)", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "resolution_vs_params.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Resolution vs params plot saved to {out_path}")


def plot_auc_vs_params(
    summary_data: List[Dict], output_dir: str
) -> None:
    """Plot AUC (with error bars from seeds) vs parameter count."""
    plt.figure(figsize=(10, 7))

    configs_seen = {}
    for row in summary_data:
        name = row["config"]
        if name not in configs_seen:
            configs_seen[name] = {"params": [], "auc": []}
        configs_seen[name]["params"].append(row["params"])
        configs_seen[name]["auc"].append(row["auc"])

    names = []
    param_counts = []
    mean_auc = []
    std_auc = []

    for name, data in configs_seen.items():
        valid_aucs = [a for a in data["auc"] if not np.isnan(a)]
        names.append(name)
        param_counts.append(np.mean(data["params"]))
        mean_auc.append(np.mean(valid_aucs) if valid_aucs else float("nan"))
        std_auc.append(np.std(valid_aucs) if len(valid_aucs) > 1 else 0.0)

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, (n, p, m, s) in enumerate(zip(names, param_counts, mean_auc, std_auc)):
        plt.errorbar(p, m, yerr=s, fmt="s", markersize=10, capsize=5,
                     color=colors[i], label=n, linewidth=2)

    plt.xlabel("Parameter Count", fontsize=14)
    plt.ylabel("AUC", fontsize=14)
    plt.title("Trigger AUC vs Model Size (3 seeds per config)", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "auc_vs_params.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"AUC vs params plot saved to {out_path}")


# ============================================================================
# Phase 4: Summary CSV
# ============================================================================


def write_summary_csv(results: List[Dict], output_dir: str) -> None:
    """Write per-model and aggregate summary CSV."""
    # Per-model CSV
    fieldnames = [
        "config", "seed", "params", "met_pt_resolution", "met_x_resolution",
        "met_y_resolution", "mean_response", "phi_resolution", "auc",
    ]
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info(f"Per-model results saved to {csv_path}")

    # Aggregate summary (mean +/- std across seeds)
    configs_seen = {}
    for row in results:
        name = row["config"]
        if name not in configs_seen:
            configs_seen[name] = []
        configs_seen[name].append(row)

    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    summary_fields = [
        "config", "params", "n_seeds",
        "met_pt_res_mean", "met_pt_res_std",
        "met_x_res_mean", "met_x_res_std",
        "met_y_res_mean", "met_y_res_std",
        "mean_response_mean", "mean_response_std",
        "phi_res_mean", "phi_res_std",
        "auc_mean", "auc_std",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for name, rows in configs_seen.items():
            def _stat(key):
                vals = [r[key] for r in rows if not np.isnan(r.get(key, float("nan")))]
                if vals:
                    return np.mean(vals), np.std(vals)
                return float("nan"), float("nan")

            res_m, res_s = _stat("met_pt_resolution")
            x_m, x_s = _stat("met_x_resolution")
            y_m, y_s = _stat("met_y_resolution")
            resp_m, resp_s = _stat("mean_response")
            phi_m, phi_s = _stat("phi_resolution")
            auc_m, auc_s = _stat("auc")

            writer.writerow({
                "config": name,
                "params": rows[0]["params"],
                "n_seeds": len(rows),
                "met_pt_res_mean": f"{res_m:.4f}",
                "met_pt_res_std": f"{res_s:.4f}",
                "met_x_res_mean": f"{x_m:.4f}",
                "met_x_res_std": f"{x_s:.4f}",
                "met_y_res_mean": f"{y_m:.4f}",
                "met_y_res_std": f"{y_s:.4f}",
                "mean_response_mean": f"{resp_m:.4f}",
                "mean_response_std": f"{resp_s:.4f}",
                "phi_res_mean": f"{phi_m:.4f}",
                "phi_res_std": f"{phi_s:.4f}",
                "auc_mean": f"{auc_m:.4f}",
                "auc_std": f"{auc_s:.4f}",
            })

    logger.info(f"Summary saved to {summary_path}")


def print_results_table(results: List[Dict], puppi_metrics: Dict) -> None:
    """Print a formatted results table to stdout."""
    print("\n" + "=" * 130)
    print("EVALUATION RESULTS — Test Set Physics Metrics")
    print("=" * 130)
    header = (
        f"{'Config':<25} {'Seed':<6} {'Params':>8} "
        f"{'pT Res':>8} {'X Res':>8} {'Y Res':>8} "
        f"{'Response':>9} {'Phi Res':>8} {'AUC':>8}"
    )
    print(header)
    print("-" * 130)

    for r in results:
        print(
            f"{r['config']:<25} {r['seed']:<6} {r['params']:>8,} "
            f"{r['met_pt_resolution']:>8.3f} {r['met_x_resolution']:>8.3f} "
            f"{r['met_y_resolution']:>8.3f} {r['mean_response']:>9.4f} "
            f"{r['phi_resolution']:>8.4f} {r['auc']:>8.4f}"
        )

    print("-" * 130)
    print(
        f"{'PUPPI baseline':<25} {'---':<6} {'---':>8} "
        f"{puppi_metrics['met_pt_resolution']:>8.3f} "
        f"{puppi_metrics['met_x_resolution']:>8.3f} "
        f"{puppi_metrics['met_y_resolution']:>8.3f} "
        f"{puppi_metrics['mean_response']:>9.4f} "
        f"{puppi_metrics['phi_resolution']:>8.4f} "
        f"{puppi_metrics.get('auc', float('nan')):>8.4f}"
    )
    print("=" * 130)


# ============================================================================
# Main
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="L1DeepMET comprehensive evaluation")
    p.add_argument("--data-dir", required=True,
                   help="Directory with train.h5, val.h5, test.h5")
    p.add_argument("--output-dir", default="outputs/evaluation",
                   help="Root output directory")
    p.add_argument("--epochs", type=int, default=50,
                   help="Max training epochs per model (default: 50)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size (default: 256)")
    p.add_argument("--normfac", type=float, default=100.0,
                   help="MET normalization factor (default: 100)")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip retraining, only evaluate existing models")
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

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    plots_dir = os.path.join(output_dir, "plots")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info(f"Loading data from {args.data_dir}")
    loader = H5DataLoader(args.data_dir)

    # Load raw test data for PUPPI baseline and metrics
    test_features, test_targets = loader.load_data("test")
    # Targets in H5 are already in GeV — no normfac multiplication needed.
    # (normfac division only happens inside create_tf_dataset for training.)
    gen_xy = test_targets
    puppi_xy = compute_puppi_met(test_features)

    logger.info(f"Test set: {test_features.shape[0]} events")
    logger.info(f"Gen MET range: {np.sqrt(gen_xy[:, 0]**2 + gen_xy[:, 1]**2).mean():.1f} GeV mean")

    # ------------------------------------------------------------------
    # PUPPI baseline metrics
    # ------------------------------------------------------------------
    logger.info("Computing PUPPI baseline metrics...")
    puppi_metrics = compute_metrics(gen_xy, puppi_xy, label="PUPPI")

    puppi_pt = np.sqrt(puppi_xy[:, 0] ** 2 + puppi_xy[:, 1] ** 2)
    puppi_trigger = compute_trigger_metrics(gen_xy, puppi_pt, label="PUPPI")
    puppi_metrics["auc"] = puppi_trigger["auc"]

    logger.info(f"PUPPI baseline: pT_res={puppi_metrics['met_pt_resolution']:.3f} GeV, "
                f"AUC={puppi_metrics['auc']:.4f}")

    # ------------------------------------------------------------------
    # Phase 1: Retrain models
    # ------------------------------------------------------------------
    if not args.skip_training:
        logger.info("=" * 80)
        logger.info(f"PHASE 1: Retraining top architectures ({len(TOP_CONFIGS)} configs x {len(SEEDS)} seeds = {len(TOP_CONFIGS)*len(SEEDS)} runs)")
        logger.info("=" * 80)

        train_ds = loader.create_tf_dataset(
            "train", batch_size=args.batch_size, shuffle=True, normfac=NORMFAC
        )
        val_ds = loader.create_tf_dataset(
            "val", batch_size=args.batch_size, shuffle=False, normfac=NORMFAC
        )

        total_runs = len(TOP_CONFIGS) * len(SEEDS)
        run_idx = 0
        for cfg in TOP_CONFIGS:
            for seed in SEEDS:
                run_idx += 1
                logger.info(f"\n[{run_idx}/{total_runs}] {cfg.name} seed={seed}")
                try:
                    retrain_model(cfg, seed, train_ds, val_ds, args.epochs, output_dir)
                except Exception as e:
                    logger.error(f"Failed to train {cfg.name} seed={seed}: {e}")
                    traceback.print_exc()
                    continue

        # Free training data
        del train_ds, val_ds
        gc.collect()

    # ------------------------------------------------------------------
    # Phase 2: Test-set evaluation
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("PHASE 2: Test-set evaluation with physics metrics")
    logger.info("=" * 80)

    all_results: List[Dict] = []
    best_models: Dict[str, Dict] = {}  # config_name -> best seed's data

    for cfg in TOP_CONFIGS:
        param_count = build_model(cfg).count_params()
        gc.collect()

        best_auc_for_config = -1.0
        for seed in SEEDS:
            run_name = f"{cfg.name}_seed{seed}"
            model_path = os.path.join(output_dir, "models", run_name, "best_model.keras")

            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}, skipping")
                continue

            logger.info(f"Evaluating {run_name}...")
            try:
                pred_xy = run_inference(model_path, cfg, test_features)
            except Exception as e:
                logger.error(f"Inference failed for {run_name}: {e}")
                traceback.print_exc()
                continue

            # Physics metrics
            metrics = compute_metrics(gen_xy, pred_xy, label=run_name)

            # Trigger metrics
            reco_pt = np.sqrt(pred_xy[:, 0] ** 2 + pred_xy[:, 1] ** 2)
            trigger = compute_trigger_metrics(gen_xy, reco_pt, label=run_name)
            metrics["auc"] = trigger["auc"]

            result = {
                "config": cfg.name,
                "seed": seed,
                "params": param_count,
                **metrics,
            }
            all_results.append(result)

            logger.info(
                f"  {run_name}: pT_res={metrics['met_pt_resolution']:.3f}, "
                f"AUC={metrics['auc']:.4f}"
            )

            # Track best seed for this config (by AUC)
            if not np.isnan(metrics["auc"]) and metrics["auc"] > best_auc_for_config:
                best_auc_for_config = metrics["auc"]
                best_models[cfg.name] = {
                    "seed": seed,
                    "pred_xy": pred_xy,
                    "reco_pt": reco_pt,
                    "trigger": trigger,
                    "metrics": metrics,
                    "model_path": model_path,
                }

    # ------------------------------------------------------------------
    # Phase 3: Generate plots
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("PHASE 3: Generating plots")
    logger.info("=" * 80)

    # 3a. MakePlots for each config's best seed
    for cfg_name, best in best_models.items():
        cfg_plot_dir = os.path.join(plots_dir, cfg_name)
        os.makedirs(cfg_plot_dir, exist_ok=True)
        path_out = cfg_plot_dir + "/"

        logger.info(f"Generating MakePlots for {cfg_name} (seed={best['seed']})...")
        try:
            MakePlots(gen_xy, best["pred_xy"], puppi_xy, path_out)
        except Exception as e:
            logger.error(f"MakePlots failed for {cfg_name}: {e}")
            traceback.print_exc()

    # 3b. Multi-model ROC comparison
    roc_data = {}
    for cfg_name, best in best_models.items():
        trig = best["trigger"]
        if len(trig["roc_sig_eff"]) > 0:
            roc_data[cfg_name] = (trig["roc_sig_eff"], trig["roc_bg_eff"], trig["auc"])

    # Add PUPPI
    if len(puppi_trigger["roc_sig_eff"]) > 0:
        roc_data["PUPPI"] = (
            puppi_trigger["roc_sig_eff"],
            puppi_trigger["roc_bg_eff"],
            puppi_trigger["auc"],
        )

    if roc_data:
        plot_roc_multi(roc_data, plots_dir)

    # 3c. Turn-on curves for the overall best model vs PUPPI
    if best_models:
        # Find the config with highest AUC
        best_cfg_name = max(best_models, key=lambda k: best_models[k]["metrics"].get("auc", -1))
        best_overall = best_models[best_cfg_name]

        gen_pt_all = np.sqrt(gen_xy[:, 0] ** 2 + gen_xy[:, 1] ** 2)
        bg_mask = gen_pt_all < 50.0
        target_rate_fraction = 30e3 / 40e6  # 30 kHz at 40 MHz

        # Find thresholds for 30 kHz equivalent
        ml_threshold = find_threshold_for_rate(best_overall["reco_pt"][bg_mask], target_rate_fraction)
        puppi_threshold = find_threshold_for_rate(puppi_pt[bg_mask], target_rate_fraction)

        logger.info(f"Turn-on thresholds: ML={ml_threshold:.1f} GeV, PUPPI={puppi_threshold:.1f} GeV")

        ml_centers, ml_eff, ml_err = compute_turn_on(gen_xy, best_overall["reco_pt"], ml_threshold)
        puppi_centers, puppi_eff, puppi_err = compute_turn_on(gen_xy, puppi_pt, puppi_threshold)

        try:
            plot_turn_on_curves(
                ml_eff, ml_err, ml_threshold,
                puppi_eff, puppi_err, puppi_threshold,
                ml_centers,
                plots_dir,
                "VBFHInv", "TT",
            )
        except Exception as e:
            logger.error(f"Turn-on plot failed: {e}")
            traceback.print_exc()

    # 3d. Resolution vs params
    if all_results:
        plot_resolution_vs_params(all_results, plots_dir)
        plot_auc_vs_params(all_results, plots_dir)

    # ------------------------------------------------------------------
    # Phase 4: Summary CSV and printout
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("PHASE 4: Writing summary")
    logger.info("=" * 80)

    if all_results:
        write_summary_csv(all_results, output_dir)
        print_results_table(all_results, puppi_metrics)
    else:
        logger.warning("No results to summarize -- all models may have failed.")

    # Write PUPPI baseline to its own row in the CSV for reference
    puppi_csv_path = os.path.join(output_dir, "puppi_baseline.csv")
    with open(puppi_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(puppi_metrics.keys()))
        writer.writeheader()
        writer.writerow(puppi_metrics)
    logger.info(f"PUPPI baseline saved to {puppi_csv_path}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
