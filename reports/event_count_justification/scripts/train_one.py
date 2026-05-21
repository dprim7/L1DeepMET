#!/usr/bin/env python3
"""
One cell of the event-count justification sweep:
  (N_train, variant ∈ {extended, baseline}, seed) → results JSON.

Self-contained — does NOT depend on the project's BaseModel infrastructure
(which is partially built). Reimplements the Dense w64 d3 mode-1 model
functionally to match `src/l1deepmet/models/dense.py` mode 1 exactly:

  feature_stack(continuous) → per-candidate scalar weight w
  output = sum over candidates of w * (px, py)

Usage:
  python train_one.py \\
      --h5-dir outputs/preprocessed/26May21_142_extended_v0 \\
      --n-train 2000 \\
      --variant extended \\
      --seed 42 \\
      --out reports/event_count_justification/results/raw/n2000_extended_seed42.json
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# --- TF thread/CPU/GPU pinning BEFORE importing tensorflow ----------------
# UAF is shared and has a per-process pthread budget; with 8 TF processes
# parallel each spawning ~30 internal threads we hit EAGAIN on pthread_create.
# These envs + the explicit tf.config calls below squeeze TF down to ~5
# threads per process so we can actually parallelise.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")            # CPU only
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")            # silence info/warn
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import h5py                     # type: ignore
import numpy as np              # type: ignore
import tensorflow as tf         # type: ignore

# Belt-and-suspenders: the env vars above set defaults; these calls assert
# them inside the TF runtime. Must be called before any TF op is created
# (so before model build), which is what this top-of-script position gives us.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Targets are normalized by this factor in training. Loss + metric in same units.
NORMFAC = 100.0
N_PARTICLES = 128

# Per-candidate feature slot indices in the H5 (must match params.yaml feature_layout_extended):
# 0..3   pt, eta, phi, puppi_weight     ← original "baseline" continuous
# 4..5   px, py                          ← momentum_inputs (always used)
# 6..7   encoded_pdgId, encoded_charge   ← categorical (unused by Dense)
# 8..25  dxyErr, z0, hw*, track*, calo*, cl*  ← NEW continuous features
CONT_SLOTS_ALL      = [0, 1, 2, 3] + list(range(8, 26))      # 22 continuous slots (extended)
CONT_SLOTS_BASELINE = [0, 1, 2, 3]                            # 4 continuous slots (baseline)
MOM_SLOTS           = [4, 5]                                  # px, py

# Gen-MET bin edges for per-bin metrics (matches BinnedDeviation convention)
PT_BIN_EDGES = [0.0, 50.0, 100.0, 200.0, 300.0, 400.0, np.inf]
PT_BIN_LABELS = [f"{int(PT_BIN_EDGES[i])}_{('inf' if not np.isfinite(PT_BIN_EDGES[i+1]) else int(PT_BIN_EDGES[i+1]))}"
                 for i in range(len(PT_BIN_EDGES)-1)]


# ─── data ──────────────────────────────────────────────────────────────────

def load_h5_split(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        X = f["features"][:]      # (N, 128, 26)
        Y = f["targets"][:]       # (N, 2) in GeV
    return X, Y


def sanitize_features(X: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0 and -999 sentinels with 0.

    The extended preprocessor (load_samples_to_numpy_extended) does NOT
    sanitize features the way the legacy preprocess_data does. We mirror
    the legacy sanitization here so this experiment doesn't conflate
    "model can't learn from new features" with "model crashes on NaN".

    Specifically:
      - clPuId / clEmId from pfCluster.egVs*MVAOut can be NaN (CMSSW
        returns NaN for some candidates where the MVA didn't evaluate).
      - caloEta / caloPhi use -999 as the sentinel for "no track" (the
        `? pfTrack.isNonnull ? … : -999` accessor in the patched recipe).
    """
    X = X.copy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # -999 sentinels for caloEta/Phi (slots 20, 21 in our layout)
    for slot in (20, 21):
        X[:, :, slot] = np.where(X[:, :, slot] <= -900.0, 0.0, X[:, :, slot])
    return X


def compute_feature_scaling(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature robust standardization stats (median, MAD-equiv scale).

    Computed over BOTH event and particle axes, excluding pad zeros to avoid
    pulling the median to zero on sparsely-populated slots. Using `pt > 0`
    as the "this is a real candidate" mask, which is the only slot whose
    zero strictly means padding.
    """
    pt = X_train[:, :, 0]
    mask = pt > 0   # (N, 128) — True where the slot is a real candidate
    medians = np.zeros(X_train.shape[2], dtype=np.float32)
    scales  = np.ones(X_train.shape[2],  dtype=np.float32)
    for j in range(X_train.shape[2]):
        vals = X_train[:, :, j][mask]
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        # MAD-equivalent scale: 1.4826 * MAD gives a normal-equivalent σ
        scale = 1.4826 * float(np.median(np.abs(vals - med)))
        if scale < 1e-6:   # constant feature (e.g., all-zero dxyErr in some samples)
            scale = 1.0
        medians[j] = med
        scales[j] = scale
    return medians, scales


def build_inputs(X_raw: np.ndarray, medians: np.ndarray, scales: np.ndarray,
                 variant: str) -> dict[str, np.ndarray]:
    """Split sanitized H5 features (N, 128, 26) into the model's input dict.

    Continuous features (slots 0..3 + 8..25) are robust-standardized using
    the medians/scales that were computed on the train subset.
    Momentum (slots 4,5) is passed through as raw px/py / NORMFAC so it
    matches the same normalization as the target (gen MET / NORMFAC).

    The "baseline" variant zeros slots 8..25 → model can't use the new
    features. Architecture and param count are identical between variants
    (input shape unchanged).
    """
    n = X_raw.shape[0]
    pt_alive = (X_raw[:, :, 0] > 0).astype(np.float32)   # (N, 128) mask for real candidates

    cont = np.zeros((n, N_PARTICLES, len(CONT_SLOTS_ALL)), dtype=np.float32)

    # Clip standardized values to ±5 to kill extreme-outlier-driven NaN
    # (e.g. one candidate with hwPt 50× the median blows the first dense
    # layer's pre-activation, triggers tanh saturation that destabilises BN
    # moving-stats on subsequent batches). Discovered after 12/15 extended
    # cells at N ≥ 800 collapsed to NaN at epoch 0 in the first sweep.
    CLIP = 5.0

    def fill(j: int, slot: int):
        s = ((X_raw[:, :, slot] - medians[slot]) / scales[slot]) * pt_alive
        cont[:, :, j] = np.clip(s, -CLIP, CLIP)

    # Slots 0..3 always populated (pt, eta, phi, puppi_weight)
    for j, slot in enumerate(CONT_SLOTS_ALL[:4]):
        fill(j, slot)
    if variant == "extended":
        for j, slot in enumerate(CONT_SLOTS_ALL[4:], start=4):
            fill(j, slot)
    elif variant == "baseline":
        pass   # leave slots 4..21 as exact zero
    else:
        raise ValueError(f"unknown variant {variant!r}")

    # Momentum: raw px,py divided by NORMFAC (matches target normalization).
    mom = np.stack([X_raw[:, :, MOM_SLOTS[0]], X_raw[:, :, MOM_SLOTS[1]]],
                   axis=-1).astype(np.float32) / NORMFAC
    return {"continuous_inputs": cont, "momentum_inputs": mom}


# ─── model ─────────────────────────────────────────────────────────────────

def build_model(continuous_dim: int, units=(64, 64, 64), activation="tanh"):
    """Dense w64 d3 mode-1 — per-candidate weight × pxpy, summed.

    Reproduces `src/l1deepmet/models/dense.py::Dense` mode 1 functionally.
    No per-candidate bias (matches `with_bias=False` from final_baseline_apr2026
    which won the apr2026 ablation).
    """
    cont = tf.keras.Input(shape=(N_PARTICLES, continuous_dim), name="continuous_inputs")
    mom  = tf.keras.Input(shape=(N_PARTICLES, 2),               name="momentum_inputs")

    x = cont
    for i, w in enumerate(units):
        x = tf.keras.layers.Dense(w, use_bias=False,
                                  kernel_initializer="lecun_uniform",
                                  name=f"dense_{i}")(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=f"bn_{i}")(x)
        x = tf.keras.layers.Activation(activation, name=f"act_{i}")(x)

    # Per-candidate scalar weight
    weight = tf.keras.layers.Dense(1, activation="linear",
                                   kernel_initializer="lecun_uniform",
                                   name="met_weight")(x)              # (B, N, 1)
    weighted = tf.keras.layers.Multiply(name="weighted_mom")([weight, mom])  # (B, N, 2)
    out = tf.keras.layers.Lambda(lambda z: tf.reduce_sum(z, axis=1),
                                 name="met_sum")(weighted)            # (B, 2)
    return tf.keras.Model(inputs={"continuous_inputs": cont, "momentum_inputs": mom},
                          outputs=out, name="DenseMode1")


# ─── metrics ───────────────────────────────────────────────────────────────

def per_bin_metrics(pred_xy_gev: np.ndarray, true_xy_gev: np.ndarray) -> dict:
    """Compute per-gen-MET-bin MAE + bias + IQR/2 + count.

    All inputs in GeV (i.e., already de-normalized by NORMFAC).
    """
    true_met = np.sqrt(true_xy_gev[:, 0] ** 2 + true_xy_gev[:, 1] ** 2)
    err_x = pred_xy_gev[:, 0] - true_xy_gev[:, 0]
    err_y = pred_xy_gev[:, 1] - true_xy_gev[:, 1]
    abs_err = (np.abs(err_x) + np.abs(err_y)) / 2.0   # per-event MAE on (x,y)

    out = {"overall": {}, "per_bin": {}}
    out["overall"]["n"] = int(true_xy_gev.shape[0])
    out["overall"]["mae_xy"] = float(abs_err.mean())
    out["overall"]["mse_xy"] = float(((err_x ** 2 + err_y ** 2) / 2.0).mean())
    out["overall"]["bias_x"] = float(err_x.mean())
    out["overall"]["bias_y"] = float(err_y.mean())
    out["overall"]["iqr2_x"] = float((np.percentile(err_x, 75) - np.percentile(err_x, 25)) / 2.0)
    out["overall"]["iqr2_y"] = float((np.percentile(err_y, 75) - np.percentile(err_y, 25)) / 2.0)
    out["overall"]["met_pt_iqr2"] = float(
        (np.percentile(np.sqrt(pred_xy_gev[:,0]**2 + pred_xy_gev[:,1]**2) - true_met, 75) -
         np.percentile(np.sqrt(pred_xy_gev[:,0]**2 + pred_xy_gev[:,1]**2) - true_met, 25)) / 2.0
    )

    for i, label in enumerate(PT_BIN_LABELS):
        lo, hi = PT_BIN_EDGES[i], PT_BIN_EDGES[i+1]
        mask = (true_met >= lo) & (true_met < hi)
        n = int(mask.sum())
        if n == 0:
            out["per_bin"][label] = {"n": 0, "mae_xy": None, "bias_x": None, "bias_y": None}
            continue
        out["per_bin"][label] = {
            "n": n,
            "mae_xy": float(abs_err[mask].mean()),
            "bias_x": float(err_x[mask].mean()),
            "bias_y": float(err_y[mask].mean()),
        }
    return out


# ─── train + eval ──────────────────────────────────────────────────────────

def train_and_eval(args, h5_dir: Path, log_fp):
    log_fp.write(f"[train_one] loading H5 from {h5_dir}\n"); log_fp.flush()
    Xtr_full, Ytr_full = load_h5_split(h5_dir / "train.h5")
    Xv,      Yv       = load_h5_split(h5_dir / "val.h5")
    Xte,     Yte      = load_h5_split(h5_dir / "test.h5")

    # Sanitize — kills NaNs in clPuId/clEmId and -999 sentinels in caloEta/Phi.
    # Mirrors what preprocess_data does for the legacy 9-feature path but the
    # extended path skips. Do this BEFORE scaling stats.
    Xtr_full = sanitize_features(Xtr_full)
    Xv       = sanitize_features(Xv)
    Xte      = sanitize_features(Xte)

    rng = np.random.default_rng(args.seed)
    # Subsample train pool
    n_train_avail = Xtr_full.shape[0]
    if args.n_train > n_train_avail:
        log_fp.write(f"[train_one] requested n_train={args.n_train} > available {n_train_avail}; capping\n")
        n_train_eff = n_train_avail
    else:
        n_train_eff = args.n_train
    idx_train = rng.choice(n_train_avail, size=n_train_eff, replace=False)
    Xtr, Ytr = Xtr_full[idx_train], Ytr_full[idx_train]

    log_fp.write(f"[train_one] N train={n_train_eff}, val={Xv.shape[0]}, test={Xte.shape[0]}; variant={args.variant}; seed={args.seed}\n")
    log_fp.flush()

    # Sanity: train/val/test were split deterministically upstream (combine_shuffle_split_extended
    # uses np.random.seed(42)); we're not re-splitting, just subsampling train.

    # Per-feature scaling stats computed on the subsampled train pool
    # (NOT val/test — that would leak the held-out distribution). Stats are
    # stored in the result JSON so the run is reproducible. Robust to
    # outliers via median + MAD-equiv scale.
    medians, scales = compute_feature_scaling(Xtr)
    log_fp.write(f"[train_one] sample feature scales: hwPt med={medians[10]:.2f} scale={scales[10]:.2f}, "
                 f"hwEta med={medians[11]:.2f} scale={scales[11]:.2f}\n")

    # Build inputs — scaling is applied inside build_inputs (continuous slots
    # get (x-med)/scale; momentum gets /NORMFAC). Baseline variant zeros
    # the new feature columns AFTER scaling, so they're true zeros.
    in_tr = build_inputs(Xtr, medians, scales, args.variant)
    in_v  = build_inputs(Xv,  medians, scales, args.variant)
    in_te = build_inputs(Xte, medians, scales, args.variant)

    # Sanity: zeroed columns for baseline
    if args.variant == "baseline":
        n_active_new = int((in_tr["continuous_inputs"][:, :, 4:] != 0).sum())
        assert n_active_new == 0, f"baseline variant has {n_active_new} non-zero entries in new feature slots!"
        log_fp.write(f"[train_one] baseline variant: confirmed new feature slots are zero\n")
    n_active_pt = int((in_tr["continuous_inputs"][:, :, 0] != 0).sum())
    log_fp.write(f"[train_one] sanity: {n_active_pt} non-zero pt entries in train ({n_active_pt/(n_train_eff*N_PARTICLES):.1%} of slots)\n")

    # Normalize targets
    Ytr_n = (Ytr / NORMFAC).astype(np.float32)
    Yv_n  = (Yv  / NORMFAC).astype(np.float32)
    Yte_n = (Yte / NORMFAC).astype(np.float32)

    # Reproducibility
    tf.keras.utils.set_random_seed(args.seed)

    # Model
    cont_dim = in_tr["continuous_inputs"].shape[-1]
    model = build_model(continuous_dim=cont_dim)
    n_params = model.count_params()
    log_fp.write(f"[train_one] model params: {n_params}\n")
    log_fp.flush()

    model.compile(
        # `clipnorm=1.0` caps the total gradient norm at 1; second-of-two
        # safety nets against the NaN-from-step-1 failure mode that hit
        # extended cells at N ≥ 800 in the first sweep. The first net is
        # input clipping at ±5 standardised units (see build_inputs).
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="mae",
    )

    # Train
    t0 = time.time()
    hist = model.fit(
        in_tr, Ytr_n,
        validation_data=(in_v, Yv_n),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )
    train_wall_s = time.time() - t0
    log_fp.write(f"[train_one] training done in {train_wall_s:.1f}s\n")
    log_fp.write(f"[train_one] final epoch: train_loss={hist.history['loss'][-1]:.4f}, val_loss={hist.history['val_loss'][-1]:.4f}\n")
    log_fp.flush()

    # Evaluate on test (in GeV — de-normalize predictions)
    pred_n = model.predict(in_te, batch_size=args.batch_size, verbose=0)
    pred_gev = pred_n * NORMFAC
    metrics = per_bin_metrics(pred_xy_gev=pred_gev, true_xy_gev=Yte)
    val_pred_n = model.predict(in_v, batch_size=args.batch_size, verbose=0)
    val_pred_gev = val_pred_n * NORMFAC
    val_metrics = per_bin_metrics(pred_xy_gev=val_pred_gev, true_xy_gev=Yv)

    # Result dict
    git_sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent.parent.parent.parent),
                              "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    result = {
        "config": {
            "n_train_requested": args.n_train,
            "n_train_effective": n_train_eff,
            "variant": args.variant,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "h5_dir": str(h5_dir),
            "normfac": NORMFAC,
            "continuous_dim": cont_dim,
            "n_params": n_params,
            "git_sha": git_sha,
        },
        "test_metrics": metrics,
        "val_metrics_final": val_metrics,
        "training": {
            "wall_s": train_wall_s,
            "loss_history": [float(x) for x in hist.history["loss"]],
            "val_loss_history": [float(x) for x in hist.history["val_loss"]],
            "loss_final": float(hist.history["loss"][-1]),
            "val_loss_final": float(hist.history["val_loss"][-1]),
        },
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5-dir", required=True, type=Path)
    p.add_argument("--n-train", required=True, type=int)
    p.add_argument("--variant", required=True, choices=["extended", "baseline"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", required=True, type=Path,
                   help="Result JSON path")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".log")
    with log_path.open("w") as log_fp:
        log_fp.write(f"[train_one] args: {vars(args)}\n"); log_fp.flush()
        try:
            result = train_and_eval(args, args.h5_dir, log_fp)
        except Exception as e:
            log_fp.write(f"[train_one] FAILED: {e!r}\n")
            log_fp.flush()
            raise

    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
