#!/usr/bin/env python3
"""QAT validation experiment driver.

Trains and evaluates **the same architecture** under two regimes — full
precision (FP32) and HGQ2 quantization-aware training — and compares each
model in Keras (full-precision inference) vs hls4ml bit-accurate
fixed-point simulation. The expected outcome (the "QAT thesis"):

  - FP32 model: large gap between Keras and hls4ml (the precision-degradation
    finding from earlier experiments).
  - HGQ2 model: small or zero gap between Keras and hls4ml, because the
    learned bit-widths were trained against.

Architecture: bounded scalar weight head, w=64, d=3, embeddings — the
saved residual-ablation winner. Loss: MAE only, no xy_balance.

Outputs land under ``--output-dir``:

  <output-dir>/
    fp32/
        best_model.keras
        result.json
        hls/  ← full hls4ml project + validation.json + pairplots
    hgq2/
        best_model.keras
        result.json
        hls/  ← full hls4ml project + validation.json + pairplots
    comparison.json    ← four physics cards in one place
    report.md          ← human-readable comparison report

Usage::

    python scripts/experiments/qat_validation.py \\
        --output-dir reports/qat_validation_may2026 \\
        --epochs 30 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import h5py  # type: ignore  # noqa: E402
import numpy as np  # type: ignore  # noqa: E402
import tensorflow as tf  # type: ignore  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(1)

logger = logging.getLogger(__name__)


# ─── FP32 training (delegates to scripts.ablation) ──────────────────────────

def train_fp32(
    output_dir: Path,
    *,
    data_dir: Path,
    epochs: int,
    seed: int,
    name: str = "fp32",
) -> Path:
    """Train the FP32 model via the existing ablation runner.

    Reuses the recipe ``combined_best.bounded_mae_only_xy0`` — the loss-and-
    output-head-ablation winner on the great-ishizaka branch.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "ablation.py"),
        "--recipe", "combined_best",
        "--cells", "bounded_mae_only_xy0",
        "--seeds", str(seed),
        "--epochs", str(epochs),
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
    ]
    logger.info("FP32 training: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    run_dir = output_dir / f"bounded_mae_only_xy0_seed{seed}"
    model_path = run_dir / "best_model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"FP32 training did not produce {model_path}")
    return run_dir


# ─── HGQ2 training (delegates to scripts.train_hgq2) ────────────────────────

def train_hgq2(
    output_dir: Path,
    *,
    data_dir: Path,
    epochs: int,
    seed: int,
    name: str = "hgq2",
) -> Path:
    """Train the HGQ2 model via ``scripts/train_hgq2.py``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train_hgq2.py"),
        "--output-dir", str(output_dir),
        "--name", name,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--data-dir", str(data_dir),
        # Architecture matches FP32:
        "--width", "64", "--depth", "3",
        "--use-embeddings", "--bounded-weight",
        # Loss recipe matches FP32:
        "--mae-weight", "1.0", "--mse-weight", "0.0",
        "--xy-balance-weight", "0.0", "--binned-weight", "0.0",
    ]
    logger.info("HGQ2 training: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    model_path = output_dir / "best_model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"HGQ2 training did not produce {model_path}")
    return output_dir


# ─── HLS conversion + bit-accurate validation ──────────────────────────────

def convert_and_validate_fp32(
    model_dir: Path,
    *,
    data_dir: Path,
    n_validation_events: int,
) -> dict:
    """FP32 → HLS via the existing ``scripts/synthesize.py``."""
    model_path = model_dir / "best_model.keras"
    hls_dir = model_dir / "hls"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "synthesize.py"),
        "--model", str(model_path),
        "--output-dir", str(hls_dir),
        "--n-validation-events", str(n_validation_events),
        "--test-h5", str(data_dir / "test.h5"),
    ]
    logger.info("FP32 HLS synth: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    with open(hls_dir / "validation.json") as f:
        return json.load(f)


def convert_and_validate_hgq2(
    model_dir: Path,
    *,
    data_dir: Path,
    n_validation_events: int,
) -> dict:
    """HGQ2 → HLS via ``scripts/synthesize.py`` with the export pass disabled.

    Why ``--no-export-pass``: the HGQ2 model is already built from hls4ml-
    friendly primitives (QDense + QBatchNormalization + QActivation +
    QGlobalAveragePooling1D + QMultiply + fixed QDense). The export rewrite
    only applies to models with custom CastToInt / BoundedWeight /
    SumOverParticles layers from the full-precision path.
    """
    model_path = model_dir / "best_model.keras"
    hls_dir = model_dir / "hls"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "synthesize.py"),
        "--model", str(model_path),
        "--output-dir", str(hls_dir),
        "--no-export-pass",
        "--n-validation-events", str(n_validation_events),
        "--test-h5", str(data_dir / "test.h5"),
    ]
    logger.info("HGQ2 HLS synth: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    with open(hls_dir / "validation.json") as f:
        return json.load(f)


# ─── Report writer ──────────────────────────────────────────────────────────

PHYSICS_KEYS = [
    ("met_x_resolution", "X resolution (IQR/2) [GeV]"),
    ("met_y_resolution", "Y resolution (IQR/2) [GeV]"),
    ("met_pt_resolution", "pT resolution (IQR/2) [GeV]"),
    ("phi_resolution", "φ resolution (IQR/2) [rad]"),
    ("auc", "AUC (signal > 200 vs bg < 50)"),
    ("mean_response", "Mean response (avg over bins)"),
]


def _fmt(val, key) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "  —  "
    if "auc" in key or "response" in key:
        return f"{val:.4f}"
    return f"{val:.2f}"


def write_report(
    output_dir: Path,
    *,
    fp32_validation: dict,
    hgq2_validation: dict,
    config: dict,
) -> Path:
    """Produce a Markdown comparison report at ``output_dir/report.md``."""
    out_path = output_dir / "report.md"
    lines = []

    L = lines.append
    L("# QAT validation: FP32 vs HGQ2 (May 2026)")
    L("")
    L("> Tests the QAT thesis: training the model with quantization-aware "
      "bit-width learning (HGQ2) should give a Keras vs hls4ml bit-accurate "
      "agreement that the float-trained baseline does not.")
    L("")
    L("## Experimental setup")
    L("")
    L(f"- Architecture: bounded scalar weight head, width={config['width']}, "
      f"depth={config['depth']}, embeddings enabled.")
    L(f"- Loss: MAE only (no xy_balance, no BinnedDeviation). Matches the "
      f"loss-form-ablation winner on `great-ishizaka`.")
    L(f"- Optimizer: AdamW (lr=1e-3, clipnorm=1.0).")
    L(f"- Training: up to {config['epochs']} epochs, batch size 256, "
      f"EarlyStopping patience 10 on val_loss.")
    L(f"- Seed: {config['seed']} (TF + numpy).")
    L(f"- Validation events: {config['n_validation_events']} from test split.")
    L(f"- Data: `{config['data_dir']}`.")
    L("")
    L("## Comparison: full physics card")
    L("")
    L("Four cells per metric. The interesting numbers are the Keras→hls4ml "
      "deltas: small Δ means QAT worked, large Δ means precision was lost.")
    L("")
    header = "| Metric | FP32 Keras | FP32 hls4ml | Δ (hls − keras) | HGQ2 Keras | HGQ2 hls4ml | Δ (hls − keras) |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    L(header)
    L(sep)
    for key, label in PHYSICS_KEYS:
        fp32_k = fp32_validation["keras_card"].get(key)
        fp32_h = fp32_validation["hls_card"].get(key)
        hgq2_k = hgq2_validation["keras_card"].get(key)
        hgq2_h = hgq2_validation["hls_card"].get(key)
        fp32_d = (fp32_h - fp32_k) if (fp32_k is not None and fp32_h is not None
                                       and not np.isnan(fp32_k) and not np.isnan(fp32_h)) else None
        hgq2_d = (hgq2_h - hgq2_k) if (hgq2_k is not None and hgq2_h is not None
                                       and not np.isnan(hgq2_k) and not np.isnan(hgq2_h)) else None
        row = (
            f"| {label} | "
            f"{_fmt(fp32_k, key)} | {_fmt(fp32_h, key)} | "
            f"{_fmt(fp32_d, key) if fp32_d is not None else '  —  '} | "
            f"{_fmt(hgq2_k, key)} | {_fmt(hgq2_h, key)} | "
            f"{_fmt(hgq2_d, key) if hgq2_d is not None else '  —  '} |"
        )
        L(row)
    L("")
    L("## PUPPI reference (baseline at the architecture's identity init)")
    L("")
    L(f"- PUPPI MET X IQR/2 = {fp32_validation['keras_card'].get('puppi_met_x_resolution', float('nan')):.2f} GeV")
    L(f"- PUPPI MET Y IQR/2 = {fp32_validation['keras_card'].get('puppi_met_y_resolution', float('nan')):.2f} GeV")
    L(f"- PUPPI MET pT IQR/2 = {fp32_validation['keras_card'].get('puppi_met_pt_resolution', float('nan')):.2f} GeV")
    L(f"- PUPPI AUC = {fp32_validation['keras_card'].get('puppi_auc', float('nan')):.4f}")
    L("")
    L("## Artifacts")
    L("")
    L("```")
    L(f"{output_dir.name}/")
    L("├── fp32/")
    L("│   ├── best_model.keras")
    L("│   ├── result.json")
    L("│   └── hls/                    ← HLS C++ project + validation.json + plots")
    L("├── hgq2/")
    L("│   ├── best_model.keras")
    L("│   ├── result.json")
    L("│   └── hls/")
    L("├── comparison.json             ← machine-readable copy of both cards")
    L("└── report.md                   ← this file")
    L("```")
    L("")
    L("## Reproducing this experiment")
    L("")
    L("From the worktree root:")
    L("")
    L("```bash")
    L("source /home/users/dprimosc/micromamba/etc/profile.d/micromamba.sh")
    L("micromamba activate l1deepmet")
    L("")
    L(f"python scripts/experiments/qat_validation.py \\")
    L(f"    --output-dir {output_dir} \\")
    L(f"    --epochs {config['epochs']} \\")
    L(f"    --seed {config['seed']}")
    L("```")
    L("")
    out_path.write_text("\n".join(lines))
    return out_path


# ─── Main driver ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write everything (models + HLS + report).")
    p.add_argument("--data-dir",
                   default="/home/users/dprimosc/L1DeepMET/preprocessed/25Jul8_140X_v0",
                   type=Path)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-validation-events", type=int, default=500)
    p.add_argument("--skip-fp32-training", action="store_true",
                   help="Skip FP32 training (use the model in --output-dir/fp32/ if "
                        "it's already there). Useful for iterating on the HGQ2 path.")
    p.add_argument("--skip-hgq2-training", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    log = logging.getLogger("qat_validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32_dir = args.output_dir / "fp32"
    hgq2_dir = args.output_dir / "hgq2"

    config = {
        "width": 64, "depth": 3,
        "epochs": args.epochs, "seed": args.seed,
        "n_validation_events": args.n_validation_events,
        "data_dir": str(args.data_dir),
    }

    # ── FP32 training ───────────────────────────────────────────────────────
    fp32_run_dir = fp32_dir / f"bounded_mae_only_xy0_seed{args.seed}"
    if args.skip_fp32_training and (fp32_run_dir / "best_model.keras").is_file():
        log.info("Skipping FP32 training, using existing %s", fp32_run_dir)
    else:
        t0 = time.time()
        fp32_run_dir = train_fp32(
            fp32_dir, data_dir=args.data_dir, epochs=args.epochs, seed=args.seed,
        )
        log.info("FP32 training elapsed: %.0fs", time.time() - t0)

    # ── HGQ2 training ───────────────────────────────────────────────────────
    if args.skip_hgq2_training and (hgq2_dir / "best_model.keras").is_file():
        log.info("Skipping HGQ2 training, using existing %s", hgq2_dir)
    else:
        t0 = time.time()
        train_hgq2(
            hgq2_dir, data_dir=args.data_dir, epochs=args.epochs, seed=args.seed,
        )
        log.info("HGQ2 training elapsed: %.0fs", time.time() - t0)

    # ── HLS conversion + validation ─────────────────────────────────────────
    log.info("Converting FP32 to HLS + bit-accurate validation…")
    fp32_validation = convert_and_validate_fp32(
        fp32_run_dir, data_dir=args.data_dir,
        n_validation_events=args.n_validation_events,
    )
    log.info("Converting HGQ2 to HLS + bit-accurate validation…")
    hgq2_validation = convert_and_validate_hgq2(
        hgq2_dir, data_dir=args.data_dir,
        n_validation_events=args.n_validation_events,
    )

    # ── Combined comparison.json ────────────────────────────────────────────
    comparison = {
        "config": config,
        "fp32": {
            "keras_card": fp32_validation.get("keras_card"),
            "hls_card": fp32_validation.get("hls_card"),
            "delta_card": fp32_validation.get("delta_card"),
            "n_events": fp32_validation.get("n_events"),
            "model_path": str((fp32_run_dir / "best_model.keras").resolve()),
        },
        "hgq2": {
            "keras_card": hgq2_validation.get("keras_card"),
            "hls_card": hgq2_validation.get("hls_card"),
            "delta_card": hgq2_validation.get("delta_card"),
            "n_events": hgq2_validation.get("n_events"),
            "model_path": str((hgq2_dir / "best_model.keras").resolve()),
        },
    }
    with open(args.output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # ── Report ──────────────────────────────────────────────────────────────
    report_path = write_report(
        args.output_dir,
        fp32_validation=fp32_validation,
        hgq2_validation=hgq2_validation,
        config=config,
    )
    log.info("Wrote report: %s", report_path)

    # Console summary.
    print()
    print("=" * 78)
    print("QAT validation experiment summary")
    print("=" * 78)
    print(f"{'metric':<22} {'fp32 keras':>10} {'fp32 hls':>10} {'Δ fp32':>9} "
          f"{'hgq2 keras':>11} {'hgq2 hls':>10} {'Δ hgq2':>9}")
    print("-" * 78)
    for key, _ in PHYSICS_KEYS:
        fk = fp32_validation["keras_card"].get(key, float("nan"))
        fh = fp32_validation["hls_card"].get(key, float("nan"))
        hk = hgq2_validation["keras_card"].get(key, float("nan"))
        hh = hgq2_validation["hls_card"].get(key, float("nan"))
        print(f"{key:<22} {fk:>10.4f} {fh:>10.4f} {(fh - fk):>+9.4f} "
              f"{hk:>11.4f} {hh:>10.4f} {(hh - hk):>+9.4f}")
    print()
    print(f"Report:  {report_path}")
    print(f"JSON:    {args.output_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
