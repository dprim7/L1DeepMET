#!/usr/bin/env python3
"""Train a QAT + pruned model and compare it to the FP32 baseline.

Trains the bounded-scalar-weight L1DeepMET model two ways under the same
architecture / loss / seed:

  1. FP32 (full-precision floating-point)
  2. HGQ2 (gradient-based quantization-aware training; bit-widths of zero =
     pruned weights, so this is QAT + pruning in one shot)

Then converts both to hls4ml HLS C++ and runs bit-accurate fixed-point
simulation, so the report can answer:

  - How does the QAT-trained model's *Keras* performance compare to the FP32
    baseline? (QAT slightly trades flexibility for FPGA fidelity.)
  - How does the QAT-trained model's *fixed-point* performance compare to the
    FP32 model's fixed-point performance? (This is the deployment-realistic
    comparison; the QAT model is what would actually ship.)
  - What bit-widths / sparsity did HGQ2 learn? (Indirectly answers "how much
    smaller is the firmware.")

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

# Try to clamp TF threading on shared systems. Will raise if TF has already
# been initialized (e.g. when imported from a test process that touched TF
# earlier); skipping is fine in that case — caller already configured it.
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
except RuntimeError:
    pass

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


def extract_bitwidth_stats(model_path: Path) -> dict:
    """Walk the saved HGQ2 model and summarize learned bit-widths.

    Returns a dict like::

        {
          "per_layer": [
            {"name": "...", "kind": "kernel",  "shape": [..],
             "min": ., "max": ., "mean": ., "n_zero": int, "n_total": int,
             "total_bits": ...},
            ...
          ],
          "total_kernel_bits":    <int>,
          "total_kernel_zeros":   <int>,
          "total_kernel_params":  <int>,
          "sparsity_pct":         <float>,   # zeros / total
          "avg_kernel_bitwidth":  <float>,
        }

    "Total bits" = Σ over kernel elements of learned bit-width — a rough proxy
    for firmware weight memory (the actual FPGA cost also depends on layer
    fanout / fan-in patterns and is reported by Vivado HLS synthesis, but this
    is a useful pre-synth gauge).

    Falls back gracefully if ``model_path`` isn't an HGQ2 model
    (returns ``{"per_layer": [], ...}``).
    """
    import tensorflow as tf
    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))
    from synthesize import load_custom_objects
    co = load_custom_objects()
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=co,
                                            compile=False)
    except Exception as e:
        return {"per_layer": [], "error": f"could not load model: {e}"}

    per_layer = []
    total_kernel_bits = 0
    total_kernel_zeros = 0
    total_kernel_params = 0

    for layer in model.layers:
        # HGQ2 quantizers live on attributes `kq` (kernel), `iq` (input),
        # `bq` (bias). Each has a `.bits` tensor with per-channel values.
        for kind in ("kq", "iq", "bq"):
            q = getattr(layer, kind, None)
            if q is None or not hasattr(q, "bits"):
                continue
            bits = q.bits
            try:
                bits_np = bits.numpy() if hasattr(bits, "numpy") else np.asarray(bits)
            except Exception:
                continue
            if bits_np.size == 0:
                continue
            row = {
                "layer": layer.name,
                "kind": kind,
                "shape": list(bits_np.shape),
                "min": float(bits_np.min()),
                "max": float(bits_np.max()),
                "mean": float(bits_np.mean()),
                "n_zero": int((bits_np <= 0.5).sum()),
                "n_total": int(bits_np.size),
                "sum_bits": float(bits_np.sum()),
            }
            per_layer.append(row)
            if kind == "kq":
                total_kernel_bits += int(bits_np.sum())
                total_kernel_zeros += int((bits_np <= 0.5).sum())
                total_kernel_params += int(bits_np.size)

    sparsity_pct = (100.0 * total_kernel_zeros / total_kernel_params
                     if total_kernel_params else 0.0)
    avg_bitwidth = (total_kernel_bits / total_kernel_params
                     if total_kernel_params else 0.0)
    return {
        "per_layer": per_layer,
        "total_kernel_bits": total_kernel_bits,
        "total_kernel_zeros": total_kernel_zeros,
        "total_kernel_params": total_kernel_params,
        "sparsity_pct": round(sparsity_pct, 2),
        "avg_kernel_bitwidth": round(avg_bitwidth, 3),
    }


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
    bitwidth_stats: dict,
) -> Path:
    """Markdown report focused on the QAT+pruned model and its improvements."""
    out_path = output_dir / "report.md"
    lines = []
    L = lines.append

    # ── Headline ─────────────────────────────────────────────────────────────
    L("# QAT + pruned L1DeepMET model (May 2026)")
    L("")
    L("> Trains a quantization-aware + pruned version of the bounded scalar "
      "weight model via HGQ2 and reports how it compares to the full-precision "
      "baseline at the same architecture. The deliverable is **a deployable "
      "QAT+pruned model** (the ``hgq2/`` folder), not a methodology paper.")
    L("")

    hgq2_k = hgq2_validation["keras_card"]
    hgq2_h = hgq2_validation["hls_card"]
    fp32_k = fp32_validation["keras_card"]
    fp32_h = fp32_validation["hls_card"]

    # ── Bottom-line numbers up top ──────────────────────────────────────────
    L("## Headline")
    L("")
    L("| | X IQR/2 | pT IQR/2 | AUC | Sparsity | Avg kernel bits |")
    L("|---|---:|---:|---:|---:|---:|")
    L(f"| **FP32 baseline (Keras inference)** | "
      f"{_fmt(fp32_k.get('met_x_resolution'),'x')} | "
      f"{_fmt(fp32_k.get('met_pt_resolution'),'pt')} | "
      f"{_fmt(fp32_k.get('auc'),'auc')} | 0% | 32 (float) |")
    L(f"| **FP32 baseline (hls4ml fixed-point)** | "
      f"{_fmt(fp32_h.get('met_x_resolution'),'x')} | "
      f"{_fmt(fp32_h.get('met_pt_resolution'),'pt')} | "
      f"{_fmt(fp32_h.get('auc'),'auc')} | 0% | ap_fixed<32,16> |")
    L(f"| **HGQ2 QAT+pruned (Keras inference)** | "
      f"{_fmt(hgq2_k.get('met_x_resolution'),'x')} | "
      f"{_fmt(hgq2_k.get('met_pt_resolution'),'pt')} | "
      f"{_fmt(hgq2_k.get('auc'),'auc')} | "
      f"{bitwidth_stats.get('sparsity_pct', 0):.1f}% | "
      f"{bitwidth_stats.get('avg_kernel_bitwidth', 0):.2f} |")
    L(f"| **HGQ2 QAT+pruned (hls4ml fixed-point)** | "
      f"{_fmt(hgq2_h.get('met_x_resolution'),'x')} | "
      f"{_fmt(hgq2_h.get('met_pt_resolution'),'pt')} | "
      f"{_fmt(hgq2_h.get('auc'),'auc')} | "
      f"{bitwidth_stats.get('sparsity_pct', 0):.1f}% | "
      f"learned (see below) |")
    L("")
    L("Two comparisons worth pulling out from the table:")
    L("")
    if all(np.isfinite([fp32_k.get('met_x_resolution', np.nan),
                         hgq2_k.get('met_x_resolution', np.nan)])):
        keras_dx = hgq2_k['met_x_resolution'] - fp32_k['met_x_resolution']
        L(f"- **QAT cost in Keras**: HGQ2 vs FP32 X resolution = "
          f"{keras_dx:+.2f} GeV. (Positive = QAT slightly worse than FP32 in "
          f"floating-point; this is the price of training under bit-width "
          f"constraints.)")
    if all(np.isfinite([fp32_h.get('met_x_resolution', np.nan),
                         hgq2_h.get('met_x_resolution', np.nan)])):
        hls_dx = hgq2_h['met_x_resolution'] - fp32_h['met_x_resolution']
        L(f"- **What ships to the FPGA**: HGQ2 hls4ml vs FP32 hls4ml X "
          f"resolution = {hls_dx:+.2f} GeV. (This is the realistic "
          f"deployment comparison; HGQ2 was trained against the precision "
          f"loss the FP32 model suffers post-training quantization.)")
    L("")

    # ── Experimental setup ──────────────────────────────────────────────────
    L("## Setup")
    L("")
    L(f"- Architecture: bounded scalar weight head, width={config['width']}, "
      f"depth={config['depth']}, embeddings enabled. Same shape both regimes.")
    L("- Loss: MAE only, no xy_balance, no BinnedDeviation. (Matches the loss-"
      "form-ablation winner on `great-ishizaka`.)")
    L("- Optimizer: AdamW (lr=1e-3, clipnorm=1.0).")
    L(f"- Training: up to {config['epochs']} epochs, batch size 256, "
      f"EarlyStopping patience 10 on val_loss.")
    L(f"- Seed: {config['seed']} (TF + numpy).")
    L(f"- Validation events: {config['n_validation_events']} from the test split.")
    L(f"- Data: `{config['data_dir']}`.")
    L("- HGQ2 path: pdgid / charge inputs pre-encoded one-hot (hls4ml cannot "
      "synthesize ``tf.one_hot`` in-graph). Activations via "
      "``hgq.layers.activation.Activation``; sum-over-particles via "
      "``QGlobalAveragePooling1D`` + fixed ``QDense(N·I)`` "
      "(workarounds documented in `src/l1deepmet/quantization/README.md`).")
    L("")

    # ── Full physics card ───────────────────────────────────────────────────
    L("## Full physics card")
    L("")
    L("| Metric | FP32 Keras | FP32 hls4ml | Δ (hls − keras) | HGQ2 Keras | HGQ2 hls4ml | Δ (hls − keras) |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    for key, label in PHYSICS_KEYS:
        fk = fp32_k.get(key); fh = fp32_h.get(key)
        hk = hgq2_k.get(key); hh = hgq2_h.get(key)
        fd = (fh - fk) if (fk is not None and fh is not None
                            and np.isfinite(fk) and np.isfinite(fh)) else None
        hd = (hh - hk) if (hk is not None and hh is not None
                            and np.isfinite(hk) and np.isfinite(hh)) else None
        L(f"| {label} | "
          f"{_fmt(fk, key)} | {_fmt(fh, key)} | {_fmt(fd, key) if fd is not None else '  —  '} | "
          f"{_fmt(hk, key)} | {_fmt(hh, key)} | {_fmt(hd, key) if hd is not None else '  —  '} |")
    L("")
    L("PUPPI MET (reference): "
      f"X = {fp32_k.get('puppi_met_x_resolution', float('nan')):.2f}, "
      f"pT = {fp32_k.get('puppi_met_pt_resolution', float('nan')):.2f}, "
      f"AUC = {fp32_k.get('puppi_auc', float('nan')):.4f}.")
    L("")

    # ── Bit-width / pruning analysis ────────────────────────────────────────
    L("## Learned bit-widths and pruning")
    L("")
    if bitwidth_stats.get("per_layer"):
        L(f"- Total kernel parameters: **{bitwidth_stats['total_kernel_params']:,}**")
        L(f"- Kernel parameters at bit-width 0 (pruned): "
          f"**{bitwidth_stats['total_kernel_zeros']:,}** "
          f"({bitwidth_stats['sparsity_pct']:.1f}%)")
        L(f"- Average non-pruned kernel bit-width: "
          f"**{bitwidth_stats['avg_kernel_bitwidth']:.2f}** bits")
        L(f"- Total kernel bits (Σ learned bit-widths over all elements): "
          f"**{bitwidth_stats['total_kernel_bits']:,}**")
        L("")
        L("Per-layer breakdown (kernel quantizer only; input/bias quantizers in `comparison.json`):")
        L("")
        L("| Layer | Shape | Bits min/mean/max | Pruned |")
        L("|---|---|---:|---:|")
        for row in bitwidth_stats["per_layer"]:
            if row["kind"] != "kq":
                continue
            shape_str = "×".join(str(s) for s in row["shape"])
            L(f"| {row['layer']} | {shape_str} | "
              f"{row['min']:.1f} / {row['mean']:.2f} / {row['max']:.1f} | "
              f"{row['n_zero']}/{row['n_total']} |")
    else:
        L("(Bit-width extraction unavailable — see `comparison.json` for "
          "the raw model. Likely cause: HGQ2 not installed or the model "
          "is not an HGQ2 model.)")
    L("")
    L("The total-kernel-bits number is a pre-synthesis proxy for the weight "
      "memory footprint on the FPGA — the actual LUT/DSP/BRAM is reported by "
      "Vivado HLS synthesis (not yet wired into this driver; ``--synth`` flag "
      "in `scripts/synthesize.py` is reserved for that step).")
    L("")

    # ── Artifacts ───────────────────────────────────────────────────────────
    L("## Artifacts")
    L("")
    L("```")
    L(f"{output_dir.name}/")
    L("├── fp32/")
    L("│   └── bounded_mae_only_xy0_seed{seed}/")
    L("│       ├── best_model.keras            ← float32 model")
    L("│       ├── result.json")
    L("│       └── hls/                        ← hls4ml C++ + bit-accurate sim")
    L("├── hgq2/")
    L("│   ├── best_model.keras                ← QAT+pruned model (the deliverable)")
    L("│   ├── result.json")
    L("│   └── hls/                            ← hls4ml C++ + bit-accurate sim")
    L("├── comparison.json                     ← machine-readable physics cards + bit-widths")
    L("├── run.log                             ← full stdout from the experiment")
    L("└── report.md                           ← this file")
    L("```")
    L("")

    # ── Reproducibility ─────────────────────────────────────────────────────
    L("## Reproducing this experiment")
    L("")
    L("```bash")
    L("source /home/users/dprimosc/micromamba/etc/profile.d/micromamba.sh")
    L("micromamba activate l1deepmet")
    L("")
    L("python scripts/experiments/qat_validation.py \\")
    L(f"    --output-dir {output_dir} \\")
    L(f"    --epochs {config['epochs']} \\")
    L(f"    --seed {config['seed']}")
    L("```")
    L("")
    L("Or train just the QAT model (faster, if you only want the deliverable "
      "and not the FP32 comparison):")
    L("")
    L("```bash")
    L("python scripts/train_hgq2.py --output-dir hgq2_only --epochs 30 --seed 42")
    L("python scripts/synthesize.py --model hgq2_only/best_model.keras \\")
    L("    --output-dir hgq2_only/hls --no-export-pass")
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

    # ── Bit-width / pruning analysis on the trained HGQ2 model ─────────────
    log.info("Extracting learned bit-widths + sparsity from HGQ2 model…")
    bitwidth_stats = extract_bitwidth_stats(hgq2_dir / "best_model.keras")
    log.info("HGQ2 kernel sparsity: %s%% (avg bit-width %.2f)",
             bitwidth_stats.get("sparsity_pct", "?"),
             bitwidth_stats.get("avg_kernel_bitwidth", 0))

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
            "bitwidth_stats": bitwidth_stats,
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
        bitwidth_stats=bitwidth_stats,
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
