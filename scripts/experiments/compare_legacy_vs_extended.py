#!/usr/bin/env python3
"""One-shot comparison: legacy-equivalent vs extended feature subsets.

Loads two `.keras` models trained on the SAME H5 with different
`--continuous-slots` and compares them on the test split. The point is
to answer one yes/no question:

    Does the extended 22-slot continuous block do anything detectable
    on top of the 4-slot legacy-equivalent baseline at the same
    architecture and training protocol?

Output:
  - stdout: a side-by-side table with MAE, MSE, X/Y/pT IQR/2, response
  - JSON: the same data, dumped to --out for the report

This is a deliberately-small script — NOT a full physics card and NOT a
LOGO ablation. It's the cheap go/no-go that motivates (or kills) any
deeper feature analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

# Keep TF threading minimal — UAF is shared.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

from l1deepmet.data.loader import H5DataLoader  # noqa: E402
from l1deepmet.metrics.physics import compute_resolution_metrics  # noqa: E402


def _parse_slots(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def _predict(model_path: Path, h5_dir: Path, slots: list[int], normfac: float, batch_size: int):
    """Load model, run prediction on test set, return (gen_xy, reco_xy) in GeV."""
    loader = H5DataLoader(str(h5_dir))
    ds = loader.create_tf_dataset(
        "test", batch_size=batch_size, shuffle=False, normfac=normfac,
        continuous_slots=slots,
    )
    # Read targets directly so we don't have to iterate the dataset twice.
    inputs, pxpy, c0, c1, gen = loader.load_split_data("test", continuous_slots=slots)
    # The model was trained with normfac-divided targets, so the prediction
    # comes out in the same normalized scale. Multiply back to GeV.
    model = tf.keras.models.load_model(str(model_path), compile=False)
    pred = model.predict(ds, verbose=0) * normfac
    return gen.astype(np.float32), pred.astype(np.float32)


def _summary(gen_xy: np.ndarray, reco_xy: np.ndarray) -> dict:
    res = compute_resolution_metrics(gen_xy, reco_xy)
    # Top-line MAE/MSE on (px, py) jointly
    diff = reco_xy - gen_xy
    mae_xy = float(np.mean(np.abs(diff)))
    mse_xy = float(np.mean(diff ** 2))
    out = {
        "mae_xy": mae_xy,
        "mse_xy": mse_xy,
        **{k: (float(v) if np.isscalar(v) else v) for k, v in res.items()},
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5-dir", required=True, help="Directory with test.h5 (shared by both models).")
    p.add_argument("--legacy-model", required=True, help="Path to model.keras for the legacy-equivalent run.")
    p.add_argument("--legacy-slots", default="0,1,2,3", help="Comma-sep slot list used to train --legacy-model.")
    p.add_argument("--extended-model", required=True, help="Path to model.keras for the extended run.")
    p.add_argument("--extended-slots",
                   default="0,1,2,3,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25",
                   help="Comma-sep slot list used to train --extended-model.")
    p.add_argument("--normfac", type=float, default=100.0)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--out", default=None, help="Optional path for comparison JSON.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    legacy_slots = _parse_slots(args.legacy_slots)
    extended_slots = _parse_slots(args.extended_slots)

    logging.info(f"legacy   model: {args.legacy_model}  slots={legacy_slots}")
    logging.info(f"extended model: {args.extended_model}  slots={extended_slots}")

    gen_L, pred_L = _predict(Path(args.legacy_model), Path(args.h5_dir), legacy_slots, args.normfac, args.batch_size)
    gen_E, pred_E = _predict(Path(args.extended_model), Path(args.h5_dir), extended_slots, args.normfac, args.batch_size)

    # Sanity: both should use the same test events
    np.testing.assert_array_equal(gen_L, gen_E)
    gen = gen_L

    summ_L = _summary(gen, pred_L)
    summ_E = _summary(gen, pred_E)

    # Print table
    print()
    print("┌─" + "─" * 28 + "┬─" + "─" * 14 + "┬─" + "─" * 14 + "┬─" + "─" * 12 + "┐")
    print(f"│ {'metric':<28} │ {'legacy (4)':>14} │ {'extended (22)':>14} │ {'Δ':>12} │")
    print("├─" + "─" * 28 + "┼─" + "─" * 14 + "┼─" + "─" * 14 + "┼─" + "─" * 12 + "┤")
    for key in sorted(summ_L.keys()):
        v_l, v_e = summ_L[key], summ_E[key]
        if isinstance(v_l, (int, float)) and isinstance(v_e, (int, float)):
            d = v_e - v_l
            print(f"│ {key:<28} │ {v_l:>14.4f} │ {v_e:>14.4f} │ {d:>+12.4f} │")
    print("└─" + "─" * 28 + "┴─" + "─" * 14 + "┴─" + "─" * 14 + "┴─" + "─" * 12 + "┘")
    print()

    # Top-line verdict on pT resolution (response-corrected IQR/2)
    pt_iqr_l = summ_L.get("met_pt_resolution")
    pt_iqr_e = summ_E.get("met_pt_resolution")
    if pt_iqr_l is not None and pt_iqr_e is not None:
        diff = pt_iqr_e - pt_iqr_l
        improved = diff < -0.5   # arbitrary 0.5 GeV bar for "real" improvement
        regressed = diff > 0.5
        verdict = "EXTENDED WINS" if improved else "EXTENDED LOSES" if regressed else "TIE (Δ within ±0.5 GeV)"
        print(f"Top-line met_pt_resolution: legacy={pt_iqr_l:.3f} GeV, "
              f"extended={pt_iqr_e:.3f} GeV, Δ={diff:+.3f} GeV → {verdict}")
        print()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "legacy": {"slots": legacy_slots, "model": str(args.legacy_model), "metrics": summ_L},
                "extended": {"slots": extended_slots, "model": str(args.extended_model), "metrics": summ_E},
                "n_test_events": int(gen.shape[0]),
            }, f, indent=2)
        logging.info(f"Saved comparison JSON → {args.out}")


if __name__ == "__main__":
    main()
