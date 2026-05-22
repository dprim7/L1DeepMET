#!/usr/bin/env python3
"""
SCAFFOLD — Evaluate the full L1 physics card on a trained model.

This is a deliberately-empty orchestration skeleton that pulls the
per-tier metrics from ``src/l1deepmet/metrics/physics.py`` and writes a
single JSON + a directory of plots that satisfies the CLAUDE.md
"Evaluation metrics for architecture / recipe sweeps" standing order.

Most of the metric primitives are still stubs (see physics.py and
``reports/physics_card_design/DESIGN.md``); this file shows where the
calls will go once those are designed.

INTENDED USAGE (once stubs land):
  python scripts/evaluate_physics_card.py \\
      --model outputs/models/<run>/model.keras \\
      --signal-h5 outputs/preprocessed/<tag>/test.h5 \\
      --signal-name VBFHToInvisible_PU200 \\
      --background-h5 /path/to/minbias_pu200.h5 \\
      --target-rate-khz 4.0 \\
      --out reports/<study>/physics_card.json \\
      --plots-dir reports/<study>/plots

OPEN DESIGN QUESTIONS (track in DESIGN.md):
  1. Model loading: Keras .keras (preferred TF2) vs SavedModel vs
     checkpoint? Affects --model arg type.
  2. Eval-time normalisation: do we re-derive medians/scales from the
     training data (need a stats file checkin) or store them in the
     model artifact?
  3. Per-signal-sample handling: currently --signal-h5 is one file; for
     a real card we probably want one card per signal sample (VBF, TT
     semilep, SMS T1tttt) so the working-point efficiency table has
     N rows. Either repeat the script or take a list.
  4. PUPPI baseline: compute on the fly from raw H5 features
     (``compute_puppi_baseline``) vs pre-computed event feature column?
  5. Plot file naming: stick with the convention used in
     ``reports/event_count_justification/plots/`` (one PNG per metric)?
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


# Imports kept here for visibility — these are the call sites the script
# will use. Wire them once the stubs are implemented.
def _eval_signal(args):
    raise NotImplementedError(
        "scaffold: load model + signal H5, predict, run "
        "full_physics_card + asymmetric_tails + per_pu_card + per_eta_card"
    )


def _eval_background_rate(args):
    raise NotImplementedError(
        "scaffold: load model + MinBias H5, predict, run "
        "compute_rate_vs_threshold"
    )


def _eval_working_point(rate_curve, signal_turn_on, target_rate_khz):
    raise NotImplementedError(
        "scaffold: combine rate + turn-on → compute_working_point"
    )


def _eval_puppi_ablation(args, eval_fn):
    raise NotImplementedError(
        "scaffold: compute_puppi_ablation(eval_fn, features, gen_xy)"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, type=Path,
                   help="Trained Keras model (format TBD — see DESIGN.md Q1)")
    p.add_argument("--signal-h5", required=True, type=Path,
                   help="Signal H5 (one sample at a time; see DESIGN.md Q3)")
    p.add_argument("--signal-name", required=True, type=str,
                   help="Label for this signal sample in the output JSON")
    p.add_argument("--background-h5", required=True, type=Path,
                   help="MinBias_PU200 H5 for the rate calculation")
    p.add_argument("--target-rate-khz", type=float, default=4.0,
                   help="L1 MET budget; defines the working point. "
                        "DESIGN.md Q: what's the actual Phase-2 number?")
    p.add_argument("--out", required=True, type=Path,
                   help="Physics card JSON output path")
    p.add_argument("--plots-dir", required=True, type=Path,
                   help="Where to write the plot PNGs")
    p.add_argument("--include-tier2", action="store_true",
                   help="Compute Tier-2 metrics (per-PU, per-eta, asymmetric "
                        "tails, PUPPI ablation). Default off until stubs land.")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    # Tier 1: resolution + response + AUC + ROC + turn-on + rate + working point
    # signal_card = _eval_signal(args)         # → resolution, response, AUC, turn-on
    # rate_curve  = _eval_background_rate(args)  # → rate vs threshold
    # wp          = _eval_working_point(rate_curve, signal_card["turn_on"],
    #                                    args.target_rate_khz)

    # Tier 2 (optional, behind --include-tier2):
    # per_pu  = compute_per_pu_card(...)         # stratify by nL1Vtx
    # per_eta = compute_per_eta_card(...)        # barrel vs endcap
    # tails   = compute_asymmetric_tails(...)    # fakes vs misses
    # puppi_ab = _eval_puppi_ablation(args, eval_fn)

    # Tier 3 (separate script — synth/HLS4ML):
    # NOT done here; see src/l1deepmet/synthesis/ when designed.

    # write_card_json(args.out, signal_card, rate_curve, wp, per_pu, per_eta, tails)
    # generate_plots(args.plots_dir, ...)

    print("evaluate_physics_card.py: scaffold only — stubs in physics.py "
          "need to be designed first. See reports/physics_card_design/DESIGN.md")
    return 1


if __name__ == "__main__":
    sys.exit(main())
