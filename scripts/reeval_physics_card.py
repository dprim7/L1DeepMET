#!/usr/bin/env python3
"""
Re-evaluate saved models with the full physics card.

Walks one or more ablation output directories, finds every saved
``best_model.keras``, runs it on the test split, computes the full physics
card via ``l1deepmet.metrics.physics.full_physics_card``, and writes:

  - <run_dir>/result_v2.json   : the full physics card per run
  - <out>/full_physics_card.csv : aggregated across all runs

The original ``result.json`` files are left untouched.

Typical use after this commit lands:

    python scripts/reeval_physics_card.py \
      outputs/loss_ablation_apr2026 \
      outputs/loss_form_ablation_apr2026 \
      outputs/residual_ablation_apr2026 \
      outputs/combined_best_apr2026 \
      --output-csv outputs/full_physics_card.csv
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import sys, argparse, json, csv, glob
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

import numpy as np
import h5py
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)

from arch_search import CastToInt, ZeroReduce, ShiftByConstant, SumOverParticles, BoundedWeight, NORMFAC
from l1deepmet.losses.corrected import CorrectedCompositeLoss
from l1deepmet.data.loader import split_preprocessed_features
from l1deepmet.metrics.physics import compute_puppi_baseline, full_physics_card


CUSTOM_OBJECTS = {
    "CastToInt": CastToInt,
    "ZeroReduce": ZeroReduce,
    "ShiftByConstant": ShiftByConstant,
    "SumOverParticles": SumOverParticles,
    "BoundedWeight": BoundedWeight,
    "CorrectedCompositeLoss": CorrectedCompositeLoss,
}


def find_runs(roots):
    """Yield (run_dir, model_path) for every best_model.keras under given dirs."""
    for root in roots:
        for model_path in sorted(glob.glob(os.path.join(root, "*/best_model.keras"))):
            run_dir = os.path.dirname(model_path)
            yield run_dir, model_path


def parse_run_name(run_dir):
    """Extract ablation source + config + seed from a run-dir name like
    .../<ablation_dir>/<config>_seed<N>/."""
    parent = os.path.basename(os.path.dirname(run_dir))
    name = os.path.basename(run_dir)
    seed = None
    if "_seed" in name:
        config, _, seed_str = name.rpartition("_seed")
        try:
            seed = int(seed_str)
        except ValueError:
            config = name
    else:
        config = name
    return parent, config, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ablation_dirs", nargs="+",
                        help="One or more dirs each containing *_seedN/best_model.keras")
    parser.add_argument("--data-dir",
                        default="/home/users/dprimosc/L1DeepMET/preprocessed/25Jul8_140X_v0")
    parser.add_argument("--output-csv", default="outputs/full_physics_card.csv")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs that already have a result_v2.json")
    args = parser.parse_args()

    with h5py.File(os.path.join(args.data_dir, "test.h5"), "r") as f:
        X = f["features"][:]
        Y = f["targets"][:]
    print(f"Test split: {X.shape}", flush=True)

    inputs, pxpy, pdg, charge = split_preprocessed_features(X)
    mi = {"continuous_inputs": inputs, "momentum_inputs": pxpy,
          "pdgid_inputs": pdg, "charge_inputs": charge}
    puppi_xy = compute_puppi_baseline(X)

    all_rows = []
    runs = list(find_runs(args.ablation_dirs))
    print(f"Found {len(runs)} models to evaluate.", flush=True)

    for i, (run_dir, model_path) in enumerate(runs):
        ablation, config, seed = parse_run_name(run_dir)
        out_json = os.path.join(run_dir, "result_v2.json")

        if args.skip_existing and os.path.exists(out_json):
            with open(out_json) as f:
                row = json.load(f)
            all_rows.append(row)
            print(f"[{i+1:2d}/{len(runs)}] (skip) {ablation}/{config}_seed{seed}", flush=True)
            continue

        try:
            model = tf.keras.models.load_model(model_path,
                                               custom_objects=CUSTOM_OBJECTS,
                                               compile=False)
        except Exception as e:
            print(f"[{i+1:2d}/{len(runs)}] LOAD FAILED {ablation}/{config}_seed{seed}: {e}",
                  flush=True)
            continue

        pred_xy = model.predict(mi, batch_size=512, verbose=0) * NORMFAC
        card = full_physics_card(gen_xy=Y, reco_xy=pred_xy, puppi_xy=puppi_xy)

        # Also pull n_params + original (v1) result info if present.
        n_params = int(model.count_params())
        v1_path = os.path.join(run_dir, "result.json")
        epochs_trained = None
        train_time_s = None
        if os.path.exists(v1_path):
            try:
                with open(v1_path) as f:
                    v1 = json.load(f)
                epochs_trained = v1.get("epochs_trained")
                train_time_s = v1.get("train_time_s")
            except Exception:
                pass

        row = {
            "ablation": ablation,
            "config": config,
            "seed": seed,
            "n_params": n_params,
            "epochs_trained": epochs_trained,
            "train_time_s": train_time_s,
            "model_path": os.path.abspath(model_path),
            **card,
        }
        with open(out_json, "w") as f:
            json.dump(row, f, indent=2, default=str)
        all_rows.append(row)

        print(f"[{i+1:2d}/{len(runs)}] {ablation}/{config}_seed{seed}: "
              f"X={card['met_x_resolution']:.2f} pT={card['met_pt_resolution']:.2f} "
              f"AUC={card['auc']:.4f} ΔpT={card.get('delta_met_pt_resolution', float('nan')):+.2f} "
              f"ΔAUC={card.get('delta_auc', float('nan')):+.4f}", flush=True)

        del model
        tf.keras.backend.clear_session()

    # Aggregate CSV
    if all_rows:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        fieldnames = sorted({k for row in all_rows for k in row.keys()})
        with open(args.output_csv, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"\nWrote {len(all_rows)} rows to {args.output_csv}", flush=True)

    # Summary table
    if all_rows:
        # group by (ablation, config)
        by_key = {}
        for r in all_rows:
            by_key.setdefault((r["ablation"], r["config"]), []).append(r)
        print("\n" + "=" * 120, flush=True)
        print("AGGREGATED PHYSICS CARD (mean ± std across seeds)", flush=True)
        print("=" * 120, flush=True)
        print(f"{'ablation':<30} {'config':<25} "
              f"{'X':>10} {'Y':>10} {'pT':>10} {'φ':>10} {'AUC':>10} "
              f"{'Δ_pT':>10} {'Δ_AUC':>10}", flush=True)
        print("-" * 120, flush=True)
        for (abl, cfg), rs in sorted(by_key.items()):
            def m(k):
                vals = [r[k] for r in rs if r.get(k) is not None and not isinstance(r.get(k), str)]
                return (np.mean(vals), np.std(vals)) if vals else (float("nan"), 0.0)
            xm, xs = m("met_x_resolution")
            ym, ys = m("met_y_resolution")
            pm, ps = m("met_pt_resolution")
            phm, _ = m("phi_resolution")
            am, _ = m("auc")
            dp, _ = m("delta_met_pt_resolution")
            da, _ = m("delta_auc")
            print(f"{abl:<30} {cfg:<25} "
                  f"{xm:>5.2f}±{xs:.2f} {ym:>5.2f}±{ys:.2f} {pm:>5.2f}±{ps:.2f} "
                  f"{phm:>10.4f} {am:>10.4f} {dp:>+10.2f} {da:>+10.4f}", flush=True)


if __name__ == "__main__":
    main()
