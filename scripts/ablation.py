#!/usr/bin/env python3
"""
Unified ablation runner.

Trains a list of (arch_config, loss_config) combinations across multiple seeds,
saves models + per-run JSON + a single ablation_results.csv, and prints a
seed-averaged summary at the end. Replaces the three near-duplicate scripts
binned_weight_ablation.py, loss_ablation.py, residual_ablation.py.

Recipes are small Python dicts in `scripts/ablation_recipes.py`. To add a new
study, append a recipe and run:

    python scripts/ablation.py --recipe my_new_study --epochs 30

Optional `--deterministic` enables TF op-level determinism for bitwise-
reproducible per-seed numbers (slower).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import sys, time, json, argparse, csv, importlib
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

import numpy as np
import h5py
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)

from arch_search import ArchConfig, build_model, NORMFAC
from l1deepmet.data.loader import H5DataLoader, split_preprocessed_features
from l1deepmet.losses.corrected import CorrectedCompositeLoss
from l1deepmet.metrics.binned import BinnedDeviation
from l1deepmet.metrics.physics import (
    compute_puppi_baseline, full_physics_card, resolqt,
)


# ─── ArchConfig + Loss kwargs ────────────────────────────────────────────────
# A "recipe" is a dict like:
#   {
#     "default_arch": dict of ArchConfig kwargs (applied to every cell),
#     "default_loss": dict of CorrectedCompositeLoss kwargs,
#     "cells": list of dicts. Each cell has:
#         "name": str,
#         "arch": dict of overrides on default_arch,
#         "loss": dict of overrides on default_loss,
#   }


def merge(base: dict, override: dict | None) -> dict:
    out = dict(base)
    if override:
        out.update(override)
    return out


def make_arch_cfg(name: str, kwargs: dict) -> ArchConfig:
    return ArchConfig(name=name, **kwargs)


def make_loss(kwargs: dict, normfac: float) -> CorrectedCompositeLoss:
    return CorrectedCompositeLoss(normfac=normfac, **kwargs)


def evaluate_model(model, X, Y) -> dict:
    """Full physics-card eval: resolution + per-bin response + phi + AUC,
    plus comparisons to the PUPPI baseline. All scalars, JSON-safe.

    Returned keys include (from src/l1deepmet/metrics/physics.py):
      Resolution:  met_pt_resolution, met_x_resolution, met_y_resolution
      Response:    mean_response, response_50_100, response_100_200, ...
      Angular:     phi_resolution
      Trigger:     auc, n_signal, n_background
      vs PUPPI:    puppi_* (same set), delta_* (model − PUPPI for resolutions
                   and angular; model − puppi for auc), mean_pred_pt_over_puppi_pt
    """
    inputs, pxpy, pdg, charge = split_preprocessed_features(X)
    mi = {"continuous_inputs": inputs, "momentum_inputs": pxpy,
          "pdgid_inputs": pdg, "charge_inputs": charge}
    pred_xy = model.predict(mi, batch_size=512, verbose=0) * NORMFAC
    puppi_xy = compute_puppi_baseline(X)
    return full_physics_card(gen_xy=Y, reco_xy=pred_xy, puppi_xy=puppi_xy)


def train_one(cell: dict, recipe: dict, seed: int,
              train_ds, val_ds, X_test, Y_test,
              epochs: int, output_dir: str) -> dict:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    name = cell["name"]
    arch_kwargs = merge(recipe.get("default_arch", {}), cell.get("arch", {}))
    loss_kwargs = merge(recipe.get("default_loss", {}), cell.get("loss", {}))

    # ArchConfig's xy_balance_weight is duplicated in the loss recipe;
    # the loss owns the actual term, but build_model uses the arch flag for
    # initializer bookkeeping. Sync them so user sets xy in one place (loss).
    if "xy_balance_weight" in loss_kwargs and "xy_balance_weight" not in arch_kwargs:
        arch_kwargs["xy_balance_weight"] = loss_kwargs["xy_balance_weight"]

    run_dir = os.path.join(output_dir, f"{name}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    arch_cfg = make_arch_cfg(name, arch_kwargs)
    model = build_model(arch_cfg)
    init_eval = evaluate_model(model, X_test, Y_test)
    print(f"  [init] X={init_eval['met_x_resolution']:.2f} "
          f"Y={init_eval['met_y_resolution']:.2f}  "
          f"pT={init_eval['met_pt_resolution']:.2f}  "
          f"AUC={init_eval['auc']:.4f}",
          flush=True)

    loss = make_loss(loss_kwargs, normfac=NORMFAC)
    pt_bins_normalized = np.array([50.0, 100.0, 200.0, 300.0, 400.0, np.inf]) / NORMFAC
    metrics = [
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanSquaredError(name="mse"),
        BinnedDeviation(pt_bins=pt_bins_normalized),
    ]
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         verbose=0, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=5, verbose=0, min_lr=1e-6),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv")),
    ]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=epochs, callbacks=callbacks, verbose=0)
    elapsed = time.time() - t0

    final_eval = evaluate_model(model, X_test, Y_test)
    val_loss = float(min(history.history["val_loss"]))
    n_ep = len(history.history["val_loss"])
    print(f"  [final, {n_ep}ep, {elapsed:.0f}s] "
          f"X={final_eval['met_x_resolution']:.2f} "
          f"Y={final_eval['met_y_resolution']:.2f}  "
          f"pT={final_eval['met_pt_resolution']:.2f}  "
          f"AUC={final_eval['auc']:.4f}  "
          f"Δ_pT={final_eval.get('delta_met_pt_resolution', float('nan')):+.2f} "
          f"Δ_AUC={final_eval.get('delta_auc', float('nan')):+.4f}",
          flush=True)

    model.save(os.path.join(run_dir, "best_model.keras"))

    result = {
        "config": name, "seed": seed,
        "arch_kwargs": json.dumps(arch_kwargs, default=str),
        "loss_kwargs": json.dumps(loss_kwargs, default=str),
        "n_params": model.count_params(),
        "epochs_trained": n_ep, "train_time_s": elapsed,
        "best_val_loss": val_loss,
        "init_met_x_resolution": init_eval["met_x_resolution"],
        "init_met_y_resolution": init_eval["met_y_resolution"],
        "init_met_pt_resolution": init_eval["met_pt_resolution"],
        "init_auc": init_eval["auc"],
        **{f"final_{k}": v for k, v in final_eval.items()},
    }
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    del model
    tf.keras.backend.clear_session()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True,
                        help="Recipe name from scripts/ablation_recipes.py")
    parser.add_argument("--data-dir", default="/home/users/dprimosc/L1DeepMET/preprocessed/25Jul8_140X_v0")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: outputs/<recipe>_<date>)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--cells", nargs="+", type=str, default=None,
                        help="Subset of cell names to run (default: all)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable TF op-level determinism for bitwise-reproducible "
                             "per-seed numbers (slower).")
    args = parser.parse_args()

    if args.deterministic:
        # Note: requires TF >= 2.8 and certain ops to have deterministic
        # implementations. Slows training. Combine with --epochs fixed (no early
        # stop) for fully deterministic numbers.
        tf.config.experimental.enable_op_determinism()
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        print("Deterministic mode: enabled", flush=True)

    recipes = importlib.import_module("ablation_recipes").RECIPES
    if args.recipe not in recipes:
        print(f"Unknown recipe '{args.recipe}'. Available: {list(recipes)}")
        sys.exit(1)
    recipe = recipes[args.recipe]

    output_dir = args.output_dir or f"outputs/{args.recipe}_apr2026"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Recipe: {args.recipe}   Output: {output_dir}", flush=True)

    loader = H5DataLoader(args.data_dir)
    train_ds = loader.create_tf_dataset("train", batch_size=args.batch_size,
                                        shuffle=True, normfac=NORMFAC)
    val_ds = loader.create_tf_dataset("val", batch_size=args.batch_size,
                                      shuffle=False, normfac=NORMFAC)

    with h5py.File(os.path.join(args.data_dir, "test.h5"), "r") as f:
        X_test = f["features"][:]
        Y_test = f["targets"][:]
    print(f"Test: {X_test.shape}", flush=True)

    cells_to_run = recipe["cells"]
    if args.cells:
        cells_to_run = [c for c in cells_to_run if c["name"] in args.cells]

    results = []
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    fieldnames = None

    for cell in cells_to_run:
        for seed in args.seeds:
            print(f"\n=== {cell['name']} | seed {seed} ===", flush=True)
            try:
                r = train_one(cell, recipe, seed, train_ds, val_ds, X_test, Y_test,
                              args.epochs, output_dir)
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                continue

            if fieldnames is None:
                fieldnames = list(r.keys())
                with open(csv_path, "w") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writeheader()
            with open(csv_path, "a") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writerow({k: r.get(k, "") for k in fieldnames})

    print("\n" + "=" * 110, flush=True)
    print(f"SUMMARY: {args.recipe}", flush=True)
    print("=" * 110, flush=True)
    print(f"{'config':<22} {'X IQR/2':>12} {'Y IQR/2':>12} "
          f"{'pT IQR/2':>12} {'φ res':>10} {'AUC':>10} {'Δ_pT':>10} "
          f"{'Δ_AUC':>10} {'#par':>7}", flush=True)
    print("-" * 110, flush=True)
    if results:
        # PUPPI reference (same across runs)
        first = results[0]
        print(f"{'PUPPI MET (ref)':<22} "
              f"{first.get('final_puppi_met_x_resolution', 38.30):>12.2f} "
              f"{first.get('final_puppi_met_y_resolution', 38.38):>12.2f} "
              f"{first.get('final_puppi_met_pt_resolution', float('nan')):>12.2f} "
              f"{first.get('final_puppi_phi_resolution', float('nan')):>10.4f} "
              f"{first.get('final_puppi_auc', float('nan')):>10.4f} "
              f"{0.0:>+10.2f} {0.0:>+10.4f} {'0':>7}",
              flush=True)
    for cell in cells_to_run:
        rs = [r for r in results if r["config"] == cell["name"]]
        if not rs: continue
        def m(k):
            vals = [r.get(f"final_{k}", float("nan")) for r in rs]
            return np.mean(vals), np.std(vals)
        xm, xs = m("met_x_resolution")
        ym, ys = m("met_y_resolution")
        pm, ps = m("met_pt_resolution")
        phm, _ = m("phi_resolution")
        am, _ = m("auc")
        dp, _ = m("delta_met_pt_resolution")
        da, _ = m("delta_auc")
        nparam = rs[0]["n_params"]
        print(f"{cell['name']:<22} "
              f"{xm:>6.2f}±{xs:.2f} {ym:>6.2f}±{ys:.2f} "
              f"{pm:>6.2f}±{ps:.2f} "
              f"{phm:>10.4f} {am:>10.4f} {dp:>+10.2f} {da:>+10.4f} "
              f"{nparam:>7}", flush=True)

    print(f"\nSaved {len(results)} results to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
