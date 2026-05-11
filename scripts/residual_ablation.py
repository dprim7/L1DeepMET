#!/usr/bin/env python3
"""
Residual / output-head ablation.

The current `weight_minus_one` setup is mathematically a residual decomposition
(model output = PUPPI_MET + correction; init correction = 0). What's left to
test is the inductive bias on the FORM of the correction. Two orthogonal axes:

  - bounded_weight:   tanh-bounded effective weight ∈ (-2, 0), init at -1.
                      Prevents over-amplification; model can only rescale.
  - with_bias:        per-particle additive (Δpx, Δpy) on top of weighting.
                      Allows direction adjustment, not just magnitude.

Crossed: 4 configs × 3 seeds = 12 runs. All use the corrected loss
(MAE + MSE + xy_balance=10, binned_weight=0). Architecture w64 d3 mode 1.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import sys, time, json, argparse, csv
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

import numpy as np
import h5py
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)

from arch_search import ArchConfig, build_model, NORMFAC
from l1deepmet.data.loader import H5DataLoader, split_preprocessed_features
from l1deepmet.losses.corrected import CorrectedCompositeLoss
from l1deepmet.metrics.binned import BinnedDeviation


def resolqt(y):
    return (np.percentile(y, 84) - np.percentile(y, 16)) / 2.0


# (name, bounded_weight, with_bias)
CONFIGS = [
    ("unbounded_no_bias",   False, False),  # control = current scalar_xybal10 (bw=0)
    ("bounded_no_bias",     True,  False),
    ("unbounded_with_bias", False, True),
    ("bounded_with_bias",   True,  True),
]


def make_arch_cfg(name, bounded_w, with_bias):
    return ArchConfig(
        name=name,
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=0.0,
        with_bias=with_bias,
        weight_minus_one=(not bounded_w),  # use shift if not bounded
        bounded_weight=bounded_w,
        use_sum=True,
        xy_balance_weight=10.0,
    )


def evaluate_model(model, X, Y) -> dict:
    inputs, pxpy, pdg, charge = split_preprocessed_features(X)
    mi = {"continuous_inputs": inputs, "momentum_inputs": pxpy,
          "pdgid_inputs": pdg, "charge_inputs": charge}
    pred = model.predict(mi, batch_size=512, verbose=0) * NORMFAC
    gen_x, gen_y = Y[:, 0], Y[:, 1]
    puppi_x = -X[:, :, 5].sum(axis=1)
    puppi_y = -X[:, :, 6].sum(axis=1)
    pred_pt = np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)
    puppi_pt = np.sqrt(puppi_x ** 2 + puppi_y ** 2) + 1e-3
    return {
        "model_iqr_x": float(resolqt(pred[:, 0] - gen_x)),
        "model_iqr_y": float(resolqt(pred[:, 1] - gen_y)),
        "model_std_x": float(np.std(pred[:, 0] - gen_x)),
        "model_std_y": float(np.std(pred[:, 1] - gen_y)),
        "puppi_iqr_x": float(resolqt(puppi_x - gen_x)),
        "puppi_iqr_y": float(resolqt(puppi_y - gen_y)),
        "delta_vs_puppi_x": float(resolqt(pred[:, 0] - gen_x) - resolqt(puppi_x - gen_x)),
        "delta_vs_puppi_y": float(resolqt(pred[:, 1] - gen_y) - resolqt(puppi_y - gen_y)),
        "mean_pred_pt_over_puppi_pt": float((pred_pt / puppi_pt).mean()),
    }


def train_one(arch_cfg, seed, train_ds, val_ds, X_test, Y_test, epochs, output_dir):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    run_dir = os.path.join(output_dir, f"{arch_cfg.name}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    model = build_model(arch_cfg)
    init_eval = evaluate_model(model, X_test, Y_test)
    print(f"  [init] X={init_eval['model_iqr_x']:.2f} Y={init_eval['model_iqr_y']:.2f}",
          flush=True)

    loss = CorrectedCompositeLoss(
        mae_weight=1.0, mse_weight=1.0,
        huber_weight=0.0, binned_weight=0.0, phi_weight=0.0,
        xy_balance_weight=arch_cfg.xy_balance_weight,
        normfac=NORMFAC,
    )
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
    print(f"  [final, {n_ep}ep, {elapsed:.0f}s] X={final_eval['model_iqr_x']:.2f} "
          f"Y={final_eval['model_iqr_y']:.2f}  Δvs PUPPI = "
          f"{final_eval['delta_vs_puppi_x']:+.2f}/{final_eval['delta_vs_puppi_y']:+.2f}  "
          f"scale={final_eval['mean_pred_pt_over_puppi_pt']:.3f}", flush=True)

    model.save(os.path.join(run_dir, "best_model.keras"))

    result = {
        "config": arch_cfg.name, "seed": seed,
        "bounded_weight": arch_cfg.bounded_weight,
        "with_bias": arch_cfg.with_bias,
        "n_params": model.count_params(),
        "epochs_trained": n_ep, "train_time_s": elapsed,
        "best_val_loss": val_loss,
        "init_iqr_x": init_eval["model_iqr_x"],
        "init_iqr_y": init_eval["model_iqr_y"],
        **{f"final_{k}": v for k, v in final_eval.items()},
    }
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    del model
    tf.keras.backend.clear_session()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/users/dprimosc/L1DeepMET/preprocessed/25Jul8_140X_v0")
    parser.add_argument("--output-dir", default="outputs/residual_ablation_apr2026")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--configs", nargs="+", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}", flush=True)

    loader = H5DataLoader(args.data_dir)
    train_ds = loader.create_tf_dataset("train", batch_size=args.batch_size,
                                        shuffle=True, normfac=NORMFAC)
    val_ds = loader.create_tf_dataset("val", batch_size=args.batch_size,
                                      shuffle=False, normfac=NORMFAC)

    with h5py.File(os.path.join(args.data_dir, "test.h5"), "r") as f:
        X_test = f["features"][:]
        Y_test = f["targets"][:]
    print(f"Test: {X_test.shape}", flush=True)

    configs_to_run = CONFIGS
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c[0] in args.configs]

    results = []
    csv_path = os.path.join(args.output_dir, "ablation_results.csv")
    fieldnames = None

    for name, bounded_w, with_bias in configs_to_run:
        for seed in args.seeds:
            print(f"\n=== {name} | seed {seed} | bounded={bounded_w} bias={with_bias} ===",
                  flush=True)
            try:
                arch_cfg = make_arch_cfg(name, bounded_w, with_bias)
                r = train_one(arch_cfg, seed, train_ds, val_ds, X_test, Y_test,
                              args.epochs, args.output_dir)
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

    print("\n" + "=" * 100, flush=True)
    print("RESIDUAL / OUTPUT-HEAD ABLATION SUMMARY", flush=True)
    print("=" * 100, flush=True)
    print(f"{'config':<22} {'X IQR/2':>14} {'Y IQR/2':>14} "
          f"{'Δ vs PUPPI X':>14} {'Δ Y':>10} {'pred/PUPPI':>12} {'#params':>8}",
          flush=True)
    print("-" * 100, flush=True)
    print(f"{'PUPPI MET (ref)':<22} {'38.30':>14} {'38.38':>14} "
          f"{'+0.00':>14} {'+0.00':>10} {'1.000':>12} {'0':>8}", flush=True)
    for name, _, _ in configs_to_run:
        rs = [r for r in results if r["config"] == name]
        if not rs: continue
        xm = np.mean([r["final_model_iqr_x"] for r in rs])
        xs = np.std([r["final_model_iqr_x"] for r in rs])
        ym = np.mean([r["final_model_iqr_y"] for r in rs])
        ys = np.std([r["final_model_iqr_y"] for r in rs])
        dx = np.mean([r["final_delta_vs_puppi_x"] for r in rs])
        dy = np.mean([r["final_delta_vs_puppi_y"] for r in rs])
        scale = np.mean([r["final_mean_pred_pt_over_puppi_pt"] for r in rs])
        nparam = rs[0]["n_params"]
        print(f"{name:<22} {xm:>6.2f}±{xs:.2f}{'':<3} {ym:>6.2f}±{ys:.2f}{'':<3} "
              f"{dx:>+14.2f} {dy:>+10.2f} {scale:>12.3f} {nparam:>8}", flush=True)

    print(f"\nSaved {len(results)} results to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
