#!/usr/bin/env python3
"""
BinnedDeviation ablation.

Hypothesis: BinnedDeviation (default weight=200) is destructive on this task.
The model with weight_minus_one starts at the raw-sum baseline (38.30 IQR/2)
but training degrades it to 42.74 IQR/2.

Test by training the SAME architecture as scalar_xybal10_emb_w64_d3
with three values of binned_weight: 0, 50, 200, three seeds each.

Pass criterion: at least one of (bw=0, bw=50) beats raw sum (38.30 IQR/2).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import sys, time, json, argparse, csv
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

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


def make_cfg(binned_weight: float, name: str) -> ArchConfig:
    return ArchConfig(
        name=name,
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=binned_weight,
        with_bias=False, weight_minus_one=True, use_sum=True,
        xy_balance_weight=10.0,
    )


def evaluate_model(model, X, Y) -> dict:
    inputs, pxpy, pdg, charge = split_preprocessed_features(X)
    mi = {
        "continuous_inputs": inputs, "momentum_inputs": pxpy,
        "pdgid_inputs": pdg, "charge_inputs": charge,
    }
    pred = model.predict(mi, batch_size=512, verbose=0) * NORMFAC
    gen_x, gen_y = Y[:, 0], Y[:, 1]
    raw_x = -X[:, :, 5].sum(axis=1)
    raw_y = -X[:, :, 6].sum(axis=1)

    return {
        "model_iqr_x": float(resolqt(pred[:, 0] - gen_x)),
        "model_iqr_y": float(resolqt(pred[:, 1] - gen_y)),
        "model_std_x": float(np.std(pred[:, 0] - gen_x)),
        "model_std_y": float(np.std(pred[:, 1] - gen_y)),
        "raw_iqr_x": float(resolqt(raw_x - gen_x)),
        "raw_iqr_y": float(resolqt(raw_y - gen_y)),
        "delta_vs_raw_x": float(resolqt(pred[:, 0] - gen_x) - resolqt(raw_x - gen_x)),
        "delta_vs_raw_y": float(resolqt(pred[:, 1] - gen_y) - resolqt(raw_y - gen_y)),
        "mean_pred_pt_over_raw_pt": float(
            (np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2) /
             (np.sqrt(raw_x ** 2 + raw_y ** 2) + 1e-3)).mean()
        ),
    }


def train_one(cfg: ArchConfig, seed: int, train_ds, val_ds, X_test, Y_test,
              epochs: int, output_dir: str) -> dict:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    run_dir = os.path.join(output_dir, f"{cfg.name}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    model = build_model(cfg)
    n_params = model.count_params()

    # Initial eval (untrained)
    init_eval = evaluate_model(model, X_test, Y_test)
    print(f"  [init] X IQR/2 = {init_eval['model_iqr_x']:.2f}, "
          f"Y IQR/2 = {init_eval['model_iqr_y']:.2f}", flush=True)

    loss = CorrectedCompositeLoss(
        binned_weight=cfg.binned_weight,
        phi_weight=cfg.phi_loss_weight,
        xy_balance_weight=cfg.xy_balance_weight,
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
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, verbose=0,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=0, min_lr=1e-6
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv")),
    ]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=epochs, callbacks=callbacks, verbose=0)
    elapsed = time.time() - t0

    final_eval = evaluate_model(model, X_test, Y_test)
    val_loss = float(min(history.history["val_loss"]))
    epochs_trained = len(history.history["val_loss"])

    print(f"  [final, {epochs_trained}ep, {elapsed:.0f}s] "
          f"X IQR/2 = {final_eval['model_iqr_x']:.2f}, "
          f"Y IQR/2 = {final_eval['model_iqr_y']:.2f}, "
          f"Δvs raw = {final_eval['delta_vs_raw_x']:+.2f}/{final_eval['delta_vs_raw_y']:+.2f}",
          flush=True)

    model.save(os.path.join(run_dir, "best_model.keras"))

    result = {
        "config": cfg.name,
        "seed": seed,
        "binned_weight": cfg.binned_weight,
        "n_params": n_params,
        "epochs_trained": epochs_trained,
        "train_time_s": elapsed,
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
    parser.add_argument("--output-dir", default="outputs/loss_ablation_apr2026")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--binned-weights", nargs="+", type=float, default=[0.0, 50.0, 200.0])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}", flush=True)
    print(f"Loading data from {args.data_dir}", flush=True)

    loader = H5DataLoader(args.data_dir)
    train_ds = loader.create_tf_dataset("train", batch_size=args.batch_size,
                                        shuffle=True, normfac=NORMFAC)
    val_ds = loader.create_tf_dataset("val", batch_size=args.batch_size,
                                      shuffle=False, normfac=NORMFAC)

    with h5py.File(os.path.join(args.data_dir, "test.h5"), "r") as f:
        X_test = f["features"][:]
        Y_test = f["targets"][:]
    print(f"Test: {X_test.shape}", flush=True)

    results = []
    csv_path = os.path.join(args.output_dir, "ablation_results.csv")
    fieldnames = None

    for bw in args.binned_weights:
        cfg = make_cfg(bw, name=f"scalar_xybal10_bw{int(bw)}_emb_w64_d3")
        for seed in args.seeds:
            print(f"\n=== {cfg.name} | seed {seed} | binned_weight={bw} ===", flush=True)
            try:
                r = train_one(cfg, seed, train_ds, val_ds, X_test, Y_test,
                              args.epochs, args.output_dir)
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                continue

            # Append to CSV after each run
            if fieldnames is None:
                fieldnames = list(r.keys())
                with open(csv_path, "w") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writeheader()
            with open(csv_path, "a") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writerow({k: r.get(k, "") for k in fieldnames})

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("ABLATION SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"{'binned_weight':>15} {'X IQR/2':>10} {'Y IQR/2':>10} {'Δvs raw X':>12} {'Δvs raw Y':>12} "
          f"{'pred/raw':>10}", flush=True)
    print("-" * 80, flush=True)
    print(f"{'(raw sum)':>15} {38.30:>10.2f} {38.38:>10.2f} {0.0:>12.2f} {0.0:>12.2f} {1.00:>10.2f}", flush=True)
    for bw in args.binned_weights:
        rs = [r for r in results if r["binned_weight"] == bw]
        if not rs:
            continue
        x_mean = np.mean([r["final_model_iqr_x"] for r in rs])
        x_std = np.std([r["final_model_iqr_x"] for r in rs])
        y_mean = np.mean([r["final_model_iqr_y"] for r in rs])
        y_std = np.std([r["final_model_iqr_y"] for r in rs])
        dx = np.mean([r["final_delta_vs_raw_x"] for r in rs])
        dy = np.mean([r["final_delta_vs_raw_y"] for r in rs])
        scale = np.mean([r["final_mean_pred_pt_over_raw_pt"] for r in rs])
        print(f"{bw:>15.1f} {x_mean:>6.2f}±{x_std:.2f} {y_mean:>6.2f}±{y_std:.2f} "
              f"{dx:>+12.2f} {dy:>+12.2f} {scale:>10.3f}", flush=True)

    print(f"\nSaved {len(results)} results to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
