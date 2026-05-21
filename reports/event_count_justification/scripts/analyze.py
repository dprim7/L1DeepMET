#!/usr/bin/env python3
"""
Aggregate the 30-cell sweep into:
  - results/aggregated.csv: one row per (N, variant, seed)
  - results/fits.json: learning-curve fits + recommended N*
  - plots/learning_curve.png: val MAE vs N, both variants, with seed-band
  - plots/per_bin_mae.png: per-pT-bin MAE at multiple N
  - plots/extended_vs_baseline.png: variant comparison at N=8000
  - plots/extrapolation.png: fit + extrapolation to target SEM

Decision rule (from PLAN.md):
  if val_MAE(8000_extended) > val_MAE(4000_extended) - 1·σ_seed
    → data is NOT the bottleneck at 10k, recommend N≈10k
  else
    → fit MAE ≈ A + B·N^{-α}, recommend N* where doubling-improvement ≤ σ_seed
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt    # type: ignore
import numpy as np                 # type: ignore
import pandas as pd                # type: ignore
from scipy.optimize import curve_fit  # type: ignore

VARIANT_COLORS = {"extended": "#1f77b4", "baseline": "#ff7f0e"}
PT_BIN_LABELS = ["0_50", "50_100", "100_200", "200_300", "300_400", "400_inf"]


def load_raw(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for jf in sorted(raw_dir.glob("*.json")):
        d = json.loads(jf.read_text())
        cfg = d["config"]; tm = d["test_metrics"]; tr = d["training"]
        row = {
            "cell": jf.stem,
            "n_train": cfg["n_train_effective"],
            "variant": cfg["variant"],
            "seed": cfg["seed"],
            "epochs": cfg["epochs"],
            "wall_s": tr["wall_s"],
            "train_loss_final": tr["loss_final"],
            "val_loss_final": tr["val_loss_final"],
            "test_mae_xy": tm["overall"]["mae_xy"],
            "test_mse_xy": tm["overall"]["mse_xy"],
            "test_bias_x": tm["overall"]["bias_x"],
            "test_bias_y": tm["overall"]["bias_y"],
            "test_iqr2_x": tm["overall"]["iqr2_x"],
            "test_iqr2_y": tm["overall"]["iqr2_y"],
            "test_met_pt_iqr2": tm["overall"]["met_pt_iqr2"],
        }
        for label in PT_BIN_LABELS:
            row[f"mae_{label}"] = tm["per_bin"][label].get("mae_xy")
            row[f"n_{label}"] = tm["per_bin"][label].get("n")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per (n_train, variant) over seeds."""
    metrics = ["test_mae_xy", "test_mse_xy", "test_iqr2_x", "test_iqr2_y",
               "test_met_pt_iqr2"] + [f"mae_{b}" for b in PT_BIN_LABELS]
    agg = df.groupby(["n_train", "variant"])[metrics].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index().sort_values(["variant", "n_train"])


def fit_learning_curve(n_arr, mae_arr):
    """Fit MAE(N) = A + B * N^{-alpha}. Returns (A, B, alpha, residuals)."""
    def model(N, A, B, alpha):
        return A + B * N ** (-alpha)
    try:
        popt, _ = curve_fit(model, n_arr, mae_arr,
                            p0=[mae_arr.min() * 0.9, mae_arr[0], 0.3],
                            bounds=([0, 0, 0.01], [100, 1000, 2.0]),
                            maxfev=5000)
        return popt
    except Exception as e:
        return None


def decide(agg: pd.DataFrame, sigma_seed: float, target_marginal_gev: float):
    """Apply the pre-registered decision rule. Compare the two LARGEST N values
    actually present (handles `--n-events 8000` being capped to the actual
    train pool size, e.g. 7328)."""
    ext = agg[agg["variant"] == "extended"].set_index("n_train").sort_index()
    Ns = sorted(ext.index.values)
    if len(Ns) < 2:
        raise ValueError(f"need ≥ 2 N values in extended variant, got {Ns}")
    n_top, n_prev = Ns[-1], Ns[-2]
    mae_top  = ext.loc[n_top,  "test_mae_xy_mean"]
    mae_prev = ext.loc[n_prev, "test_mae_xy_mean"]
    marginal = mae_prev - mae_top   # >0 means we gained going n_prev → n_top
    rule = {
        "sigma_seed_used": float(sigma_seed),
        "n_top": int(n_top),
        "n_prev": int(n_prev),
        "mae_at_n_top_extended":  float(mae_top),
        "mae_at_n_prev_extended": float(mae_prev),
        "marginal_n_prev_to_top": float(marginal),
        "is_data_limited": bool(marginal > sigma_seed),
    }
    # Fit + extrapolate
    n_arr = ext.index.values.astype(float)
    mae_arr = ext["test_mae_xy_mean"].values
    fit = fit_learning_curve(n_arr, mae_arr)
    if fit is not None:
        A, B, alpha = map(float, fit)
        rule["fit"] = {"A": A, "B": B, "alpha": alpha}
        # Find N* such that MAE(N) - MAE(2N) <= target_marginal
        Ns_search = np.logspace(np.log10(400), np.log10(1_000_000), 1000)
        marginals = (A + B * Ns_search ** (-alpha)) - (A + B * (2 * Ns_search) ** (-alpha))
        ok_idx = np.where(marginals <= target_marginal_gev)[0]
        if len(ok_idx):
            n_star = float(Ns_search[ok_idx[0]])
            rule["n_star_for_target"] = n_star
            rule["mae_at_n_star"] = float(A + B * n_star ** (-alpha))
            rule["mae_asymptote"] = float(A)
        else:
            rule["n_star_for_target"] = None
            rule["note"] = f"no N up to 1M gets marginal ≤ {target_marginal_gev}"
    return rule


def plot_learning_curve(agg: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
    for variant in ["baseline", "extended"]:
        sub = agg[agg["variant"] == variant].sort_values("n_train")
        ns = sub["n_train"].values
        mean = sub["test_mae_xy_mean"].values * 100   # back to GeV (already in GeV here)
        std  = sub["test_mae_xy_std"].values  * 100
        # Actually, mae_xy is in normalized units? No — train_one.py de-normalizes pred before metrics.
        # So mae_xy is in GeV. Don't multiply.
        mean = sub["test_mae_xy_mean"].values
        std  = sub["test_mae_xy_std"].values
        c = VARIANT_COLORS[variant]
        ax.errorbar(ns, mean, yerr=std, marker="o", color=c, capsize=4, label=variant)
        ax.fill_between(ns, mean - std, mean + std, color=c, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("training event count N")
    ax.set_ylabel("test MAE on (px, py) — GeV per component")
    ax.set_title("Event-count learning curve (mean ± std over 3 seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_bin(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=110, sharey=False)
    for ax, variant in zip(axes, ["baseline", "extended"]):
        sub = df[df["variant"] == variant]
        ns = sorted(sub["n_train"].unique())
        for n in ns:
            row = sub[sub["n_train"] == n]
            means = [row[f"mae_{b}"].mean() for b in PT_BIN_LABELS]
            stds  = [row[f"mae_{b}"].std()  for b in PT_BIN_LABELS]
            ax.errorbar(range(len(PT_BIN_LABELS)), means, yerr=stds,
                        marker="o", label=f"N={n}", capsize=3)
        ax.set_xticks(range(len(PT_BIN_LABELS)))
        ax.set_xticklabels(PT_BIN_LABELS, rotation=30, ha="right")
        ax.set_xlabel("gen-MET bin (GeV)")
        ax.set_ylabel("MAE on (px, py) [GeV]")
        ax.set_title(f"variant: {variant}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_extended_vs_baseline(df: pd.DataFrame, out_path: Path, n_compare: int = None):
    """Compare per-bin MAE at the largest N actually present in the data."""
    if n_compare is None:
        n_compare = int(df["n_train"].max())
    sub = df[df["n_train"] == n_compare]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
    for variant in ["baseline", "extended"]:
        rows = sub[sub["variant"] == variant]
        means = [rows[f"mae_{b}"].mean() for b in PT_BIN_LABELS]
        stds  = [rows[f"mae_{b}"].std()  for b in PT_BIN_LABELS]
        c = VARIANT_COLORS[variant]
        ax.errorbar(range(len(PT_BIN_LABELS)), means, yerr=stds,
                    marker="o", color=c, label=variant, capsize=4)
    ax.set_xticks(range(len(PT_BIN_LABELS)))
    ax.set_xticklabels(PT_BIN_LABELS, rotation=30, ha="right")
    ax.set_xlabel("gen-MET bin (GeV)")
    ax.set_ylabel("MAE on (px, py) [GeV]")
    ax.set_title(f"Extended vs baseline at N = {n_compare}  (mean ± std, {len(sub)//2} seeds per variant)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_extrapolation(agg: pd.DataFrame, rule: dict, out_path: Path):
    ext = agg[agg["variant"] == "extended"].set_index("n_train").sort_index()
    n_arr = ext.index.values.astype(float)
    mae_arr = ext["test_mae_xy_mean"].values
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
    ax.errorbar(n_arr, mae_arr, yerr=ext["test_mae_xy_std"].values,
                marker="o", color="#1f77b4", capsize=4, label="measured (extended)")
    if "fit" in rule:
        A, B, alpha = rule["fit"]["A"], rule["fit"]["B"], rule["fit"]["alpha"]
        Ns = np.logspace(np.log10(n_arr.min() * 0.5), np.log10(1e6), 200)
        ax.plot(Ns, A + B * Ns ** (-alpha),
                "k--", alpha=0.5, label=f"fit: {A:.2f} + {B:.2f}·N^(-{alpha:.2f})")
        ax.axhline(A, color="grey", ls=":", alpha=0.6, label=f"asymptote A={A:.2f}")
        if rule.get("n_star_for_target"):
            ax.axvline(rule["n_star_for_target"], color="red", ls=":",
                       label=f"N* = {rule['n_star_for_target']:.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("training event count N")
    ax.set_ylabel("test MAE on (px, py) — GeV/component")
    ax.set_title("Learning curve fit + extrapolation (extended variant)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--plots-dir", required=True, type=Path)
    p.add_argument("--target-marginal-gev", type=float, default=0.5,
                   help="Pre-registered target: marginal improvement from doubling N "
                        "must be ≤ this to call the curve plateaued.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(args.raw_dir)
    if df.empty:
        raise SystemExit(f"no JSON results found under {args.raw_dir}")
    df.to_csv(args.out_dir / "aggregated_raw.csv", index=False)

    agg = aggregate(df)
    agg.to_csv(args.out_dir / "aggregated.csv", index=False)

    # σ_seed = median of (per-cell std over 3 seeds) for the EXTENDED variant
    sigma_seed = float(agg[agg["variant"] == "extended"]["test_mae_xy_std"].median())

    rule = decide(agg, sigma_seed=sigma_seed, target_marginal_gev=args.target_marginal_gev)
    (args.out_dir / "fits.json").write_text(json.dumps(rule, indent=2))

    # Plots
    plot_learning_curve(agg, args.plots_dir / "learning_curve.png")
    plot_per_bin(df, args.plots_dir / "per_bin_mae.png")
    plot_extended_vs_baseline(df, args.plots_dir / "extended_vs_baseline.png")
    plot_extrapolation(agg, rule, args.plots_dir / "extrapolation.png")

    # Print summary to stdout
    print("=== aggregated ===")
    print(agg.to_string(index=False))
    print()
    print("=== decision ===")
    print(json.dumps(rule, indent=2))


if __name__ == "__main__":
    main()
