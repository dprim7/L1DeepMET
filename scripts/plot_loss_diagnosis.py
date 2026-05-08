#!/usr/bin/env python3
"""Generate diagnostic plots for the loss-diagnosis report."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

import json, csv
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)

from arch_search import CastToInt, ZeroReduce, ShiftByConstant, SumOverParticles
from l1deepmet.losses.corrected import CorrectedCompositeLoss
from l1deepmet.data.loader import split_preprocessed_features

resolqt = lambda y: (np.percentile(y, 84) - np.percentile(y, 16)) / 2.0
DATA_DIR = Path("/home/users/dprimosc/L1DeepMET/preprocessed/25Jul8_140X_v0")
ABL_DIR = Path("outputs/loss_ablation_apr2026")
OUT = Path("reports/loss_diagnosis_apr2026")
OUT.mkdir(parents=True, exist_ok=True)

# Load test data
with h5py.File(DATA_DIR / "test.h5", "r") as f:
    X = f["features"][:]; Y = f["targets"][:]
gen_x, gen_y = Y[:, 0], Y[:, 1]
gen_pt = np.sqrt(gen_x ** 2 + gen_y ** 2)
raw_x = -X[:, :, 5].sum(axis=1)
raw_y = -X[:, :, 6].sum(axis=1)

inputs, pxpy, pdg, charge = split_preprocessed_features(X)
mi = {"continuous_inputs": inputs, "momentum_inputs": pxpy,
      "pdgid_inputs": pdg, "charge_inputs": charge}

# Load one representative model per binned_weight
custom = {"CastToInt": CastToInt, "ZeroReduce": ZeroReduce,
          "ShiftByConstant": ShiftByConstant, "SumOverParticles": SumOverParticles,
          "CorrectedCompositeLoss": CorrectedCompositeLoss}

models = {}
preds = {}
for bw in [0, 50, 200]:
    path = ABL_DIR / f"scalar_xybal10_bw{bw}_emb_w64_d3_seed42" / "best_model.keras"
    m = tf.keras.models.load_model(path, custom_objects=custom, compile=False)
    models[bw] = m
    preds[bw] = m.predict(mi, batch_size=512, verbose=0) * 100.0
    print(f"bw={bw}: X IQR/2 = {resolqt(preds[bw][:,0]-gen_x):.2f}", flush=True)

# Load ablation CSV for summary stats
ablation = []
with open(ABL_DIR / "ablation_results.csv") as f:
    for row in csv.DictReader(f):
        ablation.append(row)


# ────── PLOT 1: Headline summary ──────
fig, ax = plt.subplots(figsize=(10, 6))
labels = ["raw sum\n(0 params)", "constant w=0.69\n(1 param)", "per-type oracle\n(5 params)",
          "ML model\nbw=0 (9953)", "ML model\nbw=200 (9953)"]
xres = [38.30, 34.36, 34.97, 34.86, 42.78]
errs = [0, 0, 0, 0.30, 0.04]
colors = ["#888888", "#22cc77", "#22aa55", "#3377cc", "#cc3322"]
bars = ax.bar(labels, xres, yerr=errs, capsize=5, color=colors,
              edgecolor="black", linewidth=0.5)
for bar, val, e in zip(bars, xres, errs):
    label = f"{val:.2f}" + (f"\n±{e:.2f}" if e > 0 else "")
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.4, label,
            ha="center", va="bottom", fontweight="bold", fontsize=11)

ax.axhline(38.30, color="gray", linestyle="--", alpha=0.5, label="raw sum")
ax.set_ylabel("X IQR/2 [GeV]", fontsize=13)
ax.set_title("MET X Resolution: BinnedDeviation breaks ML, removing it fixes everything", fontsize=13)
ax.set_ylim(33, 45)
ax.grid(axis="y", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "headline_results.png", dpi=150)
plt.close()
print("Saved headline_results.png")


# ────── PLOT 2: Effective scaling per bw ──────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: histogram of model_pT/raw_pT per event for bw=0 vs bw=200
ax = axes[0]
raw_pt = np.sqrt(raw_x**2 + raw_y**2) + 1e-3
for bw, color in [(0, "#3377cc"), (200, "#cc3322")]:
    pt_pred = np.sqrt(preds[bw][:, 0]**2 + preds[bw][:, 1]**2)
    ratio = pt_pred / raw_pt
    ax.hist(ratio, bins=80, range=(0, 2.5), alpha=0.55, label=f"bw={bw} (mean={ratio.mean():.3f})",
            color=color, density=True)
ax.axvline(0.69, color="green", linestyle="--", alpha=0.7, label="optimal scalar w=0.69")
ax.axvline(1.00, color="black", linestyle=":", alpha=0.5, label="raw sum (w=1)")
ax.set_xlabel("model pT / raw-sum pT (per event)")
ax.set_ylabel("Density")
ax.set_title("Effective scaling: bw=0 finds optimal w; bw=200 over-amplifies")
ax.legend()
ax.set_xlim(0, 2.5)

# Right: response per gen-pT bin
ax = axes[1]
bins = [0, 30, 60, 100, 150, 250, 500]
centers = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
for bw, color in [(0, "#3377cc"), (200, "#cc3322")]:
    pred_pt = np.sqrt(preds[bw][:, 0]**2 + preds[bw][:, 1]**2)
    resps = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (gen_pt >= lo) & (gen_pt < hi)
        if m.sum() > 0:
            resps.append(pred_pt[m].mean() / gen_pt[m].mean())
        else:
            resps.append(np.nan)
    ax.plot(centers, resps, "o-", markersize=8, label=f"bw={bw}", color=color)

# Raw sum response
raw_pt_v = np.sqrt(raw_x**2 + raw_y**2)
raw_resp = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (gen_pt >= lo) & (gen_pt < hi)
    raw_resp.append(raw_pt_v[m].mean() / gen_pt[m].mean() if m.sum() else np.nan)
ax.plot(centers, raw_resp, "s--", markersize=8, label="raw sum", color="gray")
ax.axhline(1.0, color="black", linestyle=":", alpha=0.5)
ax.set_xlabel("Gen MET pT [GeV]")
ax.set_ylabel("⟨pT_reco⟩ / ⟨pT_gen⟩ in bin")
ax.set_title("Response per bin: bw=0 has bias (good); bw=200 forces response=1")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "scaling_and_response.png", dpi=150)
plt.close()
print("Saved scaling_and_response.png")


# ────── PLOT 3: Per-pT-bin resolution ──────
fig, ax = plt.subplots(figsize=(10, 6))
for bw, color, ls in [(0, "#3377cc", "-"), (200, "#cc3322", "-")]:
    rs = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (gen_pt >= lo) & (gen_pt < hi)
        if m.sum() > 30:
            rs.append(resolqt(preds[bw][m, 0] - gen_x[m]))
        else:
            rs.append(np.nan)
    ax.plot(centers, rs, "o-", markersize=8, label=f"ML bw={bw}", color=color, linestyle=ls)

raw_rs = [resolqt(raw_x[(gen_pt >= lo) & (gen_pt < hi)] - gen_x[(gen_pt >= lo) & (gen_pt < hi)])
          for lo, hi in zip(bins[:-1], bins[1:])]
ax.plot(centers, raw_rs, "s--", markersize=8, label="raw sum", color="gray")

ax.set_xlabel("Gen MET pT [GeV]")
ax.set_ylabel("X IQR/2 in bin [GeV]")
ax.set_title("Per-bin X resolution: bw=0 wins everywhere, bw=200 is worst at low MET")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "per_pt_bin_resolution.png", dpi=150)
plt.close()
print("Saved per_pt_bin_resolution.png")


# ────── PLOT 4: Learned per-particle weights ──────
# Extract w_i = Dense(...) - 1/normfac, applied to pxpy. Need to grab intermediate output.
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# We pull the post-shift weight ("weight_minus_one" or "met_weight" output) tensor
def get_weights(model, mi):
    # find layer "weight_minus_one" or "met_weight"
    for layer_name in ["weight_minus_one", "met_weight"]:
        try:
            extractor = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
            return extractor.predict(mi, batch_size=512, verbose=0).squeeze(-1)
        except (ValueError, KeyError):
            continue
    return None

w_per_bw = {}
for bw in [0, 200]:
    w_per_bw[bw] = get_weights(models[bw], mi) * 100.0  # un-normalize back to "physical" weight scale
                                                          # (since w is in units of 1/normfac in the network)

valid = X[:, :, 7] > 0
pdgid_arr = X[:, :, 7]
pid_names = ["ch. hadron", "n. hadron", "photon", "muon", "electron"]
oracle_w = [0.748, 0.692, 0.661, 0.469, 0.609]

for col_idx, bw in enumerate([0, 200]):
    if w_per_bw[bw] is None:
        continue
    w = w_per_bw[bw]
    # Top row: histogram of all valid weights
    ax = axes[0, col_idx]
    w_valid = w[valid]
    ax.hist(w_valid, bins=80, range=(-1, 3), alpha=0.7,
            color="#3377cc" if bw == 0 else "#cc3322")
    ax.axvline(1.0, color="black", linestyle=":", alpha=0.5, label="w=1 (raw sum)")
    ax.axvline(0.69, color="green", linestyle="--", alpha=0.7, label="optimal w=0.69")
    ax.axvline(w_valid.mean(), color="red", linestyle="-", alpha=0.7,
               label=f"mean = {w_valid.mean():.3f}")
    ax.set_xlabel("Learned per-particle weight w_i")
    ax.set_ylabel("Count")
    ax.set_title(f"bw={bw}: weight distribution (mean = {w_valid.mean():.3f})")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 2.5)

    # Bottom: per-type weight means vs oracle
    ax = axes[1, col_idx]
    type_means = []
    type_stds = []
    for pid in [1, 2, 3, 4, 5]:
        mask = pdgid_arr == pid
        wv = w[mask]
        type_means.append(wv.mean())
        type_stds.append(wv.std())
    x = np.arange(5)
    ax.bar(x - 0.2, type_means, 0.4, yerr=type_stds, capsize=5, alpha=0.7,
           label="learned", color="#3377cc" if bw == 0 else "#cc3322")
    ax.bar(x + 0.2, oracle_w, 0.4, alpha=0.7, label="oracle", color="green")
    for i, (lm, om) in enumerate(zip(type_means, oracle_w)):
        ax.text(i - 0.2, lm, f"{lm:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, om, f"{om:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(pid_names, rotation=20)
    ax.set_ylabel("Mean weight")
    ax.set_title(f"bw={bw}: per-type mean weight vs oracle")
    ax.legend()
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.4)

# Hide third column
axes[0, 2].axis("off")
axes[1, 2].axis("off")

# Add summary text in the empty third column
fig.text(0.72, 0.7, "Oracle (least-squares):\n"
                     "  ch. hadron : 0.748\n"
                     "  n. hadron  : 0.692\n"
                     "  photon     : 0.661\n"
                     "  muon       : 0.469\n"
                     "  electron   : 0.609\n\n"
                     "bw=0 model:\n"
                     "  finds means near 0.7\n"
                     "  → matches oracle\n\n"
                     "bw=200 model:\n"
                     "  forced toward >1\n"
                     "  → over-amplifies\n",
         fontsize=11, family="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.suptitle("Learned per-particle weights: bw=0 finds the oracle solution",
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "learned_weights.png", dpi=150)
plt.close()
print("Saved learned_weights.png")

print(f"\nAll plots written to {OUT}/")
