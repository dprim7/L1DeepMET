#!/usr/bin/env python3
"""Physics-validation plots for the extended-v1 batch, per sample type.

CMS style (mplhep, as in src/l1deepmet/plotting.py). Reads perfNano ROOT
directly (pre-preprocessing, so we validate the ntuples themselves).

Figures (default output: outputs/ntuple_validation_plots/):
  fig_v1_kinematics.png    — per-sample overlays of core distributions
  fig_v1_charged_vs_pt.png — charged-candidate variables profiled vs pT
  fig_v1_neutral_vs_pt.png — neutral-candidate variables profiled vs pT
  fig_v1_consistency.png   — hw-vs-float, Layer2-vs-Puppi MET, depth-vs-eta
  fig_v1_met_physics.png   — gen/PUPPI MET spectra + response per sample

Usage: python scripts/ntuple_validation_plots.py [--tag-dir DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

plt.style.use(hep.style.CMS)

TAG_DIR = Path("/ceph/cms/store/user/dprimosc/l1deepmet/26Sep2_142_extended_v1")
SAMPLES = ["TT_PU200", "VBFHToInvisible_PU200", "MinBias_PU200",
           "WJetsToLNu_PU200", "DYToLL_PU200"]
COLORS = {"TT_PU200": "#5790fc", "VBFHToInvisible_PU200": "#f89c20",
          "MinBias_PU200": "#e42536", "WJetsToLNu_PU200": "#964a8b",
          "DYToLL_PU200": "#9c9ca1"}
LABELS = {"TT_PU200": "TT", "VBFHToInvisible_PU200": "VBF H(inv)",
          "MinBias_PU200": "MinBias", "WJetsToLNu_PU200": "W+jets",
          "DYToLL_PU200": "DY"}

PT_BINS = np.geomspace(1.0, 120.0, 16)
PT_MID = np.sqrt(PT_BINS[:-1] * PT_BINS[1:])


def _cms(ax):
    hep.cms.label(ax=ax, llabel="Simulation Preliminary",
                  rlabel="Phase-2 PU 200 (14 TeV)", fontsize=13)


def load(sample: str) -> dict | None:
    files = sorted((TAG_DIR / sample / "FP" / "Phase2Spring24").glob("perfNano_*.root"))
    if not files:
        return None
    branches = ["pt", "eta", "phi", "puppiWeight", "charge", "pdgId", "z0",
                "hwZ0", "hwPt", "hwEta", "hwTkQuality", "hwEmID",
                "trackNStubs", "trackChi2RPhi", "trackChi2Bend", "trackMvaQual",
                "trackPtError", "trackHitPattern",
                "clPt", "clEmEt", "clSigmaRR", "clAbsZBary", "clHoE"]
    out: dict = {}
    arrays = []
    ev_arrays = []
    for f in files:
        t = uproot.open(f)["Events"]
        arrays.append(t.arrays([f"L1PuppiCands_{b}" for b in branches]))
        ev_arrays.append(t.arrays(["genMet_pt", "L1PuppiMet_pt", "L1Layer2Met_pt",
                                   "L1Vtx_z0", "nL1PuppiCands",
                                   "HGCal3DCl_pfPuIdScore"]))
    cands = ak.concatenate(arrays)
    evs = ak.concatenate(ev_arrays)
    for b in branches:
        out[b] = cands[f"L1PuppiCands_{b}"]
    for b in ("genMet_pt", "L1PuppiMet_pt", "L1Layer2Met_pt", "nL1PuppiCands"):
        out[b] = np.asarray(ak.flatten(evs[b], axis=None) if evs[b].ndim > 1 else evs[b])
    out["lead_vtx_z0"] = np.asarray(ak.firsts(evs["L1Vtx_z0"]))
    out["hgc_puid"] = np.asarray(ak.flatten(evs["HGCal3DCl_pfPuIdScore"], axis=None))
    out["charged"] = out["charge"] != 0
    out["neutral"] = ~out["charged"]
    return out


def profile(pt, val, mask):
    """Mean ± sem of val in PT_BINS, over candidates passing mask."""
    p = np.asarray(ak.flatten(pt[mask]))
    v = np.asarray(ak.flatten(val[mask]), dtype=np.float64)
    means, errs = np.full(len(PT_MID), np.nan), np.full(len(PT_MID), np.nan)
    idx = np.digitize(p, PT_BINS) - 1
    for i in range(len(PT_MID)):
        sel = v[idx == i]
        if len(sel) > 3:
            means[i] = sel.mean()
            errs[i] = sel.std() / np.sqrt(len(sel))
    return means, errs


def overlay_hist(ax, data, bins, xlabel, log=False, density=True):
    for s, d in data.items():
        if d is None:
            continue
        ax.hist(np.clip(d, bins[0], bins[-1]), bins=bins, histtype="step",
                lw=1.8, density=density, color=COLORS[s], label=LABELS[s])
    ax.set_xlabel(xlabel, fontsize=13)
    if log:
        ax.set_yscale("log")
    ax.tick_params(labelsize=11)


def fig_kinematics(D, out):
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    panels = [
        ("pt", np.geomspace(1, 120, 40), "candidate $p_T$ [GeV]", True, "flat"),
        ("eta", np.linspace(-5.2, 5.2, 40), r"candidate $\eta$", False, "flat"),
        ("puppiWeight", np.linspace(0, 1.02, 41), "PUPPI weight (neutral)", True, "neutral"),
        ("z0", np.linspace(-16, 16, 40), "$z_0$ [cm] (charged)", False, "charged"),
        ("hwTkQuality", np.arange(-0.5, 8.5, 1), "hwTkQuality (charged)", True, "charged"),
        ("hwEmID", np.arange(-0.5, 40.5, 1), "hwEmID (neutral)", True, "neutral"),
        ("nL1PuppiCands", np.linspace(0, 130, 40), "PUPPI candidates / event", False, "event"),
        ("clAbsZBary", np.linspace(0, 550, 40), "cl |z barycenter| [cm]", True, "cluster"),
        ("hgc_puid", np.linspace(-0.6, 0.9, 40), "HGCal3DCl PU-ID score", False, "event"),
    ]
    for ax, (var, bins, xl, log, kind) in zip(axes.flat, panels):
        data = {}
        for s, d in D.items():
            if d is None:
                data[s] = None
                continue
            if kind == "event":
                data[s] = d[var]
            elif kind == "flat":
                data[s] = np.asarray(ak.flatten(d[var]))
            elif kind in ("charged", "neutral"):
                data[s] = np.asarray(ak.flatten(d[var][d[kind]]))
            elif kind == "cluster":
                m = d["clPt"] > 0
                data[s] = np.asarray(ak.flatten(d[var][m]))
        overlay_hist(ax, data, bins, xl, log=log)
    _cms(axes[0, 0])
    axes[0, 0].legend(fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_v1_kinematics.png", dpi=110)
    plt.close(fig)


def fig_profiles(D, out, which):
    if which == "charged":
        panels = [("hwTkQuality", "mean hwTkQuality"),
                  ("trackNStubs", "mean N stubs"),
                  ("trackChi2RPhi", r"mean $\chi^2_{r\phi}$ (binned)"),
                  ("trackChi2Bend", r"mean $\chi^2_{bend}$ (binned)"),
                  ("trackMvaQual", "mean track MVA quality"),
                  ("dz0", r"mean $|z_0 - z_0^{PV}|$ [cm]")]
        fname = "fig_v1_charged_vs_pt.png"
    else:
        panels = [("puppiWeight", "mean PUPPI weight"),
                  ("emid_frac", "frac(hwEmID > 0)"),
                  ("cl_frac", "frac(has cluster)"),
                  ("clSigmaRR", r"mean cl $\sigma_{RR}$"),
                  ("clAbsZBary", "mean cl |z bary| [cm]"),
                  ("clHoE", "mean cl H/E")]
        fname = "fig_v1_neutral_vs_pt.png"
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (var, yl) in zip(axes.flat, panels):
        for s, d in D.items():
            if d is None:
                continue
            base = d["charged"] if which == "charged" else d["neutral"]
            if var == "dz0":
                val = abs(d["z0"] - d["lead_vtx_z0"][:, None])
                mask = base
            elif var == "emid_frac":
                val, mask = d["hwEmID"] > 0, base
            elif var == "cl_frac":
                val, mask = d["clPt"] > 0, base
            elif var.startswith("cl"):
                val, mask = d[var], base & (d["clPt"] > 0)
            else:
                val, mask = d[var], base
            m, e = profile(d["pt"], val, mask)
            ax.errorbar(PT_MID, m, yerr=e, fmt="o-", ms=3.5, lw=1.4,
                        color=COLORS[s], label=LABELS[s])
        ax.set_xscale("log")
        ax.set_xlabel("candidate $p_T$ [GeV]", fontsize=13)
        ax.set_ylabel(yl, fontsize=13)
        ax.tick_params(labelsize=11)
    _cms(axes[0, 0])
    axes[0, 0].legend(fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=110)
    plt.close(fig)


def fig_consistency(D, out):
    ref = next(d for d in D.values() if d is not None)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    # hwZ0 * LSB vs z0
    ax = axes[0, 0]
    z0 = np.asarray(ak.flatten(ref["z0"][ref["charged"]]))
    hw = np.asarray(ak.flatten(ref["hwZ0"][ref["charged"]])) * 0.05
    ax.hist2d(z0, hw, bins=60, range=[[-16, 16], [-16, 16]], cmin=1)
    ax.plot([-16, 16], [-16, 16], "r--", lw=1)
    ax.set_xlabel("$z_0$ [cm]"), ax.set_ylabel(r"hwZ0 $\times$ 0.05 cm")
    # hwPt * LSB vs pt
    ax = axes[0, 1]
    pt = np.asarray(ak.flatten(ref["pt"]))
    hwpt = np.asarray(ak.flatten(ref["hwPt"])) * 0.25
    ax.hist2d(pt, hwpt, bins=60, range=[[0, 60], [0, 60]], cmin=1)
    ax.plot([0, 60], [0, 60], "r--", lw=1)
    ax.set_xlabel("$p_T$ [GeV]"), ax.set_ylabel(r"hwPt $\times$ 0.25 GeV")
    # Layer2 vs Puppi MET (all samples)
    ax = axes[0, 2]
    for s, d in D.items():
        if d is None:
            continue
        ax.scatter(d["L1PuppiMet_pt"], d["L1Layer2Met_pt"], s=4,
                   color=COLORS[s], label=LABELS[s], alpha=0.5)
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("L1PuppiMet [GeV]"), ax.set_ylabel("L1Layer2Met [GeV]")
    ax.legend(fontsize=9, frameon=False)
    # cluster depth vs |eta|
    ax = axes[1, 0]
    m = ref["clPt"] > 0
    ax.hist2d(np.abs(np.asarray(ak.flatten(ref["eta"][m]))),
              np.asarray(ak.flatten(ref["clAbsZBary"][m])),
              bins=60, range=[[0, 3.2], [300, 550]], cmin=1)
    ax.set_xlabel(r"candidate $|\eta|$"), ax.set_ylabel("cl |z bary| [cm]")
    # puppiWeight vs pT (neutral, 2D)
    ax = axes[1, 1]
    n = ref["neutral"]
    ax.hist2d(np.asarray(ak.flatten(ref["pt"][n])),
              np.asarray(ak.flatten(ref["puppiWeight"][n])),
              bins=[np.geomspace(1, 60, 40), np.linspace(0, 1.02, 40)], cmin=1)
    ax.set_xscale("log")
    ax.set_xlabel("neutral $p_T$ [GeV]"), ax.set_ylabel("PUPPI weight")
    # trackPtError/pt vs eta (charged)
    ax = axes[1, 2]
    c = ref["charged"]
    rel = np.asarray(ak.flatten(ref["trackPtError"][c])) / \
        np.maximum(np.asarray(ak.flatten(ref["pt"][c])), 1e-6)
    ax.hist2d(np.asarray(ak.flatten(ref["eta"][c])), np.clip(rel, 0, 1.5),
              bins=60, range=[[-2.6, 2.6], [0, 1.5]], cmin=1)
    ax.set_xlabel(r"charged $\eta$"), ax.set_ylabel(r"trackPtError / $p_T$")
    for a in axes.flat:
        a.tick_params(labelsize=11)
    _cms(axes[0, 0])
    fig.tight_layout()
    fig.savefig(out / "fig_v1_consistency.png", dpi=110)
    plt.close(fig)


def fig_met(D, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    overlay_hist(axes[0], {s: (d["genMet_pt"] if d else None) for s, d in D.items()},
                 np.linspace(0, 500, 40), "gen MET [GeV]", log=True)
    overlay_hist(axes[1], {s: (d["L1PuppiMet_pt"] if d else None) for s, d in D.items()},
                 np.linspace(0, 400, 40), "L1 PUPPI MET [GeV]", log=True)
    ax = axes[2]
    gbins = np.linspace(20, 400, 12)
    gmid = 0.5 * (gbins[:-1] + gbins[1:])
    for s, d in D.items():
        if d is None:
            continue
        g, r = d["genMet_pt"], d["L1PuppiMet_pt"] / np.maximum(d["genMet_pt"], 1e-9)
        means = [r[(g >= lo) & (g < hi)].mean() if ((g >= lo) & (g < hi)).sum() > 3
                 else np.nan for lo, hi in zip(gbins[:-1], gbins[1:])]
        ax.plot(gmid, means, "o-", ms=4, color=COLORS[s], label=LABELS[s])
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("gen MET [GeV]", fontsize=13)
    ax.set_ylabel(r"$\langle$PUPPI MET / gen MET$\rangle$", fontsize=13)
    ax.legend(fontsize=10, frameon=False)
    _cms(axes[0])
    axes[0].legend(fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_v1_met_physics.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", type=Path, default=TAG_DIR)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/ntuple_validation_plots"))
    args = ap.parse_args()
    globals()["TAG_DIR"] = args.tag_dir

    D = {s: load(s) for s in SAMPLES}
    present = [s for s, d in D.items() if d is not None]
    print("samples loaded:", present)
    args.out.mkdir(parents=True, exist_ok=True)
    fig_kinematics(D, args.out)
    fig_profiles(D, args.out, "charged")
    fig_profiles(D, args.out, "neutral")
    fig_consistency(D, args.out)
    fig_met(D, args.out)
    print("wrote 5 figures to", args.out)


if __name__ == "__main__":
    main()
