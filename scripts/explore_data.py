#!/usr/bin/env python3
"""
Data exploration script for L1DeepMET.

Investigates:
1. Feature distributions across particle types
2. PUPPI weight behavior and its relationship to genMET
3. Information available to the model vs PUPPI
4. Per-particle contributions to MET resolution
5. PUPPI-weighted px/py as potential new features
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/users/dprimosc/L1DeepMET/preprocessed/25Jul8_140X_v0")
OUTPUT_DIR = Path("reports/data_exploration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature indices
FEAT = {
    'pt': 0, 'eta': 1, 'phi': 2, 'puppi_w': 3, 'hcal_depth': 4,
    'px': 5, 'py': 6, 'pdgid': 7, 'charge': 8
}
PDGID_LABELS = {0: 'padding', 1: 'ch. hadron', 2: 'n. hadron', 3: 'photon', 4: 'muon', 5: 'electron'}
CHARGE_LABELS = {0: 'padding', 1: 'negative', 2: 'neutral', 3: 'positive'}


def load_all():
    """Load test data."""
    with h5py.File(DATA_DIR / "test.h5", 'r') as f:
        X = f['features'][:]
        Y = f['targets'][:]
    print(f"Loaded: X={X.shape}, Y={Y.shape}")
    return X, Y


def plot_feature_distributions(X):
    """Plot distributions of all 9 features."""
    # Mask out padding particles (pdgid == 0)
    valid = X[:, :, FEAT['pdgid']] > 0
    n_valid = valid.sum()
    n_total = valid.size
    n_events = X.shape[0]
    particles_per_event = valid.sum(axis=1)

    print(f"\n=== Particle Occupancy ===")
    print(f"Total particles: {n_total:,} ({n_events:,} events × 128 slots)")
    print(f"Valid particles: {n_valid:,} ({100*n_valid/n_total:.1f}%)")
    print(f"Per-event: mean={particles_per_event.mean():.1f}, "
          f"median={np.median(particles_per_event):.0f}, "
          f"min={particles_per_event.min()}, max={particles_per_event.max()}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    feat_names = ['pT', 'eta', 'phi', 'PUPPI weight', 'HGCal depth',
                  'px', 'py', 'pdgId (encoded)', 'charge (encoded)']

    for i, (ax, name) in enumerate(zip(axes.flat, feat_names)):
        vals = X[:, :, i][valid]
        if i in [7, 8]:  # categorical
            unique, counts = np.unique(vals, return_counts=True)
            labels = PDGID_LABELS if i == 7 else CHARGE_LABELS
            ax.bar([labels.get(int(u), str(int(u))) for u in unique], counts)
            ax.set_ylabel('Count')
            for j, (u, c) in enumerate(zip(unique, counts)):
                ax.text(j, c, f'{100*c/n_valid:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            ax.hist(vals, bins=100, log=True, alpha=0.7)
            ax.set_ylabel('Count (log)')
            ax.text(0.95, 0.95, f'mean={vals.mean():.3f}\nstd={vals.std():.3f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title(name)

    plt.suptitle('Feature Distributions (valid particles only)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=150)
    plt.close()
    print(f"Saved feature_distributions.png")


def plot_puppi_weight_analysis(X, Y):
    """Deep dive into PUPPI weights and their relationship to MET."""
    valid = X[:, :, FEAT['pdgid']] > 0
    puppi_w = X[:, :, FEAT['puppi_w']]
    px = X[:, :, FEAT['px']]
    py = X[:, :, FEAT['py']]
    pt = X[:, :, FEAT['pt']]
    pdgid = X[:, :, FEAT['pdgid']]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. PUPPI weight distribution by particle type
    ax = axes[0, 0]
    for pid_code in [1, 2, 3, 4, 5]:
        mask = pdgid == pid_code
        if mask.sum() > 0:
            w = puppi_w[mask]
            ax.hist(w, bins=50, alpha=0.5, label=PDGID_LABELS[pid_code], density=True)
    ax.set_xlabel('PUPPI weight')
    ax.set_ylabel('Density')
    ax.set_title('PUPPI weight by particle type')
    ax.legend(fontsize=8)

    # 2. PUPPI weight vs pT
    ax = axes[0, 1]
    w_flat = puppi_w[valid]
    pt_flat = pt[valid]
    ax.hist2d(pt_flat, w_flat, bins=[100, 50], cmap='viridis',
              range=[[0, 10], [0, 1.05]], norm=matplotlib.colors.LogNorm())
    ax.set_xlabel('pT (normalized)')
    ax.set_ylabel('PUPPI weight')
    ax.set_title('PUPPI weight vs pT')
    plt.colorbar(ax.collections[0], ax=ax)

    # 3. Fraction of pT carried by PUPPI w>0.5 particles
    ax = axes[0, 2]
    high_puppi = puppi_w > 0.5
    pt_high = (pt * high_puppi).sum(axis=1)
    pt_all = (pt * valid).sum(axis=1)
    frac = pt_high / (pt_all + 1e-8)
    ax.hist(frac, bins=50, alpha=0.7)
    ax.set_xlabel('Fraction of total pT from PUPPI w > 0.5')
    ax.set_ylabel('Events')
    ax.set_title(f'High-PUPPI pT fraction (mean={frac.mean():.3f})')
    ax.axvline(frac.mean(), color='red', linestyle='--')

    # 4. PUPPI MET vs gen MET (component-wise)
    puppi_met_x = -(puppi_w * px).sum(axis=1)
    puppi_met_y = -(puppi_w * py).sum(axis=1)
    gen_met_x = Y[:, 0]
    gen_met_y = Y[:, 1]

    # Raw sum (no PUPPI weights)
    raw_met_x = -(px * valid.astype(np.float32)).sum(axis=1)
    raw_met_y = -(py * valid.astype(np.float32)).sum(axis=1)

    ax = axes[1, 0]
    ax.hist2d(gen_met_x, puppi_met_x, bins=100, cmap='viridis',
              range=[[-300, 300], [-300, 300]], norm=matplotlib.colors.LogNorm())
    ax.plot([-300, 300], [-300, 300], 'r--', alpha=0.5)
    ax.set_xlabel('Gen MET_x [GeV]')
    ax.set_ylabel('PUPPI MET_x [GeV]')
    ax.set_title('PUPPI-weighted MET_x vs Gen')
    plt.colorbar(ax.collections[0], ax=ax)

    ax = axes[1, 1]
    ax.hist2d(gen_met_x, raw_met_x, bins=100, cmap='viridis',
              range=[[-300, 300], [-300, 300]], norm=matplotlib.colors.LogNorm())
    ax.plot([-300, 300], [-300, 300], 'r--', alpha=0.5)
    ax.set_xlabel('Gen MET_x [GeV]')
    ax.set_ylabel('Raw MET_x [GeV]')
    ax.set_title('Raw sum MET_x vs Gen')
    plt.colorbar(ax.collections[0], ax=ax)

    # 5. Resolution comparison: PUPPI-weighted vs raw
    ax = axes[1, 2]
    puppi_res_x = np.std(puppi_met_x - gen_met_x)
    puppi_res_y = np.std(puppi_met_y - gen_met_y)
    raw_res_x = np.std(raw_met_x - gen_met_x)
    raw_res_y = np.std(raw_met_y - gen_met_y)

    gen_met_pt = np.sqrt(gen_met_x**2 + gen_met_y**2)
    puppi_met_pt = np.sqrt(puppi_met_x**2 + puppi_met_y**2)
    raw_met_pt = np.sqrt(raw_met_x**2 + raw_met_y**2)
    puppi_res_pt = np.std(puppi_met_pt - gen_met_pt)
    raw_res_pt = np.std(raw_met_pt - gen_met_pt)

    labels = ['X', 'Y', 'pT']
    puppi_res = [puppi_res_x, puppi_res_y, puppi_res_pt]
    raw_res = [raw_res_x, raw_res_y, raw_res_pt]

    x_pos = np.arange(len(labels))
    w = 0.35
    ax.bar(x_pos - w/2, puppi_res, w, label='PUPPI-weighted', color='royalblue')
    ax.bar(x_pos + w/2, raw_res, w, label='Raw sum', color='coral')
    for i, (p, r) in enumerate(zip(puppi_res, raw_res)):
        ax.text(i - w/2, p, f'{p:.1f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + w/2, r, f'{r:.1f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Resolution (σ) [GeV]')
    ax.set_title('MET Resolution: PUPPI-weighted vs Raw sum')
    ax.legend()

    plt.suptitle('PUPPI Weight Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'puppi_weight_analysis.png', dpi=150)
    plt.close()

    print(f"\n=== PUPPI Weight Analysis ===")
    print(f"PUPPI-weighted MET resolution:  X={puppi_res_x:.1f}, Y={puppi_res_y:.1f}, pT={puppi_res_pt:.1f} GeV")
    print(f"Raw sum MET resolution:         X={raw_res_x:.1f}, Y={raw_res_y:.1f}, pT={raw_res_pt:.1f} GeV")
    print(f"Ratio (raw/PUPPI):              X={raw_res_x/puppi_res_x:.2f}, Y={raw_res_y/puppi_res_y:.2f}, pT={raw_res_pt/puppi_res_pt:.2f}")
    print(f"Saved puppi_weight_analysis.png")

    return puppi_met_x, puppi_met_y, raw_met_x, raw_met_y


def plot_puppi_weighted_features(X, Y):
    """
    Analyze PUPPI-weighted px/py as potential new features.
    If included, the model would start closer to the PUPPI baseline.
    """
    valid = X[:, :, FEAT['pdgid']] > 0
    puppi_w = X[:, :, FEAT['puppi_w']]
    px = X[:, :, FEAT['px']]
    py = X[:, :, FEAT['py']]
    gen_met_x = Y[:, 0]
    gen_met_y = Y[:, 1]

    # PUPPI-weighted momentum
    pw_px = puppi_w * px
    pw_py = puppi_w * py

    # Residual after PUPPI: what the model needs to learn
    puppi_met_x = -(puppi_w * px).sum(axis=1)
    puppi_met_y = -(puppi_w * py).sum(axis=1)
    residual_x = gen_met_x - puppi_met_x
    residual_y = gen_met_y - puppi_met_y

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Residual distribution
    ax = axes[0, 0]
    ax.hist(residual_x, bins=100, alpha=0.6, label=f'Δx (σ={np.std(residual_x):.1f})', range=[-200, 200])
    ax.hist(residual_y, bins=100, alpha=0.6, label=f'Δy (σ={np.std(residual_y):.1f})', range=[-200, 200])
    ax.set_xlabel('Gen MET - PUPPI MET [GeV]')
    ax.set_ylabel('Events')
    ax.set_title('Residual after PUPPI weighting')
    ax.legend()

    # 2. Residual vs gen MET pT
    gen_met_pt = np.sqrt(gen_met_x**2 + gen_met_y**2)
    ax = axes[0, 1]
    pt_bins = [0, 50, 100, 150, 200, 300, 500]
    residual_pt_std = []
    bin_centers = []
    for i in range(len(pt_bins) - 1):
        mask = (gen_met_pt >= pt_bins[i]) & (gen_met_pt < pt_bins[i+1])
        if mask.sum() > 10:
            res_x = residual_x[mask]
            res_y = residual_y[mask]
            residual_pt_std.append((np.std(res_x) + np.std(res_y)) / 2)
            bin_centers.append((pt_bins[i] + pt_bins[i+1]) / 2)
    ax.plot(bin_centers, residual_pt_std, 'o-', markersize=8)
    ax.set_xlabel('Gen MET pT [GeV]')
    ax.set_ylabel('Mean X/Y residual σ [GeV]')
    ax.set_title('PUPPI residual vs Gen MET pT')
    ax.grid(True, alpha=0.3)

    # 3. Correlation between PUPPI-weighted sum and residual
    ax = axes[1, 0]
    ax.hist2d(puppi_met_x, residual_x, bins=100, cmap='viridis',
              range=[[-300, 300], [-200, 200]], norm=matplotlib.colors.LogNorm())
    ax.set_xlabel('PUPPI MET_x [GeV]')
    ax.set_ylabel('Residual_x [GeV]')
    ax.set_title(f'PUPPI MET_x vs Residual (corr={np.corrcoef(puppi_met_x, residual_x)[0,1]:.3f})')
    plt.colorbar(ax.collections[0], ax=ax)

    # 4. Per-particle |Δw| needed
    # If PUPPI uses w_puppi and we need w_true, the correction is w_true - w_puppi
    # where w_true would make sum(w_true * px) = -gen_met_x
    # This is underdetermined, but we can look at the magnitude of correction needed
    ax = axes[1, 1]
    # How much MET do particles with different PUPPI weights contribute?
    w_bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    met_contrib_x = []
    met_contrib_labels = []
    for i in range(len(w_bins) - 1):
        mask = (puppi_w >= w_bins[i]) & (puppi_w < w_bins[i+1]) & valid
        contrib_x = np.abs(px * mask.astype(np.float32)).sum(axis=1).mean()
        met_contrib_x.append(contrib_x)
        met_contrib_labels.append(f'[{w_bins[i]:.1f},{w_bins[i+1]:.1f})')
    ax.bar(range(len(met_contrib_labels)), met_contrib_x)
    ax.set_xticks(range(len(met_contrib_labels)))
    ax.set_xticklabels(met_contrib_labels, rotation=45)
    ax.set_xlabel('PUPPI weight range')
    ax.set_ylabel('Mean |px| contribution per event [GeV]')
    ax.set_title('Momentum contribution by PUPPI weight bin')

    plt.suptitle('Can PUPPI-weighted features help?', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'puppi_weighted_features.png', dpi=150)
    plt.close()
    print(f"\n=== PUPPI Residual Analysis ===")
    print(f"Residual σ:   X={np.std(residual_x):.1f}, Y={np.std(residual_y):.1f} GeV")
    print(f"PUPPI-gen correlation: X={np.corrcoef(puppi_met_x, residual_x)[0,1]:.3f}, "
          f"Y={np.corrcoef(puppi_met_y, residual_y)[0,1]:.3f}")
    print(f"Saved puppi_weighted_features.png")


def plot_particle_type_met_contribution(X, Y):
    """Break down MET contribution by particle type."""
    valid = X[:, :, FEAT['pdgid']] > 0
    puppi_w = X[:, :, FEAT['puppi_w']]
    px = X[:, :, FEAT['px']]
    py = X[:, :, FEAT['py']]
    pdgid = X[:, :, FEAT['pdgid']]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pid_names = []
    puppi_met_x_parts = []
    raw_met_x_parts = []
    n_particles_parts = []

    for pid_code in [1, 2, 3, 4, 5]:
        mask = pdgid == pid_code
        n_part = mask.sum(axis=1).mean()
        # PUPPI-weighted MET contribution per type
        puppi_contrib_x = -(puppi_w * px * mask.astype(np.float32)).sum(axis=1)
        raw_contrib_x = -(px * mask.astype(np.float32)).sum(axis=1)

        pid_names.append(PDGID_LABELS[pid_code])
        puppi_met_x_parts.append(puppi_contrib_x)
        raw_met_x_parts.append(raw_contrib_x)
        n_particles_parts.append(n_part)

    # 1. Number of particles
    ax = axes[0]
    ax.bar(pid_names, n_particles_parts, color=['C0', 'C1', 'C2', 'C3', 'C4'])
    for i, n in enumerate(n_particles_parts):
        ax.text(i, n, f'{n:.1f}', ha='center', va='bottom')
    ax.set_ylabel('Mean particles per event')
    ax.set_title('Particle multiplicity by type')

    # 2. Resolution contribution by type (PUPPI weighted)
    ax = axes[1]
    gen_met_x = Y[:, 0]
    # Variance of PUPPI MET contribution from each type
    var_contribs = [np.std(c) for c in puppi_met_x_parts]
    ax.bar(pid_names, var_contribs, color=['C0', 'C1', 'C2', 'C3', 'C4'])
    for i, v in enumerate(var_contribs):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    ax.set_ylabel('σ of MET_x contribution [GeV]')
    ax.set_title('PUPPI MET_x fluctuation by particle type')

    # 3. Charged vs neutral breakdown
    ax = axes[2]
    charge = X[:, :, FEAT['charge']]
    charged_mask = (charge == 1) | (charge == 3)  # neg or pos
    neutral_mask = (charge == 2) & valid

    # MET from charged (PUPPI-weighted) vs neutral
    ch_puppi_met_x = -(puppi_w * px * charged_mask.astype(np.float32)).sum(axis=1)
    ne_puppi_met_x = -(puppi_w * px * neutral_mask.astype(np.float32)).sum(axis=1)
    ch_res = np.std(ch_puppi_met_x)
    ne_res = np.std(ne_puppi_met_x)

    n_charged = charged_mask.sum(axis=1).mean()
    n_neutral = neutral_mask.sum(axis=1).mean()

    labels = [f'Charged\n(n≈{n_charged:.0f})', f'Neutral\n(n≈{n_neutral:.0f})']
    ax.bar(labels, [ch_res, ne_res], color=['steelblue', 'coral'])
    ax.text(0, ch_res, f'{ch_res:.1f}', ha='center', va='bottom')
    ax.text(1, ne_res, f'{ne_res:.1f}', ha='center', va='bottom')
    ax.set_ylabel('σ of PUPPI MET_x contribution [GeV]')
    ax.set_title('Charged vs Neutral MET fluctuation')

    plt.suptitle('MET Contribution by Particle Type', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'particle_type_met.png', dpi=150)
    plt.close()

    print(f"\n=== Particle Type Analysis ===")
    for name, n, v in zip(pid_names, n_particles_parts, var_contribs):
        print(f"  {name:12s}: n={n:.1f}/event, σ(MET_x)={v:.1f} GeV")
    print(f"  Charged: n={n_charged:.0f}, σ={ch_res:.1f} GeV")
    print(f"  Neutral: n={n_neutral:.0f}, σ={ne_res:.1f} GeV")
    print(f"Saved particle_type_met.png")


def plot_gen_met_distributions(Y):
    """Analyze gen MET target distributions."""
    gen_met_x = Y[:, 0]
    gen_met_y = Y[:, 1]
    gen_met_pt = np.sqrt(gen_met_x**2 + gen_met_y**2)
    gen_met_phi = np.arctan2(gen_met_y, gen_met_x)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.hist(gen_met_pt, bins=100, range=[0, 500], alpha=0.7)
    ax.set_xlabel('Gen MET pT [GeV]')
    ax.set_ylabel('Events')
    ax.set_title(f'Gen MET pT (mean={gen_met_pt.mean():.1f}, median={np.median(gen_met_pt):.1f})')
    ax.axvline(gen_met_pt.mean(), color='red', linestyle='--', label=f'mean={gen_met_pt.mean():.1f}')
    ax.legend()

    ax = axes[0, 1]
    ax.hist(gen_met_phi, bins=50, alpha=0.7)
    ax.set_xlabel('Gen MET phi [rad]')
    ax.set_ylabel('Events')
    ax.set_title('Gen MET phi (should be uniform)')

    ax = axes[1, 0]
    ax.hist(gen_met_x, bins=100, range=[-300, 300], alpha=0.6, label=f'x (σ={np.std(gen_met_x):.1f})')
    ax.hist(gen_met_y, bins=100, range=[-300, 300], alpha=0.6, label=f'y (σ={np.std(gen_met_y):.1f})')
    ax.set_xlabel('Gen MET component [GeV]')
    ax.set_ylabel('Events')
    ax.set_title('Gen MET X/Y components')
    ax.legend()

    # pT spectrum breakdown
    ax = axes[1, 1]
    pt_bins = [0, 20, 50, 100, 200, 300, 500]
    counts = []
    for i in range(len(pt_bins) - 1):
        mask = (gen_met_pt >= pt_bins[i]) & (gen_met_pt < pt_bins[i+1])
        counts.append(mask.sum())
    labels = [f'{pt_bins[i]}-{pt_bins[i+1]}' for i in range(len(pt_bins)-1)]
    ax.bar(labels, counts)
    for i, c in enumerate(counts):
        ax.text(i, c, f'{c}\n({100*c/len(gen_met_pt):.1f}%)', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Gen MET pT bin [GeV]')
    ax.set_ylabel('Events')
    ax.set_title('Gen MET pT spectrum')

    plt.suptitle('Gen MET Target Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gen_met_distributions.png', dpi=150)
    plt.close()
    print(f"\n=== Gen MET Distributions ===")
    print(f"pT: mean={gen_met_pt.mean():.1f}, std={gen_met_pt.std():.1f}, median={np.median(gen_met_pt):.1f} GeV")
    print(f"X:  mean={gen_met_x.mean():.1f}, std={np.std(gen_met_x):.1f} GeV")
    print(f"Y:  mean={gen_met_y.mean():.1f}, std={np.std(gen_met_y):.1f} GeV")
    print(f"Saved gen_met_distributions.png")


def information_gap_analysis(X, Y):
    """
    Quantify the information gap between what the model sees and PUPPI.
    Key question: if we added PUPPI-weighted px/py as features, how much
    closer to optimal would the starting point be?
    """
    valid = X[:, :, FEAT['pdgid']] > 0
    puppi_w = X[:, :, FEAT['puppi_w']]
    px = X[:, :, FEAT['px']]
    py = X[:, :, FEAT['py']]
    gen_met_x = Y[:, 0]
    gen_met_y = Y[:, 1]
    gen_met_pt = np.sqrt(gen_met_x**2 + gen_met_y**2)

    # Three MET reconstruction strategies the model could learn:
    # 1. Raw sum (all particles equally weighted)
    raw_met_x = -(px * valid.astype(np.float32)).sum(axis=1)
    raw_met_y = -(py * valid.astype(np.float32)).sum(axis=1)

    # 2. PUPPI-weighted sum (use PUPPI weights directly)
    puppi_met_x = -(puppi_w * px).sum(axis=1)
    puppi_met_y = -(puppi_w * py).sum(axis=1)

    # 3. Optimal scalar weight (oracle: best constant w)
    # Minimize ||gen + w * sum(px)||^2 over w
    sum_px = px.sum(axis=1)
    sum_py = py.sum(axis=1)
    # w* = -<gen_x * sum_px + gen_y * sum_py> / <sum_px^2 + sum_py^2>
    w_opt = -(gen_met_x * sum_px + gen_met_y * sum_py).mean() / (sum_px**2 + sum_py**2).mean()
    scalar_met_x = -w_opt * sum_px
    scalar_met_y = -w_opt * sum_py

    results = {}
    for name, met_x, met_y in [
        ('Raw sum', raw_met_x, raw_met_y),
        ('PUPPI-weighted', puppi_met_x, puppi_met_y),
        (f'Optimal scalar (w={w_opt:.4f})', scalar_met_x, scalar_met_y),
    ]:
        res_x = np.std(met_x - gen_met_x)
        res_y = np.std(met_y - gen_met_y)
        met_pt = np.sqrt(met_x**2 + met_y**2)
        res_pt = np.std(met_pt - gen_met_pt)
        results[name] = (res_x, res_y, res_pt)

    print(f"\n=== Information Gap Analysis ===")
    print(f"{'Strategy':<30s} {'X res':>8s} {'Y res':>8s} {'pT res':>8s}")
    print('-' * 60)
    for name, (rx, ry, rpt) in results.items():
        print(f"{name:<30s} {rx:8.1f} {ry:8.1f} {rpt:8.1f}")
    print('-' * 60)
    print(f"{'ML model (best, scalar_xybal10)':<30s} {'42.7':>8s} {'42.7':>8s} {'44.1':>8s}")
    print(f"{'PUPPI baseline':<30s} {'38.3':>8s} {'38.4':>8s} {'44.3':>8s}")
    print()
    print("Key insight: The gap between PUPPI-weighted sum and PUPPI baseline")
    print("represents what PUPPI's full algorithm adds beyond simple weighting.")
    print(f"PUPPI uses vertex association + track quality info that our 9 features lack.")


def main():
    print("=" * 70)
    print("L1DeepMET Data Exploration")
    print("=" * 70)

    X, Y = load_all()

    plot_feature_distributions(X)
    plot_puppi_weight_analysis(X, Y)
    plot_puppi_weighted_features(X, Y)
    plot_particle_type_met_contribution(X, Y)
    plot_gen_met_distributions(Y)
    information_gap_analysis(X, Y)

    print("\n" + "=" * 70)
    print(f"All plots saved to {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
