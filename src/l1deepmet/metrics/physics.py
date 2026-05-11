"""Physics evaluation metrics for L1 MET reconstruction.

Shared by ``scripts/evaluate.py`` and ``scripts/ablation.py``. Each function
takes raw (gen_xy, reco_xy) or (gen_xy, reco_pt) arrays in GeV and returns a
plain Python dict of named floats — suitable for JSON / CSV export.

The metrics are physics-meaningful, not just regression-statistical:

- ``met_pt_resolution``    : IQR/2 of (gen_pt − response_corrected_reco_pt) for
                             events with gen_pt ≥ 50 GeV. Response correction
                             applied per-bin so resolution is not contaminated
                             by bias. This is the L1 figure-of-merit.
- ``met_x_resolution``     : IQR/2 of (gen_x − reco_x). All events.
- ``met_y_resolution``     : IQR/2 of (gen_y − reco_y). All events.
- ``mean_response``        : average of per-bin response over pT bins.
- ``phi_resolution``       : IQR/2 of wrapped Δφ between gen and reco.
- ``response_{lo}_{hi}``   : per-pT-bin response (reco/gen mean).
- ``auc``                  : ROC-AUC for signal (gen MET > 200) vs background
                             (gen MET < 50) discrimination by reco MET pT.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Default pT bin edges for response and resolution (GeV).
DEFAULT_PT_BINS = (50.0, 100.0, 200.0, 300.0, 400.0, float("inf"))


def resolqt(y: np.ndarray) -> float:
    """Inter-quantile half-width: (P84 − P16) / 2. Robust 1-sigma proxy."""
    return float((np.percentile(y, 84) - np.percentile(y, 16)) / 2.0)


def convert_xy_to_pt_phi(xy: np.ndarray) -> np.ndarray:
    """(N, 2) of (px, py) → (N, 2) of (pt, phi)."""
    out = np.empty_like(xy)
    out[:, 0] = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    out[:, 1] = np.arctan2(xy[:, 1], xy[:, 0])
    return out


def phi_diff(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """Wrapped (gen − reco) phi in (−π, π]."""
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def compute_resolution_metrics(
    gen_xy: np.ndarray,
    reco_xy: np.ndarray,
    pt_bins: Tuple[float, ...] = DEFAULT_PT_BINS,
) -> Dict[str, float]:
    """Resolution + response per pT bin + phi resolution.

    Returns the same keys as the prior ``compute_metrics`` in evaluate.py
    so downstream CSVs and plot scripts continue to work.
    """
    gen_pt_phi = convert_xy_to_pt_phi(gen_xy)
    reco_pt_phi = convert_xy_to_pt_phi(reco_xy)
    gen_pt, gen_phi = gen_pt_phi[:, 0], gen_pt_phi[:, 1]
    reco_pt, reco_phi = reco_pt_phi[:, 0], reco_pt_phi[:, 1]

    # Per-bin response.
    responses = []
    for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
        mask = (gen_pt >= lo) & (gen_pt < hi)
        n = int(mask.sum())
        resp = float(np.mean(reco_pt[mask] / gen_pt[mask])) if n > 10 else 1.0
        responses.append((lo, hi, resp, n))

    # Apply response correction to reco_pt for the resolution computation.
    high_pt_mask = gen_pt >= pt_bins[0]
    reco_pt_corrected = np.copy(reco_pt)
    for lo, hi, resp, _ in responses:
        if resp > 0.01:
            mask = (gen_pt >= lo) & (gen_pt < hi)
            reco_pt_corrected[mask] = reco_pt[mask] / resp

    pt_residuals = gen_pt - reco_pt_corrected
    met_pt_resolution = resolqt(pt_residuals[high_pt_mask])

    met_x_resolution = resolqt(gen_xy[:, 0] - reco_xy[:, 0])
    met_y_resolution = resolqt(gen_xy[:, 1] - reco_xy[:, 1])

    mean_response = float(np.mean([r[2] for r in responses if r[3] > 10]))
    phi_resolution = resolqt(phi_diff(gen_phi, reco_phi))

    out: Dict[str, float] = {
        "met_pt_resolution": met_pt_resolution,
        "met_x_resolution": met_x_resolution,
        "met_y_resolution": met_y_resolution,
        "mean_response": mean_response,
        "phi_resolution": phi_resolution,
    }
    for lo, hi, resp, _ in responses:
        hi_str = "inf" if np.isinf(hi) else f"{hi:.0f}"
        out[f"response_{lo:.0f}_{hi_str}"] = resp
    return out


def compute_trigger_metrics(
    gen_xy: np.ndarray,
    reco_pt: np.ndarray,
    signal_pt_min: float = 200.0,
    background_pt_max: float = 50.0,
) -> Dict[str, float]:
    """ROC-AUC for signal-vs-background discrimination by reco MET pT.

    Returns just the scalar ``auc``; the curve arrays are computed separately
    by ``compute_roc_curve`` if needed for plotting (kept out of the metrics
    dict so the result is JSON-serializable as plain floats).
    """
    gen_pt = np.sqrt(gen_xy[:, 0] ** 2 + gen_xy[:, 1] ** 2)
    signal_mask = gen_pt > signal_pt_min
    background_mask = gen_pt < background_pt_max
    n_sig, n_bg = int(signal_mask.sum()), int(background_mask.sum())
    if n_sig < 10 or n_bg < 10:
        return {"auc": float("nan"), "n_signal": n_sig, "n_background": n_bg}

    thresholds = np.arange(0.0, 501.0, 1.0)
    sig_eff = np.array([np.mean(reco_pt[signal_mask] > t) for t in thresholds])
    bg_eff = np.array([np.mean(reco_pt[background_mask] > t) for t in thresholds])
    sort_idx = np.argsort(bg_eff)
    auc = float(np.trapz(sig_eff[sort_idx], bg_eff[sort_idx]))
    return {"auc": auc, "n_signal": n_sig, "n_background": n_bg}


def compute_roc_curve(
    gen_xy: np.ndarray,
    reco_pt: np.ndarray,
    signal_pt_min: float = 200.0,
    background_pt_max: float = 50.0,
) -> Dict[str, np.ndarray]:
    """ROC arrays (not in the scalar metric dict; for plotting)."""
    gen_pt = np.sqrt(gen_xy[:, 0] ** 2 + gen_xy[:, 1] ** 2)
    signal_mask = gen_pt > signal_pt_min
    background_mask = gen_pt < background_pt_max
    thresholds = np.arange(0.0, 501.0, 1.0)
    return {
        "thresholds": thresholds,
        "sig_eff": np.array([np.mean(reco_pt[signal_mask] > t) for t in thresholds]),
        "bg_eff": np.array([np.mean(reco_pt[background_mask] > t) for t in thresholds]),
    }


def compute_turn_on(
    gen_xy: np.ndarray,
    reco_pt: np.ndarray,
    threshold: float,
    bin_edges: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Trigger efficiency vs gen MET at a fixed reco MET threshold."""
    if bin_edges is None:
        bin_edges = np.arange(0, 601, 20)
    gen_pt = np.sqrt(gen_xy[:, 0] ** 2 + gen_xy[:, 1] ** 2)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    efficiencies = np.zeros(len(centers))
    errors = np.zeros(len(centers))
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (gen_pt >= lo) & (gen_pt < hi)
        n = int(mask.sum())
        if n > 0:
            eff = float(np.mean(reco_pt[mask] > threshold))
            efficiencies[i] = eff
            errors[i] = np.sqrt(eff * (1 - eff) / n) if n > 1 else 0.0
    return {"centers": centers, "efficiencies": efficiencies, "errors": errors}


def compute_puppi_baseline(features: np.ndarray) -> np.ndarray:
    """PUPPI MET = ``-sum(px, py)`` over candidates.

    NOTE: ``L1PuppiCands`` are PUPPI-corrected at storage (stored ``pt`` =
    ``puppi_weight × pt_input``), so summing them reproduces ``L1PuppiMet`` to
    floating-point precision. Do NOT multiply by ``puppi_weight`` again.
    """
    puppi_px = -np.sum(features[:, :, 5], axis=1)
    puppi_py = -np.sum(features[:, :, 6], axis=1)
    return np.stack([puppi_px, puppi_py], axis=1)


def full_physics_card(
    gen_xy: np.ndarray,
    reco_xy: np.ndarray,
    puppi_xy: np.ndarray | None = None,
) -> Dict[str, float]:
    """Convenience: resolution metrics + AUC + comparisons to PUPPI baseline.

    All scalars, JSON-safe. Computes:
      - everything from compute_resolution_metrics(gen, reco)
      - ``auc``, ``n_signal``, ``n_background`` from compute_trigger_metrics
      - if ``puppi_xy`` given: ``puppi_*`` versions of all the above plus
        ``delta_*_resolution`` (model − PUPPI) and ``delta_auc``.
    """
    out: Dict[str, float] = {}
    out.update(compute_resolution_metrics(gen_xy, reco_xy))
    reco_pt = np.sqrt(reco_xy[:, 0] ** 2 + reco_xy[:, 1] ** 2)
    out.update(compute_trigger_metrics(gen_xy, reco_pt))

    if puppi_xy is not None:
        puppi_metrics = compute_resolution_metrics(gen_xy, puppi_xy)
        puppi_pt = np.sqrt(puppi_xy[:, 0] ** 2 + puppi_xy[:, 1] ** 2)
        puppi_trig = compute_trigger_metrics(gen_xy, puppi_pt)
        for k, v in puppi_metrics.items():
            out[f"puppi_{k}"] = v
        for k, v in puppi_trig.items():
            out[f"puppi_{k}"] = v
        # Deltas (model − PUPPI). Negative = model is better for resolutions;
        # positive = model is better for AUC.
        for k in ("met_pt_resolution", "met_x_resolution", "met_y_resolution",
                  "phi_resolution"):
            if k in puppi_metrics and k in out:
                out[f"delta_{k}"] = out[k] - puppi_metrics[k]
        if "auc" in out and "auc" in puppi_trig:
            out["delta_auc"] = out["auc"] - puppi_trig["auc"]
        # Convenient scale ratio
        out["mean_pred_pt_over_puppi_pt"] = float(
            (reco_pt / (puppi_pt + 1e-3)).mean()
        )

    return out
