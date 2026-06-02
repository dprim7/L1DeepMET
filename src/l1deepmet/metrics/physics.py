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

    Response convention is **ratio of means** per truth-pT bin:
        ``response = mean(pt_reco) / mean(pt_gen)``
    This matches ``src/l1deepmet/plotting.py`` and the L1METML legacy
    convention. It also has the nice property that the response correction
    ``pt_corrected = pt_reco / response`` produces
    ``mean(pt_corrected) = mean(pt_gen)`` exactly per bin, by construction,
    so any residual ``pt_corrected − pt_gen`` is mean-zero in the bin and
    its IQR/2 is a clean resolution.

    Earlier versions used **mean of ratios** (``mean(pt_reco / pt_gen)``),
    which is a different quantity — it coincides with the ratio of means
    only when reco scales linearly with gen, and it can blow up at low
    gen_pt due to per-event division. Reports before this change used the
    mean-of-ratios variant; absolute pT resolution numbers shifted slightly
    upon switching but qualitative conclusions (BinnedDeviation harm,
    MLP plateau, transformer over-shrinkage) held.
    """
    gen_pt_phi = convert_xy_to_pt_phi(gen_xy)
    reco_pt_phi = convert_xy_to_pt_phi(reco_xy)
    gen_pt, gen_phi = gen_pt_phi[:, 0], gen_pt_phi[:, 1]
    reco_pt, reco_phi = reco_pt_phi[:, 0], reco_pt_phi[:, 1]

    # Per-bin response = mean(reco) / mean(gen).  (Ratio of means.)
    # Distinct from mean(reco/gen), which we previously used by mistake.
    responses = []
    for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
        mask = (gen_pt >= lo) & (gen_pt < hi)
        n = int(mask.sum())
        if n > 10:
            mean_gen = float(np.mean(gen_pt[mask]))
            if mean_gen > 1e-3:
                resp = float(np.mean(reco_pt[mask])) / mean_gen
            else:
                resp = 1.0
        else:
            resp = 1.0
        responses.append((lo, hi, resp, n))

    # Apply response correction to reco_pt for the resolution computation.
    # mean(pt_reco / resp) == mean(pt_gen) per bin, so residuals are zero-mean.
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


# ─── L1 physics card extensions (STUBS — design pending) ──────────────────
# These are the missing pieces for the CLAUDE.md "Evaluation metrics" standing
# order. The interfaces below are deliberate guesses; sign off on each one
# before implementing — see reports/physics_card_design/DESIGN.md for the
# open questions.
#
# Test-first rule (CLAUDE.md): tests in tests/unit/metrics/ come before
# implementation. Stubs currently raise NotImplementedError; the test file
# at tests/unit/metrics/test_physics_card_stubs.py mirrors the layout and
# is `pytest.skip`-marked until each design is fixed.


# Rate normalization, ported from FastPUPPI
# NtupleProducer/python/scripts/jetHtSuite.py (``norm = 2760.0*11246/1000``):
# 2760 colliding bunches at the HL-LHC × 11246 Hz LHC orbit frequency / 1000.
# rate[kHz] = NORM_KHZ × (fraction of minbias events above the L1 threshold).
NORM_KHZ: float = 2760.0 * 11246.0 / 1000.0  # ≈ 31039.0 kHz total minbias collision rate


def compute_rate_vs_threshold(
    reco_pt_minbias: np.ndarray,
    thresholds_gev: np.ndarray | None = None,
    norm_khz: float = NORM_KHZ,
) -> Dict[str, np.ndarray]:
    """L1 MET trigger rate vs threshold on a minbias sample (FastPUPPI def).

    ``rate(thr) = norm_khz × P(reco_MET > thr)`` — the cumulative-from-above of
    the reco-MET distribution scaled to kHz by the colliding-bunch rate. Mirrors
    ``jetHtSuite.makeCumulativeHTEff`` / ``makeInclusiveEffRate``.

    NOTE: for a faithful L1 rate, ``reco_pt_minbias`` must be the **MinBias_PU200**
    (zero-bias) events. In a mixed test set with no per-event sample label, the
    standard proxy is low-gen-MET events; the caller selects the population.

    Returns ``{"thresholds", "pass_fraction", "rate_khz"}`` (length-N arrays).
    """
    if thresholds_gev is None:
        thresholds_gev = np.arange(0.0, 501.0, 1.0)
    thresholds_gev = np.asarray(thresholds_gev, dtype=float)
    reco = np.asarray(reco_pt_minbias, dtype=float)
    if reco.size == 0:
        raise ValueError("reco_pt_minbias must be non-empty")

    reco_sorted = np.sort(reco)
    idx = np.searchsorted(reco_sorted, thresholds_gev, side="right")
    pass_fraction = (reco_sorted.size - idx) / reco_sorted.size
    return {
        "thresholds": thresholds_gev,
        "pass_fraction": pass_fraction,
        "rate_khz": norm_khz * pass_fraction,
    }


def compute_working_point(
    reco_pt_minbias: np.ndarray,
    reco_pt_signal: np.ndarray,
    gen_pt_signal: np.ndarray,
    target_rate_khz: float,
    plateau_gen_min: float = 200.0,
    norm_khz: float = NORM_KHZ,
) -> Dict[str, float]:
    """Tier-1 decision quantity: signal efficiency at the fixed-rate working point.

    The threshold that yields ``target_rate_khz`` on minbias inverts the rate
    relation analytically (mirrors ``jetHtSuite.effForRate``):

        rate(thr) = norm_khz × P(reco_mb > thr) = target_rate_khz
        ⟹ thr = quantile(reco_mb, 1 − target_rate_khz / norm_khz)

    The deliverable is the **plateau signal efficiency** — the fraction of signal
    events with gen MET > ``plateau_gen_min`` that pass that threshold, i.e. the
    efficiency the trigger achieves at a fixed background rate. Unlike AUC, this is
    NOT invariant to a non-uniform under-prediction: a model whose response varies
    across MET pays here even if its AUC looks fine. That is precisely why this is
    the metric that adjudicates whether the uniform ~0.6 response actually costs us.

    Returns JSON-safe scalars: ``{threshold_gev, target_rate_khz,
    achieved_rate_khz, plateau_efficiency, n_signal_plateau}``.
    """
    reco_mb = np.asarray(reco_pt_minbias, dtype=float)
    p = float(target_rate_khz) / norm_khz
    if p <= 0.0:
        thr = float("inf")
    elif p >= 1.0:
        thr = 0.0
    else:
        thr = float(np.quantile(reco_mb, 1.0 - p))
    achieved = norm_khz * float(np.mean(reco_mb > thr)) if np.isfinite(thr) else 0.0

    reco_sig = np.asarray(reco_pt_signal, dtype=float)
    gen_sig = np.asarray(gen_pt_signal, dtype=float)
    plateau_mask = gen_sig > plateau_gen_min
    n_plateau = int(plateau_mask.sum())
    eff = float(np.mean(reco_sig[plateau_mask] > thr)) if n_plateau > 0 else float("nan")

    return {
        "threshold_gev": thr,
        "target_rate_khz": float(target_rate_khz),
        "achieved_rate_khz": achieved,
        "plateau_efficiency": eff,
        "n_signal_plateau": n_plateau,
    }


def compute_asymmetric_tails(
    gen_xy: np.ndarray,
    reco_xy: np.ndarray,
    fake_delta_gev: float = 50.0,
    miss_delta_gev: float = 50.0,
    pt_bins: Tuple[float, ...] = DEFAULT_PT_BINS,
) -> Dict[str, float]:
    """Fakes vs misses — asymmetric tail behaviour.

    Returns (proposed):
      ``frac_fake``  = P(reco_pt > gen_pt + ``fake_delta_gev``)  — eats rate
      ``frac_miss``  = P(reco_pt < gen_pt - ``miss_delta_gev``)  — kills efficiency
      ``frac_fake_<lo>_<hi>``, ``frac_miss_<lo>_<hi>``           — per gen-pT bin

    OPEN DESIGN QUESTIONS:
      - Fixed Δ_gev vs scaled (e.g., ``Δ = max(20, 0.2 * gen_pt)``)?
        Fixed is simpler; scaled tracks relative resolution at high pT.
      - Also report the *median* of the high-side / low-side tail
        distance? Useful when ``frac == 0`` but distributions still differ.
      - At our current test-set sizes the high-pT bins have <50 events
        — need a min-n-per-bin guard like ``compute_resolution_metrics``.
    """
    raise NotImplementedError("stub — pick fixed vs scaled Δ; per-bin output shape")


def compute_per_pu_card(
    gen_xy: np.ndarray,
    reco_xy: np.ndarray,
    n_vertex: np.ndarray,
    pu_bin_edges: Tuple[float, ...] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Tier-1 card stratified by reconstructed pileup.

    ``n_vertex`` is one integer per event — use ``nL1Vtx`` from the
    extended H5's ``event_features`` array.

    Returns ``{pu_bin_label: <full_physics_card result dict>}``.

    OPEN DESIGN QUESTIONS:
      - ``pu_bin_edges``: HL-LHC nominal PU is 140-200; reconstructed
        nL1Vtx is typically 0-30 depending on the emulator. Need
        calibrated bin edges. Sensible default: quartiles of the actual
        nL1Vtx distribution on the eval sample so bins are populated.
      - Per-bin AUC: each bin needs both signal-class and background-
        class events — at low N this can be empty. Skip or return NaN?
      - Should this also compute per-bin working_point efficiency?
        (Probably yes, but depends on compute_working_point's design.)
    """
    raise NotImplementedError("stub — pick PU bin edges + sparsity policy")


def compute_per_eta_card(
    gen_xy: np.ndarray,
    reco_xy: np.ndarray,
    per_event_eta_summary: np.ndarray,
    eta_split: float = 1.5,
) -> Dict[str, Dict[str, float]]:
    """Tier-1 card split by an event-level eta summary statistic.

    Returns ``{"barrel": <card>, "endcap": <card>}`` (or N-region
    generalisation).

    OPEN DESIGN QUESTIONS — the big one:
      What does "per-event eta" mean for a MET algorithm? MET itself
      doesn't have an eta. Options:
        (a) eta of the LEADING-pT candidate
        (b) eta of the highest-|w_i × p_i| candidate (model-dependent)
        (c) flag events as "barrel-MET" vs "endcap-MET" based on which
            region's candidates dominate the MET sum
        (d) skip per-event eta; instead, evaluate per-candidate
            resolution (residual on the contribution to MET) per eta —
            but that requires a per-candidate ground truth we don't have
      The candidate-eta summary stat needs to be computed UPSTREAM in
      the H5 schema or in the eval loop — it's not currently a column.
      Likely a small preprocess.py addition. User to pick.
    """
    raise NotImplementedError(
        "stub — define what 'per-event eta' is for a MET algorithm"
    )


def compute_puppi_ablation(
    eval_fn,
    features: np.ndarray,
    gen_xy: np.ndarray,
    puppi_slot_index: int = 3,
) -> Dict[str, float]:
    """PUPPI-ablation: re-eval the model with ``puppi_weight = 1.0`` for
    every candidate; report the full physics card delta vs unmodified.

    Workflow:
      1. ``reco_baseline = eval_fn(features)``.
      2. ``features_ablated = features.copy();
            features_ablated[:, :, puppi_slot_index] = 1.0;
            reco_ablated = eval_fn(features_ablated)``.
      3. Compute ``full_physics_card`` on both, return ``ablated − baseline``.

    A model that doesn't degrade under ablation isn't using PUPPI weights —
    it's re-deriving (probably worse) PU rejection from raw inputs.

    OPEN DESIGN QUESTIONS:
      - ``eval_fn`` signature: features → predicted MET in GeV. Caller
        passes a closure that handles model + normalisation + split into
        ``continuous_inputs`` / ``momentum_inputs``. Reasonable, but
        commits us to a specific eval-time API.
      - ``puppi_slot_index`` defaults to 3 to match the extended layout in
        ``params.yaml``; should this be a feature-name lookup against the
        H5's ``feature_layout`` attr instead?
      - Random-feature ablation (set puppi_weight to a U[0,1] random)
        tests something slightly different (robustness vs reliance) —
        separate function, maybe.
    """
    raise NotImplementedError("stub — fix eval_fn API + slot-vs-name lookup")
