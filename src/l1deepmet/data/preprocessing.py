"""
Data preprocessing functions for L1DeepMET.

This module contains all functions for loading, preprocessing, and preparing
data for training/evaluation.
"""

from typing import Dict, Tuple, List, Optional, Union, Any
from pathlib import Path
import numpy as np # type: ignore
import awkward as ak # type: ignore
import uproot # type: ignore
import h5py # type: ignore
import logging

from l1deepmet.utils import to_np_array

logger = logging.getLogger(__name__)


# ─── Event-level feature catalogue ─────────────────────────────────────────
# Maps a clean feature name (used in params.yaml::event_feature_layout) to
# (ROOT branch name, aggregation mode). Aggregation modes:
#   - "scalar": branch is already one value per event (e.g. L1PuppiMet_pt
#               from a singleton FlatTable, or nL1Vtx count).
#   - "first":  branch is jagged (e.g. L1Vtx_z0 with multiple vertices/event);
#               take the leading entry per event, default 0.0 for empty rows.
#
# Adding a new feature here AND in params.yaml::preprocess.event_feature_layout
# is enough to surface it as an extra column in the H5 `event_features`
# dataset. Branches missing from the input ROOT file are zero-filled with a
# logged warning (mirrors _existing_branches/_safe_get for per-candidate).
EVENT_FEATURE_BRANCHES: Dict[str, Tuple[str, str]] = {
    # alternative MET algorithms (present in legacy + new ntuples)
    "puppi_met_pt":          ("L1PuppiMet_pt",         "scalar"),
    "puppi_met_phi":         ("L1PuppiMet_phi",        "scalar"),
    "puppi_met_central_pt":  ("L1PuppiMetCentral_pt",  "scalar"),
    "puppi_met_central_phi": ("L1PuppiMetCentral_phi", "scalar"),
    "pf_met_pt":             ("L1PFMet_pt",            "scalar"),
    "pf_met_phi":            ("L1PFMet_phi",           "scalar"),
    "calo_met_pt":           ("L1CaloMet_pt",          "scalar"),
    "calo_met_phi":          ("L1CaloMet_phi",         "scalar"),
    "tk_met_pt":             ("L1TKMet_pt",            "scalar"),
    "tk_met_phi":            ("L1TKMet_phi",           "scalar"),
    # Layer-2 correlator MET (NEW — from patched runPerformanceNTuple.py)
    "layer2_met_pt":         ("L1Layer2Met_pt",        "scalar"),
    "layer2_met_phi":        ("L1Layer2Met_phi",       "scalar"),
    # L1 primary vertices (NEW — from patched VertexWordFlatTableProducer)
    "lead_vtx_z0":           ("L1Vtx_z0",              "first"),
    "lead_vtx_sumpt":        ("L1Vtx_sumPt",           "first"),
    "n_vtx":                 ("nL1Vtx",                "scalar"),
}


def _load_event_field(field, agg: str = "scalar", default: float = 0.0) -> np.ndarray:
    """Extract per-event values from an awkward field as a (n_events,) np.float32 array.

    agg='scalar': field is already per-event (shape (n_events,)). Pass through.
    agg='first':  field is jagged var-length per row. Take [0] of each row;
                  rows with zero entries fall back to ``default``.
    """
    if agg == "scalar":
        return np.asarray(field).astype(np.float32)
    if agg == "first":
        padded = ak.pad_none(field, 1, clip=True)
        filled = ak.fill_none(padded, default)
        return np.asarray(filled[:, 0]).astype(np.float32)
    raise ValueError(
        f"Unknown agg mode: {agg!r}. Expected 'scalar' or 'first'."
    )


def event_features_from_arrays(
    arrays,
    event_feature_layout: List[str],
    n_events: int,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Build a (n_events, len(event_feature_layout)) array from one uproot
    iterate batch.

    Each feature is looked up in EVENT_FEATURE_BRANCHES; missing ROOT branches
    are silently zero-filled (the caller logs once at file-open time via
    _existing_branches). Unknown feature names raise ValueError — this is a
    configuration error, not a data-quality issue.
    """
    fields_available = set(getattr(arrays, "fields", ()))
    out = np.zeros((n_events, len(event_feature_layout)), dtype=dtype)
    for j, feat in enumerate(event_feature_layout):
        if feat not in EVENT_FEATURE_BRANCHES:
            raise ValueError(
                f"Unknown event feature: {feat!r}. "
                f"Add it to EVENT_FEATURE_BRANCHES in preprocessing.py."
            )
        branch, agg = EVENT_FEATURE_BRANCHES[feat]
        if branch not in fields_available:
            continue  # zero-fill
        out[:, j] = _load_event_field(arrays[branch], agg=agg, default=0.0)
    return out


def _existing_branches(files_with_tree: List[str], wanted: List[str]) -> List[str]:
    """Filter `wanted` down to branches that actually exist in the first ROOT file.

    Used so uproot.iterate doesn't crash when a config lists branches that exist
    in NEW ntuples (extended recipe) but not in OLD ones (or vice versa). The
    caller is responsible for substituting zeros for branches that get filtered
    out — see _safe_get below.
    """
    if not files_with_tree:
        return list(wanted)
    # Open the first file with uproot to read available branches
    first = files_with_tree[0]
    path = first.split(":")[0] if ":" in first else first
    tree = first.split(":")[1] if ":" in first else "Events"
    try:
        with uproot.open(path) as h:
            available = set(h[tree].keys())
    except Exception:
        return list(wanted)
    missing = [b for b in wanted if b not in available]
    if missing:
        logger.warning(
            f"Branches not in {path} (will be pad-filled at load time with "
            f"each column's declared pad/sentinel): {missing}"
        )
    return [b for b in wanted if b in available]


def _safe_get(arrays: Dict[str, Any], branch: str, max_pf: int,
              pad: float, default_shape_from: str) -> np.ndarray:
    """Load a per-candidate branch as a padded (n_events, max_pf) numpy array.

    If the branch is missing from `arrays` (because the file didn't have it
    and it was filtered out by _existing_branches), return a constant array
    of `pad` shaped like the always-present `default_shape_from` branch.
    Using the column's declared pad (its "absent" sentinel — see
    EXTENDED_DIRECT_BRANCHES) rather than zero keeps missing-branch cells
    indistinguishable from genuinely absent values, so one cleaning pass
    handles both and columns whose sentinel is -1/-999 never masquerade as
    real zeros (changed 2026-09-02 with the extended-v1 layout; previously
    missing branches were zero-filled).
    """
    # NB: do NOT use `branch in arrays` here — for an awkward Record that's
    # an element-wise broadcast, not a field-name lookup. Inspect .fields.
    if branch in getattr(arrays, "fields", ()):
        return to_np_array(arrays[branch], maxN=max_pf, pad=pad)
    # Fallback — shape from a reference branch we know exists
    ref = to_np_array(arrays[default_shape_from], maxN=max_pf, pad=0.0)
    return np.full_like(ref, pad)


def sanitize_extreme_values(
    X: np.ndarray,
    threshold: float = 1.0e6,
    replace: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Clamp |x| > threshold or non-finite (NaN/Inf) to `replace`.

    Some L1 cluster-ID MVAs (e.g. ``pfCluster.egVsPUMVAOut``,
    ``egVsPionMVAOut``) return numerical sentinels on the order of ±FLT_MAX
    (~±3.4e+38) when their inputs are invalid. Those values silently
    poison downstream normalization (std → ∞, gradients → NaN) without
    tripping a NaN/Inf check, because the floats are still finite — just
    enormous.

    The default threshold (1e6) is well above any legitimate physics value
    at L1 (hwPt maxes at ~3200, trackChi2 cuts ≤20, MET pt ≤ a few TeV)
    and well below FLT_MAX, so legitimate features pass through untouched
    while sentinels get replaced.

    Returns a sanitized COPY (input is not mutated) plus a stats dict
    with ``n_touched_per_feature`` — a 1-D array of length F summing
    touched cells over every axis except the last.
    """
    mask = (~np.isfinite(X)) | (np.abs(X) > threshold)
    if mask.ndim >= 2:
        axes = tuple(range(mask.ndim - 1))
        n_touched = mask.sum(axis=axes)
    else:
        n_touched = mask.astype(np.int64)
    Y = np.where(mask, np.asarray(replace, dtype=X.dtype), X)
    return Y, {"n_touched_per_feature": n_touched}


# Per-feature cleaning for the extended-26 layout, applied AFTER the global
# `sanitize_extreme_values` clamp. Each entry: (sentinels, max_abs, replace).
#   - any cell == a listed sentinel        -> replace
#   - any cell with |x| > max_abs (if set) -> replace
# Only the two failure modes the extended-26 validation found are targeted;
# the global clamp already handles ±FLT_MAX / NaN / Inf:
#   * caloEta/caloPhi carry -999 on neutrals (no pfTrack) — ~600x the real
#     ±2.5 scale, blows up BatchNorm -> neutralize to 0.
#   * clPuId/clEmId carry non-physical values from the -1 charged-candidate
#     fallback getting bit-promoted: a ±1e5–1e6 garbage population (slips under
#     the 1e6 global threshold) AND a softer tail out to a saturation spike at
#     exactly ±99.9. The real EG cluster MVA scores are O(1) (clEmId p99≈1.7,
#     clPuId p90≈0.25), so anything with |x| > 10 is corruption/overflow
#     (≤0.55% of neutrals) -> restore to -1, the sentinel the other cl* use.
# The -1 "absent" markers on z0 / track* / clPt / clEmEt are deliberately
# left alone (sane scale; model has encoded_pdgId/encoded_charge to read them).
# extended-v1 additions (audited 2026-09-02 on the 26Sep2 2k batch, all 5
# samples — no FLT_MAX/NaN garbage anywhere in the new columns):
#   * clEta/clPhi carry -999 on no-cluster candidates (same recipe guard as
#     caloEta/caloPhi) -> neutralize to 0.
#   * clHoE has a LONG PHYSICAL tail (p99.9 = 41, max = 688): tiny EM
#     denominators, real "very hadronic" clusters. Cap |x| > 50 TO 50 —
#     unlike clPuId we preserve ordering rather than restore a sentinel,
#     because large H/E is signal, not corruption. The -1 no-cluster /
#     no-HCal sentinel is untouched.
#   * everything else measured clean at source and stays raw. In particular
#     clAbsZBary == 0 is MEANINGFUL (barrel cluster — only HGCal fills a z
#     barycenter) and must never be cleaned.
EXTENDED_CLEANING_SPEC: Dict[str, Tuple[Tuple[float, ...], Optional[float], float]] = {
    "caloEta": ((-999.0,), None, 0.0),
    "caloPhi": ((-999.0,), None, 0.0),
    "clPuId": ((), 10.0, -1.0),
    "clEmId": ((), 10.0, -1.0),
    "clEta": ((-999.0,), None, 0.0),
    "clPhi": ((-999.0,), None, 0.0),
    "clHoE": ((), 50.0, 50.0),
}


# Per-candidate branch map for the extended layout: feature name ->
# (ROOT branch, pad value). Module-level so tests can pin the contract.
# Pads deliberately equal the recipe's own "absent" sentinel for each
# column, so padded slots and real absent-values are indistinguishable
# and one cleaning pass handles both.
EXTENDED_DIRECT_BRANCHES: Dict[str, Tuple[str, float]] = {
    "pt": ("L1PuppiCands_pt", 0.0),
    "eta": ("L1PuppiCands_eta", 0.0),
    "phi": ("L1PuppiCands_phi", 0.0),
    "puppi_weight": ("L1PuppiCands_puppiWeight", 0.0),
    "puppiWeight": ("L1PuppiCands_puppiWeight", 0.0),
    # `dxy` is the L1 PF candidate's impact-parameter field. Dead at source
    # in the current productions — the prompt 4-parameter track fit does not
    # fit d0, so even hwDxy is zero in the hardware word (verified 26Sep2).
    # Exposed for forward-compat with displaced tracking.
    "dxy":    ("L1PuppiCands_dxy",    0.0),
    # `dxyErr` is a legacy alias retained for back-compat with old
    # 25Jul8 ntuples (which had an L1PuppiCands_dxyErr branch that was
    # also uniformly zero); not present in 14_2_X production output.
    "dxyErr": ("L1PuppiCands_dxyErr", 1000.0),
    "mass": ("L1PuppiCands_mass", 0.0),
    "z0": ("L1PuppiCands_z0", 0.0),
    "hwPt": ("L1PuppiCands_hwPt", 0.0),
    "hwEta": ("L1PuppiCands_hwEta", 0.0),
    "hwPhi": ("L1PuppiCands_hwPhi", 0.0),
    "hwPuppiWeight": ("L1PuppiCands_hwPuppiWeight", 0.0),
    "hwQual": ("L1PuppiCands_hwQual", 0.0),
    "trackChi2RPhi": ("L1PuppiCands_trackChi2RPhi", -1.0),
    "trackChi2RZ":   ("L1PuppiCands_trackChi2RZ", -1.0),
    "trackChi2Bend": ("L1PuppiCands_trackChi2Bend", -1.0),
    "trackNStubs":   ("L1PuppiCands_trackNStubs", -1.0),
    "trackMvaQual":  ("L1PuppiCands_trackMvaQual", -1.0),
    "caloEta": ("L1PuppiCands_caloEta", -999.0),
    "caloPhi": ("L1PuppiCands_caloPhi", -999.0),
    "clPuId":  ("L1PuppiCands_clPuId", -1.0),
    "clEmId":  ("L1PuppiCands_clEmId", -1.0),
    "clPt":    ("L1PuppiCands_clPt", -1.0),
    "clEmEt":  ("L1PuppiCands_clEmEt", -1.0),
    # ─── extended-v1 (26Sep2 production): deployment-legal word fields.
    "hwZ0":        ("L1PuppiCands_hwZ0", 0.0),
    "hwDxy":       ("L1PuppiCands_hwDxy", 0.0),
    "hwTkQuality": ("L1PuppiCands_hwTkQuality", 0.0),
    "hwEmID":      ("L1PuppiCands_hwEmID", 0.0),
    # ─── extended-v1: cluster position/shape/depth (privileged).
    "clEta":      ("L1PuppiCands_clEta", -999.0),
    "clPhi":      ("L1PuppiCands_clPhi", -999.0),
    "clSigmaRR":  ("L1PuppiCands_clSigmaRR", -1.0),
    "clAbsZBary": ("L1PuppiCands_clAbsZBary", -1.0),
    "clHoE":      ("L1PuppiCands_clHoE", -1.0),
    "clPtError":  ("L1PuppiCands_clPtError", -1.0),
    # ─── extended-v1: track extras (privileged).
    "trackPtError":    ("L1PuppiCands_trackPtError", -1.0),
    "trackHitPattern": ("L1PuppiCands_trackHitPattern", -1.0),
}


def clean_extended_sentinels(
    X: np.ndarray,
    feature_layout: List[str],
    spec: Optional[Dict[str, Tuple[Tuple[float, ...], Optional[float], float]]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Layout-aware sentinel/garbage cleaning for the extended candidate block.

    Operates on the last axis using `feature_layout` to locate each named
    feature, so it works on (B, N, F) candidate arrays. Returns a cleaned
    COPY (input untouched) plus ``{"n_touched_per_feature": <len-F int array>}``,
    matching the `sanitize_extreme_values` interface so the two compose.
    See `EXTENDED_CLEANING_SPEC` for the policy and its rationale.
    """
    if spec is None:
        spec = EXTENDED_CLEANING_SPEC
    Y = X.copy()
    name_to_idx = {n: i for i, n in enumerate(feature_layout)}
    n_touched = np.zeros(len(feature_layout), dtype=np.int64)
    for name, (sentinels, max_abs, replace) in spec.items():
        i = name_to_idx.get(name)
        if i is None:
            continue
        col = Y[..., i]
        mask = np.zeros(col.shape, dtype=bool)
        for s in sentinels:
            mask |= col == s
        if max_abs is not None:
            mask |= np.abs(col) > max_abs
        n_touched[i] = int(mask.sum())
        col[mask] = np.asarray(replace, dtype=X.dtype)
        Y[..., i] = col
    return Y, {"n_touched_per_feature": n_touched}


def HCalDepth(
    hcal_first1: np.ndarray, hcal_first3: np.ndarray, hcal_first5: np.ndarray
) -> np.ndarray:
    """Effective center of energy depth in the hadronic calorimeter of Phase-2 HGCal.

    Mirrors convertNanoToHDF5.HCalDepth with numerically safe handling of zeros.
    """
    epsilon = 1.0e-10
    hcal_first5_safe = np.where(hcal_first5 == 0, epsilon, hcal_first5)
    depth_weighted = (
        hcal_first1 * 1.0
        + (hcal_first3 - hcal_first1) * 3.0
        + (hcal_first5 - hcal_first3) * 5.0
    ) / hcal_first5_safe
    depth_weighted = np.where(hcal_first5 == 0, 0.0, depth_weighted)
    return depth_weighted


def load_samples_to_numpy(
    data_root: Union[str, Path],
    sample_names: List[str],
    var_list: List[str],
    var_list_mc: Optional[List[str]] = None,
    *,
    max_pf: int,
    encoding: Dict[str, Dict[float, int]],
    include_mc: bool = True,
    step_size: str = "100 MB",
    file_pattern: str = "*.root",
    tree_name: str = "Events",
    dtype=np.float32,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]: 
    """
    Load ROOT samples and convert to numpy arrays.
    
    Returns a tuple of numpy arrays (features, targets) over all samples.
    """
    data_root = Path(data_root)
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    logger.info(f"Starting data loading from {data_root}")
    logger.info(f"Processing {len(sample_names)} samples: {sample_names}")

    for i, sample in enumerate(sample_names, 1):
        logger.info(f"[{i}/{len(sample_names)}] Processing sample: {sample}")
        
        files = sorted((data_root / sample).rglob(file_pattern))
        # Skip 0-byte ROOT files (e.g. from failed/held production jobs) — uproot.iterate
        # raises OSError ("received 0 bytes") on an empty file and aborts the whole sample.
        _n_all = len(files)
        files = [f for f in files if f.stat().st_size > 0]
        if len(files) < _n_all:
            logger.warning(f"  {sample}: skipped {_n_all - len(files)} empty/0-byte ROOT file(s)")
        if not files:
            raise FileNotFoundError(f"No ROOT files found for sample '{sample}' under {data_root / sample}")

        logger.info(f"  Found {len(files)} ROOT files")
        
        branches = list(var_list)
        if include_mc and var_list_mc:
            branches += var_list_mc
        logger.info(f"  Loading {len(branches)} branches from tree '{tree_name}'")
        logger.debug(f"  Branches: {branches}")
        
        X_parts: List[np.ndarray] = []
        Y_parts: List[np.ndarray] = []

        # Specify the tree name by appending to file paths
        files_with_tree = [str(f) + f":{tree_name}" for f in files]
        
        logger.info(f"  Starting iteration over files with step_size={step_size}")
        batch_count = 0
        
        for arrays in uproot.iterate(files_with_tree, expressions=branches, step_size=step_size, library="ak"):
            batch_count += 1
            logger.info(f"  Processing batch {batch_count}...")
            # Per-candidate inputs (awkward jagged -> dense with padding)
            pt = to_np_array(arrays["L1PuppiCands_pt"], maxN=max_pf, pad=0.0)
            eta = to_np_array(arrays["L1PuppiCands_eta"], maxN=max_pf, pad=0.0)
            phi = to_np_array(arrays["L1PuppiCands_phi"], maxN=max_pf, pad=0.0)
            pdgid = to_np_array(arrays["L1PuppiCands_pdgId"], maxN=max_pf, pad=-999.0)
            charge = to_np_array(arrays["L1PuppiCands_charge"], maxN=max_pf, pad=-999.0)
            puppiw = to_np_array(arrays["L1PuppiCands_puppiWeight"], maxN=max_pf, pad=0.0)
            dxyErr = to_np_array(arrays["L1PuppiCands_dxyErr"], maxN=max_pf, pad=1000.0)

            # KNOWN-BROKEN LEGACY JOIN ─────────────────────────────────────
            # HGCal3DCl_* branches are PER-CLUSTER, not per-candidate. Padding
            # them up to max_pf=128 with the same shape as L1PuppiCands_* and
            # then writing to X[:, :, candidate_index] places HGCal-cluster
            # values into candidate slots whose indices don't correspond to
            # each other. The resulting hcal_depth feature in legacy H5s is
            # essentially garbage (data exploration confirmed it's ≈ 0
            # everywhere after sanitisation).
            #
            # Phase 1 of the ntuple pipeline plan keeps this path intact for
            # bitwise-compatible reproduction of 25Jul8_140X_v0 outputs (we
            # haven't shown re-training a model on the corrected join yet),
            # but the extended preprocessor (load_samples_to_numpy_extended)
            # zero-fills hcal_depth explicitly to avoid propagating the bug.
            # The proper ΔR-based join is a Phase 3 item — see
            # reports/ntuple_pipeline_plan/PLAN.md §1.1 and P1.7.
            h1 = to_np_array(arrays["HGCal3DCl_firstHcal1layers"], maxN=max_pf, pad=0.0)
            h3 = to_np_array(arrays["HGCal3DCl_firstHcal3layers"], maxN=max_pf, pad=0.0)
            h5 = to_np_array(arrays["HGCal3DCl_firstHcal5layers"], maxN=max_pf, pad=0.0)
            hcalDepth = HCalDepth(h1, h3, h5)
            # ───────────────────────────────────────────────────────────────

            px = pt * np.cos(phi)
            py = pt * np.sin(phi)

            enc_pdg = np.vectorize(encoding["L1PuppiCands_pdgId"].__getitem__)(pdgid.astype(float))
            enc_chg = np.vectorize(encoding["L1PuppiCands_charge"].__getitem__)(charge.astype(float))

            nevents = pt.shape[0]
            X = np.zeros((nevents, max_pf, 10), dtype=dtype, order="F")
            X[:, :, 0] = pt
            X[:, :, 1] = px
            X[:, :, 2] = py
            X[:, :, 3] = eta
            X[:, :, 4] = phi
            X[:, :, 5] = puppiw
            X[:, :, 6] = enc_pdg
            X[:, :, 7] = enc_chg
            X[:, :, 8] = hcalDepth

            if include_mc and var_list_mc:
                gen_pt = arrays["genMet_pt"].to_numpy()
                gen_phi = arrays["genMet_phi"].to_numpy()
                Y = np.stack([gen_pt * np.cos(gen_phi), gen_pt * np.sin(gen_phi)], axis=1).astype(dtype, copy=False)
            else:
                Y = np.zeros((nevents, 2), dtype=dtype)

            X_parts.append(X)
            Y_parts.append(Y)
            logger.info(f"Batch {batch_count} complete: {nevents} events processed")
        
        # TODO: don't hardcode the number of features
        features = np.concatenate(X_parts, axis=0) if X_parts else np.zeros((0, max_pf, 9), dtype=dtype)
        targets = np.concatenate(Y_parts, axis=0) if Y_parts else np.zeros((0, 2), dtype=dtype)
        results[sample] = (features, targets)
        
        logger.info(f"Sample '{sample}' complete: {features.shape[0]} total events, shape {features.shape}")

    logger.info(f"Data loading complete. Processed {len(results)} samples")
    return results


def load_samples_to_numpy_extended(
    data_root: Union[str, Path],
    sample_names: List[str],
    feature_layout: List[str],
    var_list_mc: Optional[List[str]] = None,
    event_feature_layout: Optional[List[str]] = None,
    *,
    max_pf: int,
    encoding: Dict[str, Dict[float, int]],
    include_mc: bool = True,
    step_size: str = "100 MB",
    file_pattern: str = "*.root",
    tree_name: str = "Events",
    dtype=np.float32,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extended preprocessor: load per-candidate AND per-event branches.

    Returns ``{sample_name: (features, event_features, targets)}`` where:
      - ``features``       has shape (n_events, max_pf, len(feature_layout))
      - ``event_features`` has shape (n_events, len(event_feature_layout))
        — shape (n_events, 0) when ``event_feature_layout`` is None/empty.
      - ``targets``        has shape (n_events, 2) — gen MET (px, py).

    The per-candidate ``feature_layout`` is a declarative ordered list of
    feature names (vs the hardcoded 9-feature legacy path). Each slot is
    either a direct branch load or a derived feature.

    Supported candidate features (controlled by the L1DeepMET extended recipe):

      Direct branches (loaded as-is from L1PuppiCands_<name>; the full
      name → (branch, pad) contract is EXTENDED_DIRECT_BRANCHES):
        pt, eta, phi, puppiWeight, dxy, dxyErr, mass, z0,
        hwPt, hwEta, hwPhi, hwPuppiWeight, hwQual,
        trackChi2RPhi, trackChi2RZ, trackChi2Bend, trackNStubs,
        trackMvaQual, caloEta, caloPhi, clPuId, clEmId, clPt, clEmEt,
        and the extended-v1 additions (26Sep2 production onward):
        hwZ0, hwDxy, hwTkQuality, hwEmID, clEta, clPhi, clSigmaRR,
        clAbsZBary, clHoE, clPtError, trackPtError, trackHitPattern

      Derived:
        px         = pt × cos(phi)
        py         = pt × sin(phi)
        encoded_pdgId  (via `encoding["L1PuppiCands_pdgId"]`)
        encoded_charge (via `encoding["L1PuppiCands_charge"]`)

    Supported event features: see EVENT_FEATURE_BRANCHES (module-level dict).

    Missing branches are pad-filled with the column's declared sentinel (see _safe_get).
    Pad value for each direct branch is 0.0 unless special-cased (pdgId/charge get
    -999.0 before encoding; dxyErr gets 1000.0 to match the legacy convention).
    """
    data_root = Path(data_root)
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Feature-name → (branch, pad) map; hoisted to module level (see
    # EXTENDED_DIRECT_BRANCHES) so the layout↔loader contract is testable.
    DIRECT_BRANCHES = EXTENDED_DIRECT_BRANCHES
    DERIVED = {"px", "py", "encoded_pdgId", "encoded_charge", "hcal_depth"}

    # Wanted branches = whatever the layout needs + the always-needed ones for
    # derived features (pdgId, charge for encoding; phi for px/py)
    wanted: List[str] = []
    for feat in feature_layout:
        if feat in DIRECT_BRANCHES:
            wanted.append(DIRECT_BRANCHES[feat][0])
    # Branches needed for derived features:
    derived_in_layout = [f for f in feature_layout if f in DERIVED]
    if any(d in derived_in_layout for d in ("px", "py")):
        wanted += ["L1PuppiCands_pt", "L1PuppiCands_phi"]
    if "encoded_pdgId" in derived_in_layout:
        wanted.append("L1PuppiCands_pdgId")
    if "encoded_charge" in derived_in_layout:
        wanted.append("L1PuppiCands_charge")
    if "hcal_depth" in derived_in_layout:
        wanted += [
            "HGCal3DCl_firstHcal1layers",
            "HGCal3DCl_firstHcal3layers",
            "HGCal3DCl_firstHcal5layers",
        ]
    # Event-level branches (from EVENT_FEATURE_BRANCHES) — unknown event
    # features raise here rather than later for clearer error messages.
    event_layout = list(event_feature_layout or [])
    for feat in event_layout:
        if feat not in EVENT_FEATURE_BRANCHES:
            raise ValueError(
                f"Unknown event feature: {feat!r}. "
                f"Add it to EVENT_FEATURE_BRANCHES in preprocessing.py."
            )
        wanted.append(EVENT_FEATURE_BRANCHES[feat][0])
    # de-dup, preserve order
    seen: set[str] = set()
    wanted = [b for b in wanted if not (b in seen or seen.add(b))]
    if include_mc and var_list_mc:
        wanted += list(var_list_mc)

    n_features = len(feature_layout)
    n_event_features = len(event_layout)
    logger.info(f"Extended preprocessor: {n_features} per-candidate features from layout {feature_layout}")
    if n_event_features:
        logger.info(f"Extended preprocessor: {n_event_features} event-level features from layout {event_layout}")

    for i, sample in enumerate(sample_names, 1):
        logger.info(f"[{i}/{len(sample_names)}] Processing sample: {sample}")
        files = sorted((data_root / sample).rglob(file_pattern))
        # Skip 0-byte ROOT files (e.g. from failed/held production jobs) — uproot.iterate
        # raises OSError ("received 0 bytes") on an empty file and aborts the whole sample.
        _n_all = len(files)
        files = [f for f in files if f.stat().st_size > 0]
        if len(files) < _n_all:
            logger.warning(f"  {sample}: skipped {_n_all - len(files)} empty/0-byte ROOT file(s)")
        if not files:
            raise FileNotFoundError(f"No ROOT files for sample '{sample}' under {data_root / sample}")
        files_with_tree = [str(f) + f":{tree_name}" for f in files]

        # Filter wanted-list down to branches that actually exist in the file
        branches = _existing_branches(files_with_tree, wanted)

        X_parts: List[np.ndarray] = []
        EX_parts: List[np.ndarray] = []
        Y_parts: List[np.ndarray] = []
        ref_branch = "L1PuppiCands_pt"  # always present; used for zero-fallback shape

        for arrays in uproot.iterate(files_with_tree, expressions=branches,
                                     step_size=step_size, library="ak"):
            # Reference shape from pt
            pt = to_np_array(arrays["L1PuppiCands_pt"], maxN=max_pf, pad=0.0)
            phi = to_np_array(arrays["L1PuppiCands_phi"], maxN=max_pf, pad=0.0) \
                if "L1PuppiCands_phi" in getattr(arrays, "fields", ()) else np.zeros_like(pt)
            nevents = pt.shape[0]

            # Derived features
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            enc_pdg = enc_chg = hcal_depth = None
            if "encoded_pdgId" in feature_layout:
                pdgid = _safe_get(arrays, "L1PuppiCands_pdgId", max_pf, -999.0, ref_branch)
                enc_pdg = np.vectorize(encoding["L1PuppiCands_pdgId"].__getitem__)(
                    pdgid.astype(float)
                )
            if "encoded_charge" in feature_layout:
                chg = _safe_get(arrays, "L1PuppiCands_charge", max_pf, -999.0, ref_branch)
                enc_chg = np.vectorize(encoding["L1PuppiCands_charge"].__getitem__)(
                    chg.astype(float)
                )
            if "hcal_depth" in feature_layout:
                # NB: HGCal3DCl_* branches are PER-CLUSTER, not per-candidate.
                # The legacy code indexes them by candidate-index, which produces
                # garbage. We zero-fill this feature here as a defensive fix —
                # see reports/ntuple_pipeline_plan/PLAN.md §1.1 and P1.7.
                hcal_depth = np.zeros_like(pt)

            # Build output array slot by slot
            X = np.zeros((nevents, max_pf, n_features), dtype=dtype, order="F")
            for j, feat in enumerate(feature_layout):
                if feat in DIRECT_BRANCHES:
                    branch, pad = DIRECT_BRANCHES[feat]
                    X[:, :, j] = _safe_get(arrays, branch, max_pf, pad, ref_branch)
                elif feat == "px":
                    X[:, :, j] = px
                elif feat == "py":
                    X[:, :, j] = py
                elif feat == "encoded_pdgId":
                    X[:, :, j] = enc_pdg
                elif feat == "encoded_charge":
                    X[:, :, j] = enc_chg
                elif feat == "hcal_depth":
                    X[:, :, j] = hcal_depth
                else:
                    raise ValueError(f"Unknown feature in feature_layout: {feat!r}")

            if include_mc and var_list_mc:
                gen_pt = arrays["genMet_pt"].to_numpy()
                gen_phi = arrays["genMet_phi"].to_numpy()
                Y = np.stack([gen_pt * np.cos(gen_phi), gen_pt * np.sin(gen_phi)],
                             axis=1).astype(dtype, copy=False)
            else:
                Y = np.zeros((nevents, 2), dtype=dtype)

            # Event-level features (may be empty if event_layout is empty)
            EX = event_features_from_arrays(arrays, event_layout, nevents, dtype=dtype)

            X_parts.append(X)
            EX_parts.append(EX)
            Y_parts.append(Y)

        features = np.concatenate(X_parts, axis=0) if X_parts else np.zeros((0, max_pf, n_features), dtype=dtype)
        event_features = np.concatenate(EX_parts, axis=0) if EX_parts else np.zeros((0, n_event_features), dtype=dtype)
        targets = np.concatenate(Y_parts, axis=0) if Y_parts else np.zeros((0, 2), dtype=dtype)
        results[sample] = (features, event_features, targets)
        logger.info(
            f"Sample '{sample}' complete: {features.shape[0]} events, "
            f"candidate shape {features.shape}, event-feature shape {event_features.shape}"
        )

    return results


def select_events(results: Dict[str, Tuple[np.ndarray, ...]],
                  samples: Dict[str, int]) -> Dict[str, Tuple[np.ndarray, ...]]:
    """Select desired number of events per sample.

    Tuple-length-agnostic: works for the legacy (features, targets) 2-tuple
    AND the extended (features, event_features, targets) 3-tuple alike.
    All arrays in the tuple are sliced with the same event indices.
    """
    logger.info("Selecting desired number of events per sample")

    selected_results: Dict[str, Tuple[np.ndarray, ...]] = {}
    for sample_name, payload in results.items():
        n_desired = samples[sample_name]
        n_available = payload[0].shape[0]

        if n_available >= n_desired:
            indices = np.random.choice(n_available, size=n_desired, replace=False)
            selected = tuple(arr[indices] for arr in payload)
            logger.info(f"{sample_name}: Selected {n_desired} events from {n_available} available")
        else:
            selected = payload
            logger.warning(f"{sample_name}: Only {n_available} events available, requested {n_desired}")

        selected_results[sample_name] = selected
        logger.info(f"{sample_name} feature shape: {selected[0].shape}")
    return selected_results


def preprocess_data(selected_results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply preprocessing steps to the selected data, based on preprocessing.py::preProcessing
    but without normalization factor. Returns single concatenated feature array.
    
    Steps:
    - Extract pt, px, py, eta, phi, puppi, dxyErr, hcalDepth
    - Remove outliers in pt/px/py and sanitize dxyErr/hcalDepth
    - Concatenate into single feature array for H5 storage
    
    Feature array layout (9 features):
    [:, :, 0] = pt, [:, :, 1] = eta, [:, :, 2] = phi, [:, :, 3] = puppi,
    [:, :, 4] = hcal_depth, [:, :, 5] = px, [:, :, 6] = py,
    [:, :, 7] = encoded_pdgId, [:, :, 8] = encoded_charge
    
    Returns:
    - processed_results: dict with sample_name -> (preprocessed_features, targets)
    """
    logger.info("Applying preprocessing steps to selected data")
    
    processed_results = {}
    
    for sample_name, (features, targets) in selected_results.items():
        logger.info(f"Preprocessing sample: {sample_name}")
        
        A = features  
        
        pt = A[:, :, 0:1]          
        px = A[:, :, 1:2]          
        py = A[:, :, 2:3]          
        eta = A[:, :, 3:4]         
        phi = A[:, :, 4:5]         
        puppi = A[:, :, 5:6]       
        dxyErr = A[:, :, 8:9]      
        hcalDepth = A[:, :, 9:10]  
        
        # Remove outliers in momentum (using 500 GeV cutoff, no normalization)
        pt[np.where(np.abs(pt) > 500.0)] = 0.0
        px[np.where(np.abs(px) > 500.0)] = 0.0  
        py[np.where(np.abs(py) > 500.0)] = 0.0
        
        # Sanitize dxyErr and hcalDepth
        dxyErr[np.where(dxyErr == -999)] = 0.0
        dxyErr[np.where(np.abs(dxyErr) > 100.0)] = 0.0
        hcalDepth[np.where(~np.isfinite(hcalDepth))] = 0.0
        hcalDepth[np.where(np.abs(hcalDepth) > 100.0)] = 0.0
        
        # Create single preprocessed feature array (gets split later in training)
        # Layout: [pt, eta, phi, puppi, hcal_depth, px, py, encoded_pdgId, encoded_charge]
        inputs_cat0_expanded = A[:, :, 6:7]  # encoded pdgId 
        inputs_cat1_expanded = A[:, :, 7:8]  # encoded charge 
        
        preprocessed_features = np.concatenate([
            pt,                    # [:, :, 0] 
            eta,                   # [:, :, 1]
            phi,                   # [:, :, 2] 
            puppi,                 # [:, :, 3]
            hcalDepth,             # [:, :, 4]
            px,                    # [:, :, 5]
            py,                    # [:, :, 6]
            inputs_cat0_expanded,  # [:, :, 7] 
            inputs_cat1_expanded   # [:, :, 8]
        ], axis=2)
        
        processed_results[sample_name] = (preprocessed_features, targets)
        
        logger.info(f"  {sample_name} processed shapes:")
        logger.info(f"    features: {preprocessed_features.shape} (all preprocessed features)")
        logger.info(f"    targets: {targets.shape}")
    
    logger.info("Preprocessing complete")
    return processed_results


def combine_shuffle_split(processed_results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                         data_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine samples and create train/val/test splits.
    
    Args:
        processed_results: Dict with sample_name -> (features, targets)
        data_cfg: Data configuration dict with train-val-test-split ratios
        
    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    logger.info("Combining samples and creating train/val/test splits")

    # Combine all samples
    all_features = []
    all_targets = []
    for sample_name, (features, targets) in processed_results.items():
        logger.info(f"Adding {sample_name}: {features.shape[0]} events")
        all_features.append(features)
        all_targets.append(targets)

    # Concatenate and shuffle
    X_combined = np.concatenate(all_features, axis=0)
    Y_combined = np.concatenate(all_targets, axis=0)
    logger.info(f"Combined dataset: {X_combined.shape[0]} total events")

    # Shuffle indices
    np.random.seed(42)  # For reproducibility
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)
    X_combined = X_combined[indices]
    Y_combined = Y_combined[indices]

    # Get split ratios from config
    split_config = data_cfg["train-val-test-split"]
    train_ratio = split_config["train"]
    val_ratio = split_config["val"]
    test_ratio = split_config["test"]

    # Calculate split indices
    train_split = int(train_ratio * len(X_combined))
    val_split = int((train_ratio + val_ratio) * len(X_combined))

    # Create splits
    X_train, Y_train = X_combined[:train_split], Y_combined[:train_split]
    X_val, Y_val = X_combined[train_split:val_split], Y_combined[train_split:val_split]
    X_test, Y_test = X_combined[val_split:], Y_combined[val_split:]

    logger.info(f"Train split: {X_train.shape[0]} events ({train_ratio:.1%})")
    logger.info(f"Val split: {X_val.shape[0]} events ({val_ratio:.1%})")
    logger.info(f"Test split: {X_test.shape[0]} events ({test_ratio:.1%})")

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def combine_shuffle_split_extended(
    processed_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    data_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Variant of combine_shuffle_split for the extended preprocessor.

    Input:  {sample: (features, event_features, targets)}  — 3-tuple per sample.
    Output: (X_train, X_val, X_test,
             EX_train, EX_val, EX_test,
             Y_train, Y_val, Y_test)

    Same shuffle seed (42) and split ratios as combine_shuffle_split; all three
    arrays for a given event share the same shuffled index so per-candidate,
    per-event, and target rows stay aligned.
    """
    logger.info("Combining samples and creating train/val/test splits (extended path)")

    all_features: List[np.ndarray] = []
    all_event_features: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    for sample_name, (features, event_features, targets) in processed_results.items():
        logger.info(f"Adding {sample_name}: {features.shape[0]} events")
        all_features.append(features)
        all_event_features.append(event_features)
        all_targets.append(targets)

    X_combined = np.concatenate(all_features, axis=0)
    EX_combined = np.concatenate(all_event_features, axis=0)
    Y_combined = np.concatenate(all_targets, axis=0)
    logger.info(
        f"Combined dataset: {X_combined.shape[0]} total events, "
        f"X {X_combined.shape}, EX {EX_combined.shape}, Y {Y_combined.shape}"
    )

    np.random.seed(42)
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)
    X_combined = X_combined[indices]
    EX_combined = EX_combined[indices]
    Y_combined = Y_combined[indices]

    split_config = data_cfg["train-val-test-split"]
    train_ratio = split_config["train"]
    val_ratio = split_config["val"]
    test_ratio = split_config["test"]

    train_split = int(train_ratio * len(X_combined))
    val_split = int((train_ratio + val_ratio) * len(X_combined))

    X_train,  Y_train,  EX_train = X_combined[:train_split],          Y_combined[:train_split],          EX_combined[:train_split]
    X_val,    Y_val,    EX_val   = X_combined[train_split:val_split], Y_combined[train_split:val_split], EX_combined[train_split:val_split]
    X_test,   Y_test,   EX_test  = X_combined[val_split:],            Y_combined[val_split:],            EX_combined[val_split:]

    logger.info(f"Train split: {X_train.shape[0]} events ({train_ratio:.1%})")
    logger.info(f"Val split:   {X_val.shape[0]} events ({val_ratio:.1%})")
    logger.info(f"Test split:  {X_test.shape[0]} events ({test_ratio:.1%})")

    return X_train, X_val, X_test, EX_train, EX_val, EX_test, Y_train, Y_val, Y_test


def save_h5_files(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                  Y_train: np.ndarray, Y_val: np.ndarray, Y_test: np.ndarray,
                  output_dir: Union[str, Path], samples: Dict[str, int],
                  feature_layout: Optional[List[str]] = None,
                  *,
                  event_features_train: Optional[np.ndarray] = None,
                  event_features_val: Optional[np.ndarray] = None,
                  event_features_test: Optional[np.ndarray] = None,
                  event_feature_layout: Optional[List[str]] = None) -> None:
    """Save preprocessed data to train/val/test H5 files with metadata.

    Args:
        X_*, Y_*: feature/target arrays
        output_dir: directory to save files
        samples: sample configuration dict (event counts) for metadata
        feature_layout: optional ordered list of feature names. If None,
            falls back to the legacy 9-feature layout. The extended
            preprocessor passes its own layout from
            params.yaml::preprocess.feature_layout_extended.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving preprocessed data to separate H5 files in {output_dir}")

    layout = feature_layout if feature_layout is not None else [
        'pt', 'eta', 'phi', 'puppi_weight', 'hcal_depth',
        'px', 'py', 'encoded_pdgId', 'encoded_charge',
    ]
    if len(layout) != X_train.shape[2]:
        logger.warning(
            f"feature_layout length ({len(layout)}) doesn't match X last-axis "
            f"({X_train.shape[2]}). H5 metadata will be inconsistent."
        )
    # Metadata to save in each file
    metadata: Dict[str, Any] = {
        'feature_layout': layout,
        'n_features': X_train.shape[2],
        'max_puppi_candidates': X_train.shape[1],
        'samples_used': list(samples.keys()),
        'sample_event_counts': list(samples.values())
    }
    # Event-level metadata (only when the extended preprocessor passes
    # event_features through). Adding even an empty layout would create
    # a confusing zero-width event_features dataset, so guard on layout
    # length AND the train array being non-None.
    has_event_features = (
        event_feature_layout is not None
        and len(event_feature_layout) > 0
        and event_features_train is not None
    )
    if has_event_features:
        metadata['event_feature_layout'] = list(event_feature_layout)
        metadata['n_event_features'] = event_features_train.shape[1]
        if event_features_train.shape[1] != len(event_feature_layout):
            logger.warning(
                f"event_feature_layout length ({len(event_feature_layout)}) doesn't "
                f"match EX last-axis ({event_features_train.shape[1]}). H5 metadata "
                f"will be inconsistent."
            )

    def _write_split(path: Path, X: np.ndarray, Y: np.ndarray,
                     EX: Optional[np.ndarray], split_label: str) -> None:
        # lzf + C-contiguous, NOT gzip-9: the extended loader builds X with
        # order="F", and gzip-9 on a strided multi-GB array writes at
        # ~0.5 MB/s (hours for a full campaign). lzf writes at disk speed
        # for ~1.5x the size; matches preprocessing_legacy's convention.
        with h5py.File(path, 'w') as f:
            f.create_dataset('features', data=np.ascontiguousarray(X),
                             compression='lzf')
            f.create_dataset('targets', data=np.ascontiguousarray(Y),
                             compression='lzf')
            if has_event_features and EX is not None:
                f.create_dataset('event_features', data=np.ascontiguousarray(EX),
                                 compression='lzf')
            for key, value in metadata.items():
                f.attrs[key] = value
        logger.info(f"{split_label} data saved to {path}")
        size_mb = path.stat().st_size / 1024**2
        if has_event_features and EX is not None:
            logger.info(
                f"  Shape: features {X.shape}, event_features {EX.shape}, targets {Y.shape}"
            )
        else:
            logger.info(f"  Shape: features {X.shape}, targets {Y.shape}")
        logger.info(f"  File size: {size_mb:.1f} MB")

    train_file_path = output_dir / "train.h5"
    val_file_path   = output_dir / "val.h5"
    test_file_path  = output_dir / "test.h5"
    _write_split(train_file_path, X_train, Y_train, event_features_train, "Training")
    _write_split(val_file_path,   X_val,   Y_val,   event_features_val,   "Validation")
    _write_split(test_file_path,  X_test,  Y_test,  event_features_test,  "Test")

    # Summary
    total_size = (train_file_path.stat().st_size + val_file_path.stat().st_size + 
                  test_file_path.stat().st_size) / 1024**2
    logger.info(f"All H5 files saved successfully")
    logger.info(f"Total size: {total_size:.1f} MB")



def coerce_encoding(encoding_raw: Dict[str, Dict]) -> Dict[str, Dict[float, int]]:
    """Coerce YAML keys to numeric for robust mapping."""
    return {k: {float(kk): int(vv) for kk, vv in v.items()} for k, v in encoding_raw.items()}