"""Dataset card — machine-readable JSON + human-readable Markdown sidecar.

Every preprocessed dataset under ``outputs/preprocessed/<tag>/`` should have a
``dataset_card.json`` + ``dataset_card.md`` next to its H5 files (see the
"Dataset card (STANDING ORDER)" section in CLAUDE.md). The card lets
downstream tooling answer "what's in this dataset, where did it come from,
what assumptions are baked in" without re-deriving from the pipeline.

Layout (schema v1):

  {
    "schema_version": "1",
    "tag": "...",
    "created_at": "...",
    "provenance": {
      "git_sha": "...",
      "preprocess_args": { ... },
      "source_data_root": "...",
      ...
    },
    "composition": {
      "total_events": N,
      "splits": {"train": {...}, "val": {...}, "test": {...}},
      "per_sample_loaded": {sample: count},     # what's actually in the H5
      "per_sample_requested": {sample: count},  # what params.yaml asked for
      ...
    },
    "schema": {
      "feature_layout": [...],
      "event_feature_layout": [...],
      "features_shape": [N, 128, K],
      "event_features_shape": [N, M] or None,
      "targets_shape": [N, 2],
    },
    "per_candidate_feature_stats": { feature_name: { ... } },
    "per_event_feature_stats":     { feature_name: { ... } },
    "target_stats": { ... },
    "warnings": [strings]
  }

Per-candidate feature stats distinguish "real candidates" (pt > 0) from
pad zeros — without that, a sparse feature column's median/MAD is pulled
to zero by the padding. The same `pt > 0` mask is what the training-time
robust standardiser in the event-count experiment uses.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


SCHEMA_VERSION = "1"


def _safe_float(x) -> float | None:
    """Convert numpy scalar to native float, return None for NaN/Inf."""
    if x is None:
        return None
    f = float(x)
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def compute_per_feature_stats(
    X: np.ndarray,
    feature_layout: List[str],
    pt_slot: int = 0,
    real_warning_thresholds: Dict[str, float] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-candidate feature stats with real-vs-pad separation.

    For each feature ``f`` in ``feature_layout``, computes (on the train
    split, ``X`` shape ``(N_events, N_candidates, K_features)``):

      n_real_candidates       — count of slots with pt > 0 (the candidate exists)
      frac_real_candidate     — n_real_candidates / total slots
      min_real, max_real      — over real candidates only
      median_real, mad_real   — robust location/scale over real candidates
      mean_real, std_real     — Gaussian location/scale over real candidates
      frac_nan, frac_inf      — over ALL slots (sanitization sanity check)
      warnings                — list of human-readable strings; warnings fire on
                                NaN > 0, Inf > 0, all-zero on real candidates.

    The pt mask is computed from slot ``pt_slot`` (default 0, matches
    ``feature_layout_extended`` in params.yaml).
    """
    thresholds = {
        "all_zero": 1.0, "high_nan_frac": 0.001, "high_inf_frac": 0.001,
        "extreme_abs": 1e10,  # |x| > this fires "extreme value" warning
    }
    if real_warning_thresholds:
        thresholds.update(real_warning_thresholds)

    n_events, n_candidates, n_features = X.shape
    pt = X[:, :, pt_slot]
    real_mask = (pt > 0)            # (N_events, N_candidates)
    n_real = int(real_mask.sum())
    n_total = n_events * n_candidates

    out: Dict[str, Dict[str, Any]] = {}
    for j, name in enumerate(feature_layout):
        col = X[:, :, j]
        # Promote to float64 for the reductions so e.g. a single 1e38
        # entry doesn't overflow the per-column sum and corrupt the mean.
        col_finite = np.where(np.isfinite(col), col, 0).astype(np.float64)
        frac_nan = float(np.isnan(col).sum() / n_total) if n_total else 0.0
        frac_inf = float(np.isinf(col).sum() / n_total) if n_total else 0.0

        if n_real == 0:
            stats: Dict[str, Any] = {
                "n_real_candidates": 0,
                "frac_real_candidate": 0.0,
                "frac_nan": frac_nan,
                "frac_inf": frac_inf,
                "min_real": None, "max_real": None,
                "median_real": None, "mad_real": None,
                "mean_real": None, "std_real": None,
                "warnings": [],
            }
        else:
            vals_real = col_finite[real_mask]
            med = float(np.median(vals_real))
            mad = float(np.median(np.abs(vals_real - med)))
            stats = {
                "n_real_candidates": n_real,
                "frac_real_candidate": n_real / n_total,
                "frac_nan": frac_nan,
                "frac_inf": frac_inf,
                "min_real": _safe_float(vals_real.min()),
                "max_real": _safe_float(vals_real.max()),
                "median_real": med,
                "mad_real": mad,
                "mean_real": _safe_float(vals_real.mean()),
                "std_real": _safe_float(vals_real.std()),
                "warnings": [],
            }

        warnings: List[str] = []
        if frac_nan > thresholds["high_nan_frac"]:
            warnings.append(
                f"NaN fraction {frac_nan:.2%} — check upstream sanitisation"
            )
        if frac_inf > thresholds["high_inf_frac"]:
            warnings.append(
                f"Inf fraction {frac_inf:.2%} — check upstream sanitisation"
            )
        # Extreme-magnitude values on real candidates: not infinite, but big
        # enough that downstream BN/standardisation will misbehave. Caught
        # the `clPuId` / `clEmId` ~±3.4e38 garbage in the 20k preprocess.
        if (n_real > 0 and stats.get("min_real") is not None
                and (abs(stats["min_real"]) > thresholds["extreme_abs"]
                     or abs(stats["max_real"]) > thresholds["extreme_abs"])):
            warnings.append(
                f"extreme magnitude on real candidates: "
                f"min={stats['min_real']:.2e}, max={stats['max_real']:.2e} — "
                f"likely upstream garbage that didn't trip the NaN/Inf check"
            )
        if n_real > 0:
            # All-zero on real candidates is suspicious for most features (a
            # PUPPI weight column shouldn't be all-zero on real candidates,
            # nor should pt). Sentinel-only columns like a placeholder
            # `mass` field can legitimately be all-zero — caller can adjust
            # the threshold via `real_warning_thresholds`.
            if stats["max_real"] == 0.0 and stats["min_real"] == 0.0:
                warnings.append(
                    "all-zero on real candidates — feature carries no info"
                )
        stats["warnings"] = warnings
        out[name] = stats
    return out


def compute_per_event_feature_stats(
    EX: np.ndarray,
    event_feature_layout: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Per-event feature stats: simpler than per-candidate since there's
    no pt-mask to apply — every event has an event-level scalar.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if EX is None or EX.size == 0:
        return out
    for j, name in enumerate(event_feature_layout):
        col = EX[:, j]
        frac_zero = float((col == 0).sum() / max(col.size, 1))
        warnings: List[str] = []
        if frac_zero == 1.0:
            warnings.append(
                "all-zero — feature was likely missing from input ntuples and "
                "zero-filled by the loader"
            )
        out[name] = {
            "mean": _safe_float(col.mean()),
            "std":  _safe_float(col.std()),
            "min":  _safe_float(col.min()),
            "max":  _safe_float(col.max()),
            "frac_zero": frac_zero,
            "warnings": warnings,
        }
    return out


def compute_target_stats(
    Y: np.ndarray,
    pt_bins: tuple = (0.0, 50.0, 100.0, 200.0, 300.0, 400.0, float("inf")),
) -> Dict[str, Any]:
    """Gen-MET target distribution + per-pT-bin counts."""
    gen_pt = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)
    gen_phi = np.arctan2(Y[:, 1], Y[:, 0])
    bin_counts: Dict[str, int] = {}
    for i in range(len(pt_bins) - 1):
        lo, hi = pt_bins[i], pt_bins[i + 1]
        label = f"{int(lo)}_{'inf' if not np.isfinite(hi) else int(hi)}"
        bin_counts[label] = int(((gen_pt >= lo) & (gen_pt < hi)).sum())
    return {
        "gen_met_pt": {
            "mean": _safe_float(gen_pt.mean()),
            "std":  _safe_float(gen_pt.std()),
            "min":  _safe_float(gen_pt.min()),
            "max":  _safe_float(gen_pt.max()),
        },
        "gen_met_phi": {
            "mean": _safe_float(gen_phi.mean()),
            "std":  _safe_float(gen_phi.std()),
        },
        "per_pt_bin_counts": bin_counts,
    }


def build_dataset_card(
    output_dir: Path,
    *,
    tag: str,
    splits: Dict[str, Dict[str, np.ndarray]],   # {"train": {"X": ..., "EX": ..., "Y": ...}, ...}
    feature_layout: List[str],
    event_feature_layout: Optional[List[str]],
    per_sample_loaded: Dict[str, int],
    per_sample_requested: Dict[str, int],
    provenance: Dict[str, Any],
    pt_slot: int = 0,
) -> Dict[str, Any]:
    """Build the full dataset card and write it to ``dataset_card.json`` +
    ``dataset_card.md`` under ``output_dir``. Returns the card dict.

    ``splits`` keys are split names; each value is a dict with ``"X"``,
    optionally ``"EX"``, and ``"Y"``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-feature stats are computed on the TRAIN split (val/test stats would
    # be peeking at held-out data). If train is missing, error out — that's
    # a bug, not a config option.
    if "train" not in splits or "X" not in splits["train"]:
        raise ValueError("splits['train']['X'] is required for per-feature stats")
    X_train = splits["train"]["X"]
    Y_train = splits["train"].get("Y")
    EX_train = splits["train"].get("EX")

    per_cand = compute_per_feature_stats(X_train, feature_layout, pt_slot=pt_slot)
    per_event = (
        compute_per_event_feature_stats(EX_train, event_feature_layout)
        if (event_feature_layout and EX_train is not None and EX_train.size > 0)
        else {}
    )
    target_stats = (
        compute_target_stats(Y_train) if Y_train is not None and Y_train.size > 0 else {}
    )

    splits_meta: Dict[str, Dict[str, Any]] = {}
    total_events = 0
    for name, payload in splits.items():
        n = int(payload["X"].shape[0]) if "X" in payload else 0
        splits_meta[name] = {"n_events": n}
        total_events += n
    # Add fraction-of-total
    for name, s in splits_meta.items():
        s["fraction"] = s["n_events"] / max(total_events, 1)

    # Aggregate top-level warnings (per-feature warnings remain visible
    # under their feature too, but a top-level summary is what reviewers
    # see first).
    top_warnings: List[str] = []
    for name, s in per_cand.items():
        for w in s.get("warnings", []):
            top_warnings.append(f"feature '{name}': {w}")
    for name, s in per_event.items():
        for w in s.get("warnings", []):
            top_warnings.append(f"event feature '{name}': {w}")
    # Sample-coverage warnings: requested but not delivered, or zero-loaded
    for sample, requested in per_sample_requested.items():
        loaded = per_sample_loaded.get(sample, 0)
        if loaded == 0:
            top_warnings.append(
                f"sample '{sample}': requested {requested}, loaded 0 (not in source data)"
            )
        elif loaded < requested:
            top_warnings.append(
                f"sample '{sample}': requested {requested}, loaded {loaded} "
                f"({loaded/requested:.0%} — capped by source availability)"
            )

    card: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tag": tag,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "provenance": dict(provenance),
        "composition": {
            "total_events": total_events,
            "splits": splits_meta,
            "per_sample_loaded": dict(per_sample_loaded),
            "per_sample_requested": dict(per_sample_requested),
        },
        "schema": {
            "feature_layout": list(feature_layout),
            "event_feature_layout": (list(event_feature_layout)
                                     if event_feature_layout else None),
            "features_shape": list(X_train.shape),
            "event_features_shape": (list(EX_train.shape)
                                     if EX_train is not None and EX_train.size > 0
                                     else None),
            "targets_shape": (list(Y_train.shape)
                              if Y_train is not None else None),
        },
        "per_candidate_feature_stats": per_cand,
        "per_event_feature_stats": per_event,
        "target_stats": target_stats,
        "warnings": top_warnings,
    }

    (output_dir / "dataset_card.json").write_text(json.dumps(card, indent=2))
    (output_dir / "dataset_card.md").write_text(render_markdown(card))
    return card


def render_markdown(card: Dict[str, Any]) -> str:
    """Render the JSON card to a human-readable Markdown summary.

    Designed for eyeballing in a PR review or in the report directory.
    Keep it terse — the JSON is the source of truth, this is a courtesy.
    """
    lines: List[str] = []
    lines.append(f"# Dataset card — `{card['tag']}`")
    lines.append("")
    lines.append(f"*schema_version `{card['schema_version']}` · created `{card['created_at']}`*")
    lines.append("")

    # Top-level warnings get a callout — these are the things a reviewer
    # most wants to see at a glance.
    warnings = card.get("warnings", [])
    if warnings:
        lines.append("## ⚠ Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Provenance
    lines.append("## Provenance")
    prov = card.get("provenance", {})
    if prov:
        for k, v in prov.items():
            if isinstance(v, dict):
                lines.append(f"- **{k}**:")
                for kk, vv in v.items():
                    lines.append(f"  - `{kk}` = `{vv}`")
            else:
                lines.append(f"- **{k}**: `{v}`")
    else:
        lines.append("_(none recorded)_")
    lines.append("")

    # Composition
    comp = card["composition"]
    lines.append("## Composition")
    lines.append(f"- **Total events**: {comp['total_events']:,}")
    lines.append("")
    lines.append("### Splits")
    lines.append("")
    lines.append("| split | n_events | fraction |")
    lines.append("|---|---:|---:|")
    for name, s in comp["splits"].items():
        lines.append(f"| `{name}` | {s['n_events']:,} | {s['fraction']:.1%} |")
    lines.append("")
    lines.append("### Per-sample event counts (loaded vs requested)")
    lines.append("")
    lines.append("| sample | loaded | requested | delivered |")
    lines.append("|---|---:|---:|---:|")
    for sample in sorted(set(comp["per_sample_requested"]) | set(comp["per_sample_loaded"])):
        loaded = comp["per_sample_loaded"].get(sample, 0)
        requested = comp["per_sample_requested"].get(sample, 0)
        ratio = f"{loaded/requested:.0%}" if requested else "—"
        lines.append(f"| `{sample}` | {loaded:,} | {requested:,} | {ratio} |")
    lines.append("")

    # Schema
    sch = card["schema"]
    lines.append("## Schema")
    lines.append(f"- `features` shape: `{sch['features_shape']}`")
    if sch.get("event_features_shape"):
        lines.append(f"- `event_features` shape: `{sch['event_features_shape']}`")
    if sch.get("targets_shape"):
        lines.append(f"- `targets` shape: `{sch['targets_shape']}`")
    lines.append(f"- `feature_layout` ({len(sch['feature_layout'])}): "
                 + ", ".join(f"`{f}`" for f in sch["feature_layout"]))
    if sch.get("event_feature_layout"):
        lines.append(
            f"- `event_feature_layout` ({len(sch['event_feature_layout'])}): "
            + ", ".join(f"`{f}`" for f in sch["event_feature_layout"]))
    lines.append("")

    # Per-candidate feature stats — train split only
    lines.append("## Per-candidate features (train split)")
    lines.append("")
    lines.append("| feature | n_real | %real | median | MAD | min | max | %NaN | %Inf | warnings |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, s in card["per_candidate_feature_stats"].items():
        med = f"{s['median_real']:.2f}" if s['median_real'] is not None else "—"
        mad = f"{s['mad_real']:.2f}" if s['mad_real'] is not None else "—"
        lo = f"{s['min_real']:.2f}" if s['min_real'] is not None else "—"
        hi = f"{s['max_real']:.2f}" if s['max_real'] is not None else "—"
        warns = "; ".join(s.get("warnings", [])) if s.get("warnings") else ""
        lines.append(f"| `{name}` | {s['n_real_candidates']:,} | "
                     f"{s['frac_real_candidate']:.1%} | {med} | {mad} | {lo} | {hi} | "
                     f"{s['frac_nan']:.2%} | {s['frac_inf']:.2%} | {warns} |")
    lines.append("")

    # Per-event feature stats
    if card.get("per_event_feature_stats"):
        lines.append("## Per-event features (train split)")
        lines.append("")
        lines.append("| feature | mean | std | min | max | %zero | warnings |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for name, s in card["per_event_feature_stats"].items():
            mean = f"{s['mean']:.2f}" if s['mean'] is not None else "—"
            std  = f"{s['std']:.2f}"  if s['std']  is not None else "—"
            lo   = f"{s['min']:.2f}"  if s['min']  is not None else "—"
            hi   = f"{s['max']:.2f}"  if s['max']  is not None else "—"
            warns = "; ".join(s.get("warnings", [])) if s.get("warnings") else ""
            lines.append(f"| `{name}` | {mean} | {std} | {lo} | {hi} | "
                         f"{s['frac_zero']:.2%} | {warns} |")
        lines.append("")

    # Target stats
    if card.get("target_stats"):
        ts = card["target_stats"]
        lines.append("## Targets (gen MET, train split)")
        gm = ts.get("gen_met_pt", {})
        if gm:
            lines.append(
                f"- gen_MET pT: mean={gm.get('mean', 0):.1f} GeV, "
                f"std={gm.get('std', 0):.1f}, min={gm.get('min', 0):.1f}, "
                f"max={gm.get('max', 0):.1f}"
            )
        bc = ts.get("per_pt_bin_counts", {})
        if bc:
            lines.append("")
            lines.append("| gen_MET bin (GeV) | events |")
            lines.append("|---|---:|")
            for label, n in bc.items():
                lines.append(f"| `{label}` | {n:,} |")
        lines.append("")

    return "\n".join(lines)
