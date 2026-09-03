"""Tests for the dataset-card builder.

The card is a machine-readable JSON + human-readable Markdown sidecar
written alongside each preprocessed H5 dataset. Test contract:

- The builder takes the actual numpy splits (so per-feature stats are
  computed on real data, not just declared in metadata).
- It writes BOTH dataset_card.json and dataset_card.md.
- Stats distinguish real candidates (pt > 0) from pad zeros so the
  median/MAD of a sparse feature column isn't pulled to zero.
- Sanity warnings fire on common data-quality issues (NaN, inf,
  all-zero features) so they're visible without grepping the raw H5.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from l1deepmet.data.dataset_card import (
    build_dataset_card,
    compute_per_feature_stats,
    render_markdown,
)


# ─── compute_per_feature_stats ─────────────────────────────────────────────


def test_per_feature_stats_uses_pt_mask_to_separate_real_from_pad():
    """Stats on a real-candidate feature should ignore pad zeros."""
    # 5 events, 4 candidate slots, 3 features. Only first 2 candidates of
    # event 0 and first 3 of event 1 are "real" (pt > 0).
    X = np.zeros((5, 4, 3), dtype=np.float32)
    X[0, 0, 0] = 10.0
    X[0, 1, 0] = 20.0
    X[1, 0, 0] = 5.0
    X[1, 1, 0] = 15.0
    X[1, 2, 0] = 25.0
    # Set eta values for the real candidates only
    X[0, 0, 1] = 1.0
    X[0, 1, 1] = -1.0
    X[1, 0, 1] = 2.0
    X[1, 1, 1] = 0.5
    X[1, 2, 1] = -0.5

    stats = compute_per_feature_stats(X, feature_layout=["pt", "eta", "phi"])
    # Should have 5 real candidates out of 20 total slots
    assert stats["pt"]["n_real_candidates"] == 5
    assert stats["pt"]["frac_real_candidate"] == pytest.approx(5 / 20)
    assert stats["pt"]["max_real"] == pytest.approx(25.0)
    assert stats["pt"]["min_real"] == pytest.approx(5.0)
    # eta should be median over only the 5 real candidates
    assert stats["eta"]["median_real"] == pytest.approx(0.5)
    # phi is all-zero on real candidates → flagged
    assert stats["phi"]["median_real"] == pytest.approx(0.0)
    assert any("zero" in w.lower() for w in stats["phi"]["warnings"])


def test_per_feature_stats_warns_on_nan():
    X = np.zeros((5, 4, 3), dtype=np.float32)
    X[0, 0, 0] = 1.0   # one real candidate so pt mask isn't empty
    X[0, 0, 1] = np.nan
    stats = compute_per_feature_stats(X, feature_layout=["pt", "eta", "phi"])
    assert stats["eta"]["frac_nan"] > 0
    assert any("nan" in w.lower() for w in stats["eta"]["warnings"])


def test_per_feature_stats_warns_on_inf():
    X = np.zeros((5, 4, 3), dtype=np.float32)
    X[0, 0, 0] = 1.0
    X[0, 0, 2] = np.inf
    stats = compute_per_feature_stats(X, feature_layout=["pt", "eta", "phi"])
    assert stats["phi"]["frac_inf"] > 0
    assert any("inf" in w.lower() for w in stats["phi"]["warnings"])


def test_per_feature_stats_empty_real_candidates():
    """If no candidate has pt > 0 anywhere, the stats are well-defined."""
    X = np.zeros((5, 4, 3), dtype=np.float32)
    stats = compute_per_feature_stats(X, feature_layout=["pt", "eta", "phi"])
    assert stats["pt"]["n_real_candidates"] == 0
    assert stats["pt"]["median_real"] is None
    assert stats["pt"]["max_real"] is None


# ─── build_dataset_card ────────────────────────────────────────────────────


def _toy_splits(n_train: int = 100, n_val: int = 10, n_test: int = 10):
    """Tiny well-formed splits suitable for the smoke test."""
    rng = np.random.default_rng(0)
    def mk(n):
        X = rng.uniform(0.1, 5.0, size=(n, 8, 4)).astype(np.float32)
        EX = rng.uniform(0, 100, size=(n, 3)).astype(np.float32)
        Y = rng.uniform(-100, 100, size=(n, 2)).astype(np.float32)
        return {"X": X, "EX": EX, "Y": Y}
    return {"train": mk(n_train), "val": mk(n_val), "test": mk(n_test)}


def test_build_dataset_card_writes_json_and_md(tmp_path):
    splits = _toy_splits()
    card = build_dataset_card(
        output_dir=tmp_path,
        tag="test_tag",
        splits=splits,
        feature_layout=["pt", "eta", "phi", "puppi_weight"],
        event_feature_layout=["puppi_met_pt", "puppi_met_phi", "n_vtx"],
        per_sample_loaded={"TT_PU200": 60, "VBFHToInvisible_PU200": 60},
        per_sample_requested={"TT_PU200": 200_000, "VBFHToInvisible_PU200": 80_000},
        provenance={"git_sha": "abc1234", "preprocess_args": {"--tag": "test_tag"}},
    )
    assert (tmp_path / "dataset_card.json").exists()
    assert (tmp_path / "dataset_card.md").exists()
    assert card["tag"] == "test_tag"
    assert card["schema_version"] == "1"
    # Composition matches split sizes
    comp = card["composition"]
    assert comp["splits"]["train"]["n_events"] == 100
    assert comp["splits"]["val"]["n_events"] == 10
    assert comp["splits"]["test"]["n_events"] == 10
    assert comp["total_events"] == 120
    # Per-sample loaded vs requested visible
    assert comp["per_sample_loaded"] == {"TT_PU200": 60, "VBFHToInvisible_PU200": 60}
    assert comp["per_sample_requested"] == {"TT_PU200": 200_000, "VBFHToInvisible_PU200": 80_000}


def test_build_dataset_card_is_json_serializable(tmp_path):
    splits = _toy_splits()
    build_dataset_card(
        output_dir=tmp_path,
        tag="test_tag",
        splits=splits,
        feature_layout=["pt", "eta", "phi", "puppi_weight"],
        event_feature_layout=None,
        per_sample_loaded={"TT_PU200": 100},
        per_sample_requested={"TT_PU200": 200_000},
        provenance={},
    )
    # The on-disk JSON should round-trip cleanly (no NaN, no numpy types,
    # no Path objects, etc.).
    loaded = json.loads((tmp_path / "dataset_card.json").read_text())
    assert loaded["tag"] == "test_tag"


def test_render_markdown_has_required_sections(tmp_path):
    splits = _toy_splits()
    card = build_dataset_card(
        output_dir=tmp_path,
        tag="test_tag",
        splits=splits,
        feature_layout=["pt", "eta", "phi", "puppi_weight"],
        event_feature_layout=["puppi_met_pt"],
        per_sample_loaded={"TT_PU200": 120},
        per_sample_requested={"TT_PU200": 200_000},
        provenance={"git_sha": "abc1234"},
    )
    md = render_markdown(card)
    # Section headings reviewers will want to find at a glance
    for needle in ("# Dataset card", "## Provenance", "## Composition",
                   "## Per-candidate features", "## Splits"):
        assert needle in md, f"missing section: {needle!r}"
