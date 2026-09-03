"""Tests for layout-aware sentinel/garbage cleaning of the extended layout.

Distinct from ``sanitize_extreme_values`` (a global |x|>1e6 / non-finite
clamp): this step is *feature-aware*. It targets the two failure modes the
extended-26 validation found that the global clamp misses:

  1. ``clPuId`` / ``clEmId`` carry finite garbage in the ±10^5–10^6 range —
     BELOW the 1e6 global threshold, so they survive. They are the L1 EG
     cluster MVAs whose ``-1`` charged-candidate fallback got bit-promoted.
     We restore them to ``-1`` (the intended sentinel).
  2. ``caloEta`` / ``caloPhi`` carry ``-999`` on neutrals (no pfTrack). That
     value is ~600× the real ±2.5 scale and blows up BatchNorm statistics.
     We neutralize it to 0.

The ``-1`` "absent" markers on z0 / track* / clPt / clEmEt are intentionally
KEPT — they sit on a comparable scale to the real values and the model also
sees encoded_pdgId / encoded_charge to disambiguate.
"""
from __future__ import annotations

import numpy as np
import pytest

from l1deepmet.data.preprocessing import (
    EXTENDED_CLEANING_SPEC,
    clean_extended_sentinels,
)

# minimal stand-in for feature_layout_extended (order matters for indices)
LAYOUT = [
    "pt", "eta", "phi", "puppi_weight", "px", "py", "encoded_pdgId",
    "encoded_charge", "dxy", "z0", "hwPt", "hwEta", "hwPhi", "hwPuppiWeight",
    "hwQual", "trackChi2RPhi", "trackChi2RZ", "trackChi2Bend", "trackNStubs",
    "trackMvaQual", "caloEta", "caloPhi", "clPuId", "clEmId", "clPt", "clEmEt",
]
I = {n: i for i, n in enumerate(LAYOUT)}


def _base(n=2, p=4):
    """A clean (B,N,F) array with sane physics values in every slot."""
    X = np.zeros((n, p, len(LAYOUT)), dtype=np.float32)
    X[..., I["pt"]] = 5.0
    X[..., I["eta"]] = 1.2
    X[..., I["caloEta"]] = 1.1
    X[..., I["caloPhi"]] = 0.5
    X[..., I["clPuId"]] = 0.7
    X[..., I["clEmId"]] = 0.3
    X[..., I["clPt"]] = 12.0
    X[..., I["trackChi2RZ"]] = 0.5
    return X


def test_caloeta_minus999_neutralized_to_zero():
    X = _base()
    X[0, 0, I["caloEta"]] = -999.0
    X[1, 2, I["caloPhi"]] = -999.0
    Y, _ = clean_extended_sentinels(X, LAYOUT)
    assert Y[0, 0, I["caloEta"]] == 0.0
    assert Y[1, 2, I["caloPhi"]] == 0.0


def test_cluster_garbage_restored_to_minus1():
    """Finite ±1e5–1e6 garbage below the global 1e6 clamp -> -1 sentinel."""
    X = _base()
    X[0, 0, I["clPuId"]] = 9.59e5
    X[0, 1, I["clPuId"]] = -9.92e5
    X[1, 0, I["clEmId"]] = 7.88e5
    X[1, 1, I["clEmId"]] = -1.68e5
    Y, _ = clean_extended_sentinels(X, LAYOUT)
    assert Y[0, 0, I["clPuId"]] == -1.0
    assert Y[0, 1, I["clPuId"]] == -1.0
    assert Y[1, 0, I["clEmId"]] == -1.0
    assert Y[1, 1, I["clEmId"]] == -1.0


def test_real_cluster_scores_preserved():
    """O(1) real MVA scores must pass through untouched."""
    X = _base()
    X[0, 0, I["clPuId"]] = 0.92
    X[0, 1, I["clEmId"]] = -0.4
    Y, _ = clean_extended_sentinels(X, LAYOUT)
    assert Y[0, 0, I["clPuId"]] == pytest.approx(0.92)
    assert Y[0, 1, I["clEmId"]] == pytest.approx(-0.4)


def test_intermediate_corruption_and_saturation_clipped():
    """The softer ±10–99.9 corruption / saturation spike -> -1 too."""
    X = _base()
    X[0, 0, I["clPuId"]] = 99.9      # saturation spike
    X[0, 1, I["clPuId"]] = -95.586   # observed corruption value
    X[1, 0, I["clEmId"]] = 50.0
    X[1, 1, I["clEmId"]] = -64.0
    Y, _ = clean_extended_sentinels(X, LAYOUT)
    assert Y[0, 0, I["clPuId"]] == -1.0
    assert Y[0, 1, I["clPuId"]] == -1.0
    assert Y[1, 0, I["clEmId"]] == -1.0
    assert Y[1, 1, I["clEmId"]] == -1.0


def test_minus1_markers_on_track_and_cluster_features_kept():
    """-1 'absent' markers on z0/track*/clPt/clEmEt are NOT cleaned."""
    X = _base()
    for f in ("z0", "trackChi2RPhi", "trackChi2RZ", "trackNStubs",
              "trackMvaQual", "clPt", "clEmEt"):
        X[0, 0, I[f]] = -1.0
    Y, _ = clean_extended_sentinels(X, LAYOUT)
    for f in ("z0", "trackChi2RPhi", "trackChi2RZ", "trackNStubs",
              "trackMvaQual", "clPt", "clEmEt"):
        assert Y[0, 0, I[f]] == -1.0, f"{f} -1 marker was wrongly cleaned"


def test_clean_features_untouched():
    """Slots not in the spec (pt, eta, ...) must be identical."""
    X = _base()
    Y, _ = clean_extended_sentinels(X, LAYOUT)
    for f in ("pt", "eta", "phi", "puppi_weight", "hwPt", "hwEta"):
        np.testing.assert_array_equal(Y[..., I[f]], X[..., I[f]])


def test_touched_counts_per_feature():
    X = _base()
    X[0, 0, I["caloEta"]] = -999.0
    X[0, 1, I["caloEta"]] = -999.0
    X[1, 0, I["clPuId"]] = 5e5
    Y, stats = clean_extended_sentinels(X, LAYOUT)
    t = stats["n_touched_per_feature"]
    assert t.shape == (len(LAYOUT),)
    assert t[I["caloEta"]] == 2
    assert t[I["clPuId"]] == 1
    assert t[I["pt"]] == 0


def test_does_not_mutate_input():
    X = _base()
    X[0, 0, I["caloEta"]] = -999.0
    X_copy = X.copy()
    clean_extended_sentinels(X, LAYOUT)
    np.testing.assert_array_equal(X, X_copy)


def test_idempotent():
    X = _base()
    X[0, 0, I["caloEta"]] = -999.0
    X[1, 0, I["clEmId"]] = 5e5
    Y1, _ = clean_extended_sentinels(X, LAYOUT)
    Y2, stats2 = clean_extended_sentinels(Y1, LAYOUT)
    np.testing.assert_array_equal(Y1, Y2)
    assert (stats2["n_touched_per_feature"] == 0).all()


def test_missing_feature_in_layout_is_skipped():
    """A short layout lacking spec'd features must not raise."""
    short = ["pt", "eta", "phi"]
    X = np.zeros((1, 1, 3), dtype=np.float32)
    Y, stats = clean_extended_sentinels(X, short)
    assert Y.shape == X.shape
    assert stats["n_touched_per_feature"].shape == (3,)


def test_spec_covers_the_known_problem_features():
    """Guard: the four features the validation flagged must be in the spec."""
    for f in ("caloEta", "caloPhi", "clPuId", "clEmId"):
        assert f in EXTENDED_CLEANING_SPEC
