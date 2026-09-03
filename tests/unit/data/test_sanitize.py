"""Tests for the extreme-value sanitization step.

Some recipe accessors (e.g. the HGCal EM/PU MVAs `pfCluster.egVsPUMVAOut`,
`pfCluster.egVsPionMVAOut`) return numerical sentinels of the order
±FLT_MAX (~±3.4e+38) when their inputs are invalid (cluster geometry out
of acceptance, all shower-shape moments zero, etc.). Those values
silently poison any downstream normalization layer (std → ∞, gradients →
NaN) without tripping the standard NaN/Inf checks because the floats are
still finite.

`sanitize_extreme_values` clamps |x| > threshold AND non-finite values to
a replacement (default 0). It works on both 3-D per-candidate arrays
(B, N, F) and 2-D event-feature arrays (B, F), returning the sanitized
array plus per-feature touch counts.
"""

from __future__ import annotations

import numpy as np

from l1deepmet.data.preprocessing import sanitize_extreme_values


def test_clips_flt_max_class_values_in_3d():
    """The clPuId / clEmId failure mode: ±10^34 to ±10^38 in slot 22-23."""
    X = np.zeros((2, 4, 26), dtype=np.float32)
    # legitimate values
    X[:, :, 0] = 5.0          # pt
    X[:, :, 1] = -2.0         # eta
    # garbage in clPuId / clEmId slots
    X[0, 0, 22] = 1.26e+34
    X[0, 1, 22] = -4.48e+34
    X[1, 0, 23] = 3.38e+38
    X[1, 1, 23] = -2.58e+38
    Y, stats = sanitize_extreme_values(X)
    # garbage cleared
    assert Y[0, 0, 22] == 0.0
    assert Y[0, 1, 22] == 0.0
    assert Y[1, 0, 23] == 0.0
    assert Y[1, 1, 23] == 0.0
    # legitimate untouched
    assert (Y[:, :, 0] == 5.0).all()
    assert (Y[:, :, 1] == -2.0).all()
    # counts
    assert stats["n_touched_per_feature"][22] == 2
    assert stats["n_touched_per_feature"][23] == 2
    assert stats["n_touched_per_feature"][0] == 0


def test_replaces_inf_and_nan():
    X = np.zeros((1, 1, 3), dtype=np.float32)
    X[0, 0, 0] = np.inf
    X[0, 0, 1] = -np.inf
    X[0, 0, 2] = np.nan
    Y, stats = sanitize_extreme_values(X)
    assert np.isfinite(Y).all()
    assert (Y == 0.0).all()
    assert stats["n_touched_per_feature"].tolist() == [1, 1, 1]


def test_preserves_legitimate_large_values_below_threshold():
    """hwPt legitimately goes up to ~3200 — should NOT be touched."""
    X = np.zeros((1, 1, 1), dtype=np.float32)
    X[0, 0, 0] = 3197.0
    Y, stats = sanitize_extreme_values(X)
    assert Y[0, 0, 0] == 3197.0
    assert stats["n_touched_per_feature"][0] == 0


def test_works_on_2d_event_features():
    """Event features are (B, F), not (B, N, F)."""
    EX = np.zeros((3, 4), dtype=np.float32)
    EX[0, 0] = 50.0           # puppi_met_pt, legitimate
    EX[1, 2] = 1e+38          # garbage
    EX[2, 3] = np.inf
    EY, stats = sanitize_extreme_values(EX)
    assert EY[0, 0] == 50.0
    assert EY[1, 2] == 0.0
    assert EY[2, 3] == 0.0
    assert stats["n_touched_per_feature"].shape == (4,)
    assert stats["n_touched_per_feature"][2] == 1
    assert stats["n_touched_per_feature"][3] == 1


def test_custom_threshold_and_replacement():
    X = np.array([[[100.0, -5.0]]], dtype=np.float32)
    Y, _ = sanitize_extreme_values(X, threshold=50.0, replace=-1.0)
    assert Y[0, 0, 0] == -1.0   # 100 > 50
    assert Y[0, 0, 1] == -5.0   # |-5| ≤ 50


def test_idempotent():
    """Running sanitize twice gives the same result as once."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 8, 4)).astype(np.float32)
    X[0, 0, 0] = 1e+38
    X[1, 3, 2] = np.nan
    Y1, _ = sanitize_extreme_values(X)
    Y2, stats2 = sanitize_extreme_values(Y1)
    np.testing.assert_array_equal(Y1, Y2)
    assert (stats2["n_touched_per_feature"] == 0).all()


def test_does_not_mutate_input():
    X = np.array([[[1e+38]]], dtype=np.float32)
    X_copy = X.copy()
    sanitize_extreme_values(X)
    np.testing.assert_array_equal(X, X_copy)
