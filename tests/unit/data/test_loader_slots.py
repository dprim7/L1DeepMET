"""Tests for the configurable continuous-slot routing in split_preprocessed_features.

The legacy 9-feature H5 and the extended 26-feature H5 use different column
positions for momentum (px/py at 5-6 vs 4-5) and categoricals (7-8 vs 6-7).
``split_preprocessed_features(X, continuous_slots=None)`` keeps the legacy
slicing; passing an explicit list switches to the extended convention.
"""

from __future__ import annotations

import numpy as np

from l1deepmet.data.loader import split_preprocessed_features


def _legacy_X(B: int = 2, N: int = 3) -> np.ndarray:
    """(B, N, 9) feature array with each slot set to its index (broadcast)."""
    X = np.zeros((B, N, 9), dtype=np.float32)
    for s in range(9):
        X[:, :, s] = s
    return X


def _extended_X(B: int = 2, N: int = 3) -> np.ndarray:
    X = np.zeros((B, N, 26), dtype=np.float32)
    for s in range(26):
        X[:, :, s] = s
    return X


def test_legacy_slicing_unchanged():
    """Default (no continuous_slots arg) preserves legacy 9-feature behaviour."""
    X = _legacy_X()
    inputs, pxpy, c0, c1 = split_preprocessed_features(X)
    assert inputs.shape == (2, 3, 5)
    assert pxpy.shape == (2, 3, 2)
    assert c0.shape == (2, 3)
    assert c1.shape == (2, 3)
    # Continuous = slots 0-4
    np.testing.assert_array_equal(inputs[0, 0], [0, 1, 2, 3, 4])
    # Momentum = slots 5-6
    np.testing.assert_array_equal(pxpy[0, 0], [5, 6])
    # Categoricals = slots 7, 8
    assert (c0 == 7).all()
    assert (c1 == 8).all()


def test_extended_slicing_with_explicit_slots():
    """Extended layout: continuous_slots picks columns; momentum=4-5; cats=6-7."""
    X = _extended_X()
    slots = [0, 1, 2, 3, 8, 9, 10]   # pt/eta/phi/puppi + dxy/z0/hwPt
    inputs, pxpy, c0, c1 = split_preprocessed_features(X, continuous_slots=slots)
    assert inputs.shape == (2, 3, 7)
    assert pxpy.shape == (2, 3, 2)
    np.testing.assert_array_equal(inputs[0, 0], [0, 1, 2, 3, 8, 9, 10])
    np.testing.assert_array_equal(pxpy[0, 0], [4, 5])   # extended px/py at slots 4-5
    assert (c0 == 6).all()
    assert (c1 == 7).all()


def test_extended_legacy_equivalent_slots():
    """Slots [0,1,2,3] on the extended layout pull just the 4 base kinematics
    (pt, eta, phi, puppi_weight). Momentum and cats from extended positions."""
    X = _extended_X()
    inputs, pxpy, c0, c1 = split_preprocessed_features(X, continuous_slots=[0, 1, 2, 3])
    assert inputs.shape == (2, 3, 4)
    np.testing.assert_array_equal(inputs[0, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(pxpy[0, 0], [4, 5])


def test_extended_all_continuous_slots():
    """The full 22-feature continuous block: 4 base + 18 extended."""
    X = _extended_X()
    full = [0, 1, 2, 3] + list(range(8, 26))   # exclude momentum (4,5) and cats (6,7)
    inputs, pxpy, c0, c1 = split_preprocessed_features(X, continuous_slots=full)
    assert inputs.shape == (2, 3, 22)
    # Confirms every selected slot's value flows through correctly
    np.testing.assert_array_equal(inputs[0, 0], full)


def test_continuous_slots_does_not_mutate_input():
    X = _extended_X()
    X_copy = X.copy()
    split_preprocessed_features(X, continuous_slots=[0, 8, 9])
    np.testing.assert_array_equal(X, X_copy)
