"""Tests for the extended preprocessor's event-level feature path.

The event-feature loader extracts per-event scalar quantities (MET algorithm
outputs, leading-vertex z0, vertex multiplicity, ...) from an uproot iterate
batch and assembles them into an (n_events, n_event_features) array.

Behaviours covered here:
- Scalar branches pass through unchanged.
- Jagged branches with `agg="first"` take the leading entry per event.
- Empty rows in a jagged branch fall back to the default.
- Missing branches in the input ROOT file are zero-filled.
- The assembly produces the right shape with columns in feature_layout order.
"""

from __future__ import annotations

import numpy as np
import awkward as ak  # type: ignore
import pytest

from l1deepmet.data.preprocessing import (
    EVENT_FEATURE_BRANCHES,
    _load_event_field,
    event_features_from_arrays,
)


# ─── _load_event_field ─────────────────────────────────────────────────────


def test_load_event_field_scalar_branch_passes_through():
    field = ak.Array([10.0, 20.0, 30.0])
    out = _load_event_field(field, agg="scalar")
    assert out.shape == (3,)
    np.testing.assert_array_almost_equal(out, [10.0, 20.0, 30.0])


def test_load_event_field_first_of_jagged_takes_leading_entry():
    field = ak.Array([[1.5, 2.5], [3.0], [4.0, 5.0, 6.0]])
    out = _load_event_field(field, agg="first")
    assert out.shape == (3,)
    np.testing.assert_array_almost_equal(out, [1.5, 3.0, 4.0])


def test_load_event_field_first_of_empty_uses_default():
    field = ak.Array([[1.5, 2.5], [], [4.0]])
    out = _load_event_field(field, agg="first", default=-99.0)
    np.testing.assert_array_almost_equal(out, [1.5, -99.0, 4.0])


def test_load_event_field_invalid_agg_raises():
    field = ak.Array([1.0, 2.0])
    with pytest.raises(ValueError):
        _load_event_field(field, agg="bogus")


# ─── event_features_from_arrays ────────────────────────────────────────────


def _fake_arrays(n_events: int = 3):
    """Build a fake uproot-style Record that mimics a single iterate batch.

    Includes a mix of scalar branches (per-event METs) and one jagged branch
    (vertex table). Mirrors what `uproot.iterate(..., library="ak")` returns.
    """
    return ak.zip(
        {
            "L1PuppiMet_pt": ak.Array([50.0, 60.0, 70.0][:n_events]),
            "L1PuppiMet_phi": ak.Array([0.1, -0.2, 1.5][:n_events]),
            "L1Layer2Met_pt": ak.Array([55.0, 65.0, 75.0][:n_events]),
            "L1Layer2Met_phi": ak.Array([0.2, -0.1, 1.4][:n_events]),
            "L1Vtx_z0": ak.Array([[0.5, 0.6], [], [-1.0, -0.5, 2.0]][:n_events]),
            "L1Vtx_sumPt": ak.Array([[100.0, 50.0], [], [80.0, 30.0, 20.0]][:n_events]),
            "nL1Vtx": ak.Array([2, 0, 3][:n_events]),
        },
        depth_limit=1,
    )


def test_event_features_from_arrays_returns_expected_shape():
    arrays = _fake_arrays(n_events=3)
    layout = ["puppi_met_pt", "puppi_met_phi", "lead_vtx_z0", "n_vtx"]
    out = event_features_from_arrays(arrays, layout, n_events=3)
    assert out.shape == (3, 4)
    assert out.dtype == np.float32


def test_event_features_from_arrays_columns_in_layout_order():
    arrays = _fake_arrays(n_events=3)
    layout = ["puppi_met_pt", "lead_vtx_z0", "n_vtx", "layer2_met_pt"]
    out = event_features_from_arrays(arrays, layout, n_events=3)
    # column 0 = puppi_met_pt
    np.testing.assert_array_almost_equal(out[:, 0], [50.0, 60.0, 70.0])
    # column 1 = lead_vtx_z0 — first of jagged, zero for empty event
    np.testing.assert_array_almost_equal(out[:, 1], [0.5, 0.0, -1.0])
    # column 2 = n_vtx — scalar count
    np.testing.assert_array_almost_equal(out[:, 2], [2.0, 0.0, 3.0])
    # column 3 = layer2_met_pt
    np.testing.assert_array_almost_equal(out[:, 3], [55.0, 65.0, 75.0])


def test_event_features_from_arrays_missing_branch_is_zero_filled():
    """When the input ROOT file lacks a branch (e.g. legacy ntuple without
    L1Layer2Met), the corresponding column should be all zeros rather than
    raising — same robustness guarantee as the per-candidate extended path."""
    arrays = ak.zip(
        {
            "L1PuppiMet_pt": ak.Array([50.0, 60.0]),
        },
        depth_limit=1,
    )
    layout = ["puppi_met_pt", "layer2_met_pt", "lead_vtx_z0"]
    out = event_features_from_arrays(arrays, layout, n_events=2)
    assert out.shape == (2, 3)
    np.testing.assert_array_almost_equal(out[:, 0], [50.0, 60.0])
    np.testing.assert_array_almost_equal(out[:, 1], [0.0, 0.0])  # missing
    np.testing.assert_array_almost_equal(out[:, 2], [0.0, 0.0])  # missing


def test_event_features_from_arrays_unknown_feature_raises():
    arrays = _fake_arrays(n_events=2)
    with pytest.raises(ValueError, match="Unknown event feature"):
        event_features_from_arrays(arrays, ["not_a_real_feature"], n_events=2)


# ─── EVENT_FEATURE_BRANCHES catalogue ─────────────────────────────────────


def test_event_feature_branches_covers_params_yaml_layout():
    """All event features named in params.yaml::preprocess.event_feature_layout
    must be in the catalogue. If you add a new feature to the YAML, you must
    also map it in EVENT_FEATURE_BRANCHES — this test prevents silent
    "unknown feature" errors at preprocess time."""
    expected = {
        "puppi_met_pt", "puppi_met_phi",
        "puppi_met_central_pt", "puppi_met_central_phi",
        "pf_met_pt", "pf_met_phi",
        "calo_met_pt", "calo_met_phi",
        "tk_met_pt", "tk_met_phi",
        "layer2_met_pt", "layer2_met_phi",
        "lead_vtx_z0", "lead_vtx_sumpt", "n_vtx",
    }
    catalogued = set(EVENT_FEATURE_BRANCHES.keys())
    missing = expected - catalogued
    assert not missing, f"event features in params.yaml missing from EVENT_FEATURE_BRANCHES: {missing}"
