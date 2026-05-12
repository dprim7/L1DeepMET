"""Unit tests for ``l1deepmet.quantization.data``.

The HGQ2 model takes one-hot encoded pdgid / charge inputs (since hls4ml
can't synthesize tf.one_hot inside the model graph). This module wraps
the standard H5DataLoader pipeline with on-the-fly one-hot encoding so
the trainer can stay agnostic to the encoding step.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("hgq")

from l1deepmet.quantization.data import (  # noqa: E402
    encode_split_for_hgq2,
    make_hgq2_tf_dataset_from_features,
)
from l1deepmet.quantization.hgq2_model import CHARGE_VOCAB, PDGID_VOCAB  # noqa: E402


def _synthetic_features(n_events: int = 4, n_particles: int = 128, seed: int = 0):
    """Build a synthetic preprocessed-feature tensor matching the H5 layout.

    Layout (matches src/l1deepmet/data/loader.py::split_preprocessed_features):
        X[..., 0:5] = continuous (pt, eta, phi, puppi_w, hcal_depth)
        X[..., 5:7] = momentum (px, py)
        X[..., 7]   = encoded pdgId
        X[..., 8]   = encoded charge
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_events, n_particles, 9), dtype=np.float32)
    X[..., 0:5] = rng.standard_normal((n_events, n_particles, 5)).astype(np.float32)
    X[..., 5:7] = rng.standard_normal((n_events, n_particles, 2)).astype(np.float32) * 10
    X[..., 7] = rng.integers(0, PDGID_VOCAB, size=(n_events, n_particles)).astype(np.float32)
    X[..., 8] = rng.integers(0, CHARGE_VOCAB, size=(n_events, n_particles)).astype(np.float32)
    Y = rng.standard_normal((n_events, 2)).astype(np.float32) * 50.0
    return X, Y


class TestEncodeSplitForHgq2:
    def test_output_keys_match_hgq2_model_inputs(self):
        """The encoded dict must have exactly the four keys the HGQ2 model
        consumes — same set as the full-precision model."""
        X, Y = _synthetic_features()
        encoded, targets = encode_split_for_hgq2(X, Y)
        assert set(encoded.keys()) == {
            "continuous_inputs", "momentum_inputs",
            "pdgid_inputs", "charge_inputs",
        }
        np.testing.assert_array_equal(targets, Y)

    def test_continuous_and_momentum_shapes_unchanged(self):
        X, Y = _synthetic_features(n_events=8, n_particles=128)
        encoded, _ = encode_split_for_hgq2(X, Y)
        assert encoded["continuous_inputs"].shape == (8, 128, 5)
        assert encoded["momentum_inputs"].shape == (8, 128, 2)

    def test_pdgid_charge_become_one_hot(self):
        X, Y = _synthetic_features(n_events=3)
        encoded, _ = encode_split_for_hgq2(X, Y)
        assert encoded["pdgid_inputs"].shape == (3, 128, PDGID_VOCAB)
        assert encoded["charge_inputs"].shape == (3, 128, CHARGE_VOCAB)
        # Every row sums to 1 (one-hot).
        np.testing.assert_array_equal(
            encoded["pdgid_inputs"].sum(axis=-1), np.ones((3, 128))
        )
        np.testing.assert_array_equal(
            encoded["charge_inputs"].sum(axis=-1), np.ones((3, 128))
        )

    def test_one_hot_indices_correct(self):
        """Build a tiny case where we know the encoded indices and verify the
        one-hot positions are right."""
        X = np.zeros((1, 4, 9), dtype=np.float32)
        X[0, :, 7] = [0, 2, 4, 5]   # pdgid codes
        X[0, :, 8] = [0, 1, 2, 3]   # charge codes
        Y = np.zeros((1, 2), dtype=np.float32)
        encoded, _ = encode_split_for_hgq2(X, Y)
        for i, v in enumerate([0, 2, 4, 5]):
            assert encoded["pdgid_inputs"][0, i, v] == 1.0
        for i, v in enumerate([0, 1, 2, 3]):
            assert encoded["charge_inputs"][0, i, v] == 1.0

    def test_targets_normfac(self):
        """``normfac`` divides the targets, mirroring the standard
        H5DataLoader option used in arch_search training."""
        X, Y = _synthetic_features()
        encoded, targets = encode_split_for_hgq2(X, Y, normfac=100.0)
        np.testing.assert_allclose(targets, Y / 100.0)

    def test_targets_default_normfac_one(self):
        X, Y = _synthetic_features()
        _, targets = encode_split_for_hgq2(X, Y)
        np.testing.assert_array_equal(targets, Y)


class TestMakeTfDataset:
    """The TF dataset adapter for the training loop."""

    def test_batches_have_correct_shapes(self):
        X, Y = _synthetic_features(n_events=16)
        ds = make_hgq2_tf_dataset_from_features(X, Y, batch_size=4, normfac=100.0)
        # Pull one batch and check the input dict + targets.
        inputs, targets = next(iter(ds))
        assert set(inputs.keys()) == {
            "continuous_inputs", "momentum_inputs",
            "pdgid_inputs", "charge_inputs",
        }
        assert inputs["continuous_inputs"].shape == (4, 128, 5)
        assert inputs["momentum_inputs"].shape == (4, 128, 2)
        assert inputs["pdgid_inputs"].shape == (4, 128, PDGID_VOCAB)
        assert inputs["charge_inputs"].shape == (4, 128, CHARGE_VOCAB)
        assert targets.shape == (4, 2)

    def test_dataset_yields_expected_batch_count(self):
        X, Y = _synthetic_features(n_events=12)
        ds = make_hgq2_tf_dataset_from_features(X, Y, batch_size=4)
        n_batches = sum(1 for _ in ds)
        assert n_batches == 3  # 12 / 4

    def test_no_shuffle_preserves_order(self):
        """With ``shuffle=False`` the first batch should be the first 4 events."""
        X, Y = _synthetic_features(n_events=8)
        ds = make_hgq2_tf_dataset_from_features(X, Y, batch_size=4, shuffle=False)
        inputs, targets = next(iter(ds))
        np.testing.assert_array_equal(targets.numpy(), Y[:4])
