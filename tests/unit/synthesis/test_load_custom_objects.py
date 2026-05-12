"""Test that ``synthesize.load_custom_objects()`` covers everything our
saved models can contain — both full-precision and HGQ2 — so
``tf.keras.models.load_model(path, custom_objects=...)`` works without the
caller needing to know which family the model came from.

This regression was discovered when the QAT-validation integration test
trained an HGQ2 model, saved it, and synthesize.py failed to load it back
("Could not locate class 'QDense'"). The fix added the HGQ2 layer
classes to the custom-objects dict.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

pytest.importorskip("hgq")
pytest.importorskip("hls4ml")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from synthesize import load_custom_objects  # noqa: E402


def test_load_custom_objects_includes_hgq2_layers():
    """Saved HGQ2 models reference QDense, QBatchNormalization, etc. by
    name. ``load_custom_objects()`` must surface all of them."""
    co = load_custom_objects()
    # Subset that the bounded-no-bias HGQ2 architecture uses.
    expected = {"QDense", "QBatchNormalization", "QMultiply",
                 "QGlobalAveragePooling1D"}
    missing = expected - set(co)
    assert not missing, f"load_custom_objects missing HGQ2 layers: {missing}"


def test_load_custom_objects_includes_full_precision_layers():
    """Saved full-precision models reference our custom layer classes."""
    co = load_custom_objects()
    expected = {"CastToInt", "BoundedWeight", "ShiftByConstant",
                 "SumOverParticles", "ZeroReduce", "CorrectedCompositeLoss"}
    missing = expected - set(co)
    assert not missing, \
        f"load_custom_objects missing full-precision layers: {missing}"


def test_round_trip_hgq2_model_through_disk(tmp_path: Path):
    """End-to-end: build → save → load_model(custom_objects=load_custom_objects()).

    Verifies (1) the saved file format is portable and (2) the
    custom-objects dict is complete. If either is wrong, ``load_model``
    raises during deserialization.
    """
    from l1deepmet.quantization import build_hgq2_model, one_hot_encode_features

    src = build_hgq2_model(width=8, depth=1, use_embeddings=True,
                            bounded_weight=True)
    save_path = tmp_path / "hgq2_round_trip.keras"
    src.save(save_path)
    assert save_path.is_file()

    co = load_custom_objects()
    restored = tf.keras.models.load_model(save_path, custom_objects=co,
                                          compile=False)
    assert restored.count_params() == src.count_params()

    # Numerical equivalence: same outputs on a random batch.
    rng = np.random.default_rng(7)
    B = 2
    pdg = rng.integers(0, 6, size=(B, 128))
    chg = rng.integers(0, 4, size=(B, 128))
    pdg_oh, chg_oh = one_hot_encode_features(pdg, chg)
    inputs = {
        "continuous_inputs": rng.standard_normal((B, 128, 5)).astype(np.float32),
        "momentum_inputs":   rng.standard_normal((B, 128, 2)).astype(np.float32) * 10,
        "pdgid_inputs":      pdg_oh,
        "charge_inputs":     chg_oh,
    }
    y_src = src.predict(inputs, verbose=0)
    y_restored = restored.predict(inputs, verbose=0)
    np.testing.assert_allclose(y_src, y_restored, atol=1e-6)
