"""Tests for the input-prep helper in ``l1deepmet.synthesis.validate``.

The same ``synthesize.py`` script handles both full-precision and HGQ2
models. They have different categorical-input shapes:

  - FP32:  ``pdgid_inputs`` is ``(B, 128)`` int-style.
  - HGQ2:  ``pdgid_inputs`` is ``(B, 128, 6)`` pre-encoded one-hot
            (HGQ2 lacks ``QEmbedding``; we use one-hot + ``QDense``).

The validator must auto-detect this from the loaded Keras model's input
spec and one-hot encode on the fly when needed. Without that detection
hls4ml's HGQ2-conversion path fails with
``Invalid input shape for ... Expected shape (None, 128, 6), but input has
incompatible shape (500, 128)``.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

from l1deepmet.synthesis.validate import _model_inputs_from_features


def _features():
    rng = np.random.default_rng(0)
    X = np.zeros((4, 128, 9), dtype=np.float32)
    X[..., 0:5] = rng.standard_normal((4, 128, 5))
    X[..., 5:7] = rng.standard_normal((4, 128, 2))
    X[..., 7] = rng.integers(0, 6, size=(4, 128))
    X[..., 8] = rng.integers(0, 4, size=(4, 128))
    return X


def _toy_fp32_model():
    """Mock the FP32 model's input signature: pdgid / charge are (B, N).

    All inputs must be wired into the output so Keras accepts the model.
    """
    import keras
    cont = keras.Input(shape=(128, 5), name="continuous_inputs")
    pxpy = keras.Input(shape=(128, 2), name="momentum_inputs")
    pdg = keras.Input(shape=(128,), name="pdgid_inputs")
    chg = keras.Input(shape=(128,), name="charge_inputs")
    # Use sums over the unused axes so every input has gradient path.
    h_cont = keras.layers.GlobalAveragePooling1D()(cont)
    h_pxpy = keras.layers.GlobalAveragePooling1D()(pxpy)
    h_pdg = keras.layers.GlobalAveragePooling1D()(keras.layers.Reshape((128, 1))(pdg))
    h_chg = keras.layers.GlobalAveragePooling1D()(keras.layers.Reshape((128, 1))(chg))
    h = keras.layers.Concatenate()([h_cont, h_pxpy, h_pdg, h_chg])
    return keras.Model(inputs=[cont, pxpy, pdg, chg],
                       outputs=keras.layers.Dense(2)(h))


def _toy_hgq2_model():
    """Mock the HGQ2 input signature: pdgid / charge are (B, N, vocab)."""
    import keras
    cont = keras.Input(shape=(128, 5), name="continuous_inputs")
    pxpy = keras.Input(shape=(128, 2), name="momentum_inputs")
    pdg = keras.Input(shape=(128, 6), name="pdgid_inputs")
    chg = keras.Input(shape=(128, 4), name="charge_inputs")
    h_cont = keras.layers.GlobalAveragePooling1D()(cont)
    h_pxpy = keras.layers.GlobalAveragePooling1D()(pxpy)
    h_pdg = keras.layers.GlobalAveragePooling1D()(pdg)
    h_chg = keras.layers.GlobalAveragePooling1D()(chg)
    h = keras.layers.Concatenate()([h_cont, h_pxpy, h_pdg, h_chg])
    return keras.Model(inputs=[cont, pxpy, pdg, chg],
                       outputs=keras.layers.Dense(2)(h))


class TestAutoDetectInputShape:
    def test_fp32_model_gets_int_style_categoricals(self):
        X = _features()
        inputs = _model_inputs_from_features(X, model=_toy_fp32_model())
        # pdgid_inputs should be (B, 128), not one-hot.
        assert inputs["pdgid_inputs"].shape == (4, 128)
        assert inputs["charge_inputs"].shape == (4, 128)
        # Continuous and momentum unchanged.
        assert inputs["continuous_inputs"].shape == (4, 128, 5)
        assert inputs["momentum_inputs"].shape == (4, 128, 2)

    def test_hgq2_model_gets_one_hot_categoricals(self):
        X = _features()
        inputs = _model_inputs_from_features(X, model=_toy_hgq2_model())
        assert inputs["pdgid_inputs"].shape == (4, 128, 6)
        assert inputs["charge_inputs"].shape == (4, 128, 4)
        # Each particle row should sum to 1 in the vocab dim.
        np.testing.assert_array_equal(
            inputs["pdgid_inputs"].sum(axis=-1), np.ones((4, 128)),
        )
        np.testing.assert_array_equal(
            inputs["charge_inputs"].sum(axis=-1), np.ones((4, 128)),
        )

    def test_omit_model_falls_back_to_int_style(self):
        """Backwards-compatible default: callers that don't pass ``model=``
        get the original FP32-style int categoricals."""
        X = _features()
        inputs = _model_inputs_from_features(X)
        assert inputs["pdgid_inputs"].shape == (4, 128)
        assert inputs["charge_inputs"].shape == (4, 128)
