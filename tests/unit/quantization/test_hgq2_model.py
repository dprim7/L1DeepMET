"""Unit tests for ``l1deepmet.quantization.hgq2_model``.

Covers the build_hgq2_model factory, the one-hot helper, and an end-to-end
mini-train smoke test. Does NOT test hls4ml conversion — that's a separate
integration concern (and currently blocked, see quantization/README.md).
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

# HGQ2 / keras import; skip module if unavailable in the env.
hgq = pytest.importorskip("hgq")

from l1deepmet.quantization import build_hgq2_model, one_hot_encode_features  # noqa: E402
from l1deepmet.quantization.hgq2_model import (  # noqa: E402
    CHARGE_VOCAB, N_PARTICLES, PDGID_VOCAB,
)


# ─── one_hot_encode_features ─────────────────────────────────────────────────

class TestOneHotEncode:
    def test_shapes(self):
        pdg = np.zeros((4, N_PARTICLES), dtype=np.int32)
        chg = np.zeros((4, N_PARTICLES), dtype=np.int32)
        pdg_oh, chg_oh = one_hot_encode_features(pdg, chg)
        assert pdg_oh.shape == (4, N_PARTICLES, PDGID_VOCAB)
        assert chg_oh.shape == (4, N_PARTICLES, CHARGE_VOCAB)

    def test_one_hot_correctness(self):
        """Each index encoded to a one-hot row that sums to 1."""
        pdg = np.array([[0, 1, 2, 3, 4, 5]])
        chg = np.array([[0, 1, 2, 3]])
        pdg_oh, chg_oh = one_hot_encode_features(pdg, chg)
        # Sum along the vocab axis must be 1 everywhere.
        np.testing.assert_array_equal(pdg_oh.sum(axis=-1), np.ones((1, 6)))
        np.testing.assert_array_equal(chg_oh.sum(axis=-1), np.ones((1, 4)))
        # The "1" should be at the index given by the input value.
        for i, v in enumerate(pdg[0]):
            assert pdg_oh[0, i, v] == 1.0

    def test_clamps_out_of_range(self):
        """Indices outside [0, vocab) should be clipped, not crash."""
        pdg = np.array([[-1, 999, 3]])
        chg = np.array([[-1, 999, 2]])
        pdg_oh, chg_oh = one_hot_encode_features(pdg, chg)
        assert pdg_oh[0, 0, 0] == 1.0     # -1 clipped to 0
        assert pdg_oh[0, 1, PDGID_VOCAB - 1] == 1.0  # 999 clipped to vocab-1
        assert chg_oh[0, 0, 0] == 1.0
        assert chg_oh[0, 1, CHARGE_VOCAB - 1] == 1.0

    def test_dtype_is_float32(self):
        pdg = np.zeros((1, N_PARTICLES), dtype=np.int32)
        chg = np.zeros((1, N_PARTICLES), dtype=np.int32)
        pdg_oh, chg_oh = one_hot_encode_features(pdg, chg)
        assert pdg_oh.dtype == np.float32
        assert chg_oh.dtype == np.float32


# ─── build_hgq2_model ────────────────────────────────────────────────────────

class TestBuildHgq2Model:
    def test_default_build_returns_keras_model(self):
        import keras
        m = build_hgq2_model()
        assert isinstance(m, keras.Model)

    def test_input_signature_matches_full_precision_arch(self):
        """The model exposes the same four dict-style inputs as
        ``arch_search.build_model``. pdgid / charge are pre-encoded one-hot,
        not int indices.
        """
        m = build_hgq2_model(use_embeddings=True)
        names = [layer.name for layer in m.inputs]
        assert set(names) == {
            "continuous_inputs", "momentum_inputs",
            "pdgid_inputs", "charge_inputs",
        }
        shapes = {layer.name: tuple(layer.shape) for layer in m.inputs}
        # batch dim is None, then particle axis, then feature axis
        assert shapes["continuous_inputs"] == (None, N_PARTICLES, 5)
        assert shapes["momentum_inputs"]   == (None, N_PARTICLES, 2)
        assert shapes["pdgid_inputs"]      == (None, N_PARTICLES, PDGID_VOCAB)
        assert shapes["charge_inputs"]     == (None, N_PARTICLES, CHARGE_VOCAB)

    def test_output_shape_is_2d_met(self):
        m = build_hgq2_model()
        assert tuple(m.output.shape) == (None, 2)

    def test_depth_controls_body_layer_count(self):
        m2 = build_hgq2_model(depth=2)
        m4 = build_hgq2_model(depth=4)
        n2 = sum(1 for l in m2.layers if l.name.startswith("qdense_"))
        n4 = sum(1 for l in m4.layers if l.name.startswith("qdense_"))
        assert n2 == 2
        assert n4 == 4

    def test_bounded_weight_adds_tanh_and_scale_layers(self):
        m_bounded = build_hgq2_model(bounded_weight=True)
        m_plain = build_hgq2_model(bounded_weight=False)
        names_bounded = [l.name for l in m_bounded.layers]
        names_plain = [l.name for l in m_plain.layers]
        assert "qtanh" in names_bounded and "qbounded_scale" in names_bounded
        assert "qweight_shift" in names_plain
        assert "qtanh" not in names_plain

    def test_no_embeddings_drops_embedding_layers(self):
        m = build_hgq2_model(use_embeddings=False)
        names = [l.name for l in m.layers]
        assert "emb_pdgid" not in names
        assert "emb_charge" not in names

    def test_forward_pass_with_realistic_inputs(self):
        m = build_hgq2_model(width=16, depth=2)   # small model for speed
        rng = np.random.default_rng(0)
        B = 4
        pdg = rng.integers(0, PDGID_VOCAB, size=(B, N_PARTICLES))
        chg = rng.integers(0, CHARGE_VOCAB, size=(B, N_PARTICLES))
        pdg_oh, chg_oh = one_hot_encode_features(pdg, chg)
        inputs = {
            "continuous_inputs": rng.standard_normal((B, N_PARTICLES, 5)).astype(np.float32),
            "momentum_inputs":   rng.standard_normal((B, N_PARTICLES, 2)).astype(np.float32) * 10,
            "pdgid_inputs":      pdg_oh,
            "charge_inputs":     chg_oh,
        }
        y = m.predict(inputs, verbose=0)
        assert y.shape == (B, 2)
        assert np.isfinite(y).all()


# ─── Mini-training smoke test ────────────────────────────────────────────────

class TestHgq2Training:
    def test_one_epoch_completes(self):
        """Train for one tiny epoch on synthetic data; should not raise."""
        m = build_hgq2_model(width=8, depth=1, use_embeddings=False,
                             bounded_weight=False)
        rng = np.random.default_rng(0)
        N = 16
        pdg = rng.integers(0, PDGID_VOCAB, size=(N, N_PARTICLES))
        chg = rng.integers(0, CHARGE_VOCAB, size=(N, N_PARTICLES))
        pdg_oh, chg_oh = one_hot_encode_features(pdg, chg)
        inputs = {
            "continuous_inputs": rng.standard_normal((N, N_PARTICLES, 5)).astype(np.float32),
            "momentum_inputs":   rng.standard_normal((N, N_PARTICLES, 2)).astype(np.float32),
            "pdgid_inputs":      pdg_oh,
            "charge_inputs":     chg_oh,
        }
        y = rng.standard_normal((N, 2)).astype(np.float32)
        m.compile(optimizer="adam", loss="mse")
        history = m.fit(inputs, y, epochs=1, batch_size=8, verbose=0)
        # Just confirm a loss was recorded.
        assert "loss" in history.history
        assert np.isfinite(history.history["loss"][0])
