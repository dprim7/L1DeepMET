"""Tests for ``l1deepmet.estimation.analytic`` — the technology-independent
per-layer params/MACs table.

The key requirement is rank-3 awareness: a body ``Dense`` shares its kernel
across the 128 particle slots, so its MAC count is ``128 * d_in * units``,
not ``d_in * units``. Getting this wrong makes every per-particle
architecture look 128x cheaper than it is.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: F401
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

import keras  # noqa: E402

from l1deepmet.estimation.analytic import AnalyticReport, analyze_model  # noqa: E402


def _row(report: AnalyticReport, name: str):
    matches = [r for r in report.layers if r.name == name]
    assert matches, f"no layer named {name!r} in report: {[r.name for r in report.layers]}"
    return matches[0]


class TestDenseMacs:
    def test_rank2_dense(self):
        inp = keras.Input(shape=(3,), name="x")
        model = keras.Model(inp, keras.layers.Dense(4, name="d")(inp))
        report = analyze_model(model)
        row = _row(report, "d")
        assert row.macs == 3 * 4
        assert row.params == 3 * 4 + 4

    def test_rank3_dense_counts_particle_axis(self):
        """A per-particle Dense applied over 128 slots does 128x the work."""
        inp = keras.Input(shape=(128, 11), name="x")
        model = keras.Model(inp, keras.layers.Dense(8, name="d")(inp))
        report = analyze_model(model)
        row = _row(report, "d")
        assert row.macs == 128 * 11 * 8
        assert row.params == 11 * 8 + 8


class TestFp32ScalarModel:
    """The production-family architecture: mlp body, embeddings, bounded
    scalar-weight head, sum aggregation."""

    @pytest.fixture(scope="class")
    def report(self):
        from arch_search import ArchConfig, build_model

        cfg = ArchConfig(name="tiny", width=8, depth=1, mode=1,
                         activation="relu", use_embeddings=True,
                         binned_weight=0.0, bounded_weight=True,
                         use_sum=True)
        return analyze_model(build_model(cfg))

    def test_no_unhandled_layers(self, report):
        assert report.unhandled == []

    def test_body_dense_is_rank3(self, report):
        assert _row(report, "dense_0").macs == 128 * 11 * 8

    def test_weight_head_is_rank3(self, report):
        assert _row(report, "met_weight").macs == 128 * 8 * 1

    def test_embeddings_are_lookups(self, report):
        pdg = _row(report, "emb_pdgid")
        chg = _row(report, "emb_charge")
        assert (pdg.params, pdg.macs) == (6 * 4, 0)
        assert (chg.params, chg.macs) == (4 * 2, 0)

    def test_sum_over_particles_adds(self, report):
        assert _row(report, "output").elementwise_ops == (128 - 1) * 2

    def test_bounded_weight_notes_tanh(self, report):
        assert "tanh" in _row(report, "bounded_weight").notes.lower()

    def test_totals(self, report):
        assert report.total_params == sum(r.params for r in report.layers)
        assert report.total_macs == sum(r.macs for r in report.layers
                                        if r.macs is not None)
        assert report.total_macs > 128 * 11 * 8  # at least the body


class TestTransformerModel:
    @pytest.fixture(scope="class")
    def built(self):
        from arch_search import ArchConfig, build_model

        cfg = ArchConfig(name="tiny_tr", width=8, depth=1, mode=1,
                         activation="relu", use_embeddings=True,
                         binned_weight=0.0, bounded_weight=True,
                         use_sum=True, body_type="transformer",
                         num_heads=2, ffn_dim=4)
        return cfg, analyze_model(build_model(cfg))

    def test_no_unhandled_layers(self, built):
        _, report = built
        assert report.unhandled == []

    def test_mha_macs_formula(self, built):
        cfg, report = built
        n, d = 128, cfg.width
        h = cfg.num_heads
        k = v = cfg.width // cfg.num_heads  # key_dim=0 -> width // num_heads
        expected = (
            n * d * h * k          # Q projection
            + n * d * h * k        # K projection
            + n * d * h * v        # V projection
            + h * n * n * k        # QK^T
            + h * n * n * v        # attn . V
            + n * h * v * d        # output projection
        )
        assert _row(report, "mha_0").macs == expected

    def test_ffn_denses_are_rank3(self, built):
        cfg, report = built
        assert _row(report, "ffn_in_0").macs == 128 * cfg.width * cfg.ffn_dim
        assert _row(report, "ffn_out_0").macs == 128 * cfg.ffn_dim * cfg.width

    def test_layernorm_and_mask_handled(self, built):
        _, report = built
        assert _row(report, "ln_attn_0").macs == 0
        assert "approx" in _row(report, "ln_attn_0").notes.lower()
        assert _row(report, "padding_mask").macs == 0


class TestMode0DeepSets:
    def test_rho_dense_is_rank2_after_pooling(self):
        from arch_search import ArchConfig, build_model

        cfg = ArchConfig(name="tiny_ds", width=8, depth=1, mode=0,
                         activation="relu", use_embeddings=True,
                         binned_weight=0.0, use_sum=True,
                         rho_depth=1, rho_width=4)
        report = analyze_model(build_model(cfg))
        assert report.unhandled == []
        # After SumOverParticles the particle axis is gone: no x128.
        assert _row(report, "rho_dense_0").macs == 8 * 4
        assert _row(report, "output").macs == 4 * 2


class TestHgq2Analytic:
    def test_q_layers_dispatch_via_mro(self):
        pytest.importorskip("hgq")
        from l1deepmet.quantization import build_hgq2_model

        model = build_hgq2_model(width=8, depth=1, use_embeddings=True,
                                 bounded_weight=True)
        report = analyze_model(model)
        assert report.unhandled == []
        # one-hot 'embedding' is a QDense: (128, 6) -> (128, 4)
        assert _row(report, "emb_pdgid").macs == 128 * 6 * 4
        # body QDense inherits the rank-3 Dense rule
        assert _row(report, "qdense_0").macs == 128 * 11 * 8


class TestFallbacksAndSerialization:
    def test_unknown_layer_reported_not_fatal(self):
        @keras.saving.register_keras_serializable(package="test_analytic")
        class Mystery(keras.layers.Layer):
            def call(self, x):
                return x

        inp = keras.Input(shape=(4,), name="x")
        model = keras.Model(inp, Mystery(name="mystery")(inp))
        report = analyze_model(model)
        row = _row(report, "mystery")
        assert row.macs is None
        assert "mystery" in report.unhandled

    def test_to_dict_is_json_serializable(self):
        inp = keras.Input(shape=(128, 5), name="x")
        model = keras.Model(inp, keras.layers.Dense(3, name="d")(inp))
        report = analyze_model(model)
        payload = json.dumps(report.to_dict())
        round_tripped = json.loads(payload)
        assert round_tripped["total_macs"] == 128 * 5 * 3
        assert round_tripped["layers"][0]["name"] == "d"
