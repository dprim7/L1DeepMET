"""Unit tests for ``l1deepmet.synthesis.export``.

``export_for_hls`` rewrites a trained Keras model with hls4ml-friendly
primitives, copying weights so the rebuilt model is numerically equivalent to
the source on a random batch. Failure modes we care about:

  - Numerical equivalence breaks (rebuilt model gives different outputs).
  - Custom layers leak through unrewritten.
  - Weights for layers with 1:1 counterparts aren't copied.
  - The fixed-init layers (bounded_scale, weight_shift, output) get the
    correct fixed weights.

These tests build the source model from scratch (not from a saved file) so
they don't depend on any specific training run.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
from pathlib import Path

import numpy as np
import pytest

import tensorflow as tf  # noqa: E402

# Make ``scripts/arch_search`` importable (it lives outside the src/ package).
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

# Skip the whole module if arch_search isn't importable — e.g. if the user
# is running these tests against an older snapshot before the transformer
# refactor landed on this branch.
arch_search = pytest.importorskip("arch_search")
ArchConfig = arch_search.ArchConfig
build_model = arch_search.build_model
NORMFAC = arch_search.NORMFAC

from l1deepmet.synthesis.export import export_for_hls  # noqa: E402


N_PARTICLES = 128


def _random_inputs(rng, batch=4):
    return {
        "continuous_inputs": rng.standard_normal((batch, N_PARTICLES, 5)).astype(np.float32),
        "momentum_inputs":   rng.standard_normal((batch, N_PARTICLES, 2)).astype(np.float32) * 10.0,
        "pdgid_inputs":      rng.integers(0, 6, size=(batch, N_PARTICLES)).astype(np.float32),
        "charge_inputs":     rng.integers(0, 4, size=(batch, N_PARTICLES)).astype(np.float32),
    }


def _build_bounded_no_bias_source():
    """Re-build the residual-ablation winner architecture (without training)."""
    cfg = ArchConfig(
        name="test_bounded",
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=0.0,
        with_bias=False, weight_minus_one=False,
        bounded_weight=True, use_sum=True,
        xy_balance_weight=0.0,
    )
    return build_model(cfg)


def _build_weight_minus_one_source():
    """The weight_minus_one head — the older default before BoundedWeight."""
    cfg = ArchConfig(
        name="test_wm1",
        width=64, depth=3, mode=1, activation="relu",
        use_embeddings=True, binned_weight=0.0,
        with_bias=False, weight_minus_one=True,
        bounded_weight=False, use_sum=True,
        xy_balance_weight=0.0,
    )
    return build_model(cfg)


class TestExportBoundedModel:
    @pytest.fixture(scope="class")
    def source(self):
        return _build_bounded_no_bias_source()

    @pytest.fixture(scope="class")
    def exported(self, source):
        return export_for_hls(source)

    def test_exported_has_no_custom_layers(self, exported):
        """``export_for_hls`` must remove every layer that hls4ml doesn't know."""
        custom = {"CastToInt", "BoundedWeight", "SumOverParticles",
                   "ShiftByConstant", "ZeroReduce"}
        for layer in exported.layers:
            assert type(layer).__name__ not in custom, \
                f"Custom layer {type(layer).__name__} leaked into exported model"

    def test_exported_uses_integer_categorical_inputs(self, exported):
        """``CastToInt`` is removed by declaring the inputs as int32 directly."""
        for layer in exported.inputs:
            if layer.name in ("pdgid_inputs", "charge_inputs"):
                assert layer.dtype == "int32"

    def test_exported_has_global_average_pooling_not_custom_sum(self, exported):
        """``SumOverParticles`` is replaced by GlobalAveragePooling1D + Dense×N."""
        names = {type(l).__name__ for l in exported.layers}
        assert "GlobalAveragePooling1D" in names
        assert "SumOverParticles" not in names

    def test_exported_has_tanh_and_bounded_scale(self, exported):
        """``BoundedWeight`` is replaced by ``Activation('tanh')`` + ``Dense(1)``
        with fixed kernel/bias derived from lo/hi/normfac."""
        layer_names = [l.name for l in exported.layers]
        assert "bounded_tanh" in layer_names
        assert "bounded_scale" in layer_names

    def test_exported_input_signature_unchanged(self, source, exported):
        """The dict-style input names must match so the data loader still works."""
        src_names = {layer.name for layer in source.inputs}
        dst_names = {layer.name for layer in exported.inputs}
        assert src_names == dst_names

    def test_exported_output_shape_unchanged(self, source, exported):
        assert tuple(source.output.shape) == tuple(exported.output.shape)

    def test_numerical_equivalence_on_random_batch(self, source, exported):
        """The killer test: source and exported must produce identical
        outputs (to float-precision noise) on the same input batch."""
        rng = np.random.default_rng(123)
        mi_src = _random_inputs(rng)
        # Exported expects int32 for categoricals.
        mi_dst = {
            "continuous_inputs": mi_src["continuous_inputs"],
            "momentum_inputs":   mi_src["momentum_inputs"],
            "pdgid_inputs":      mi_src["pdgid_inputs"].astype(np.int32),
            "charge_inputs":     mi_src["charge_inputs"].astype(np.int32),
        }
        y_src = source.predict(mi_src, verbose=0)
        y_dst = exported.predict(mi_dst, verbose=0)
        np.testing.assert_allclose(y_src, y_dst, atol=1e-5, rtol=1e-4)


class TestExportWeightMinusOneModel:
    @pytest.fixture(scope="class")
    def source(self):
        return _build_weight_minus_one_source()

    @pytest.fixture(scope="class")
    def exported(self, source):
        return export_for_hls(source)

    def test_weight_minus_one_path_has_weight_shift_layer(self, exported):
        """For weight_minus_one source the export path uses ``weight_shift``
        instead of ``bounded_scale``."""
        names = [l.name for l in exported.layers]
        assert "weight_shift" in names
        assert "bounded_scale" not in names

    def test_no_shift_by_constant_in_exported(self, exported):
        """``ShiftByConstant`` is folded into the preceding Dense's bias."""
        for layer in exported.layers:
            assert type(layer).__name__ != "ShiftByConstant"

    def test_numerical_equivalence_on_random_batch(self, source, exported):
        rng = np.random.default_rng(456)
        mi_src = _random_inputs(rng)
        mi_dst = {
            **mi_src,
            "pdgid_inputs":  mi_src["pdgid_inputs"].astype(np.int32),
            "charge_inputs": mi_src["charge_inputs"].astype(np.int32),
        }
        y_src = source.predict(mi_src, verbose=0)
        y_dst = exported.predict(mi_dst, verbose=0)
        np.testing.assert_allclose(y_src, y_dst, atol=1e-5, rtol=1e-4)


class TestExportErrorPaths:
    def test_with_bias_raises_not_implemented(self):
        """``with_bias=True`` isn't supported by the export pass yet."""
        cfg = ArchConfig(
            name="test_with_bias",
            width=64, depth=3, mode=1, activation="relu",
            use_embeddings=True, binned_weight=0.0,
            with_bias=True, weight_minus_one=False,
            bounded_weight=True, use_sum=True,
            xy_balance_weight=0.0,
        )
        model = build_model(cfg)
        with pytest.raises(NotImplementedError, match="with_bias"):
            export_for_hls(model)
