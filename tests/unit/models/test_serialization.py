"""Tests for ``l1deepmet.models.custom_layers`` and
``l1deepmet.models.serialization``.

The six custom full-precision layers historically lived in
``scripts/arch_search.py`` and the custom-objects loader in
``scripts/synthesize.py``; both are promoted into the library so that
``src/`` modules (synthesis, estimation) never import from ``scripts/``.
``arch_search`` re-imports the layer classes, so the registered class
object must be *identical* through either import path — Keras
serialization registers by ``"l1deepmet>Name"`` and a duplicate
definition would shadow it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from l1deepmet.models.custom_layers import (  # noqa: E402
    BoundedWeight,
    CastToInt,
    PaddingMask,
    ShiftByConstant,
    SumOverParticles,
    ZeroReduce,
)
from l1deepmet.models.serialization import (  # noqa: E402
    get_custom_objects,
    is_hgq2_model,
    load_any_model,
)


class TestCustomLayersModule:
    def test_all_six_layers_importable(self):
        for cls in (BoundedWeight, CastToInt, PaddingMask, ShiftByConstant,
                    SumOverParticles, ZeroReduce):
            assert isinstance(cls, type)

    def test_arch_search_reexports_identical_classes(self):
        """``from arch_search import CastToInt`` must be the *same object*
        as the library class — a re-definition would double-register the
        Keras serialization key and shadow one of the two."""
        import arch_search

        assert arch_search.CastToInt is CastToInt
        assert arch_search.ZeroReduce is ZeroReduce
        assert arch_search.ShiftByConstant is ShiftByConstant
        assert arch_search.PaddingMask is PaddingMask
        assert arch_search.BoundedWeight is BoundedWeight
        assert arch_search.SumOverParticles is SumOverParticles


class TestGetCustomObjects:
    def test_covers_full_precision_layers(self):
        co = get_custom_objects()
        expected = {"CastToInt", "BoundedWeight", "ShiftByConstant",
                    "SumOverParticles", "ZeroReduce", "PaddingMask",
                    "CorrectedCompositeLoss"}
        missing = expected - set(co)
        assert not missing, f"get_custom_objects missing: {missing}"

    def test_covers_hgq2_layers(self):
        pytest.importorskip("hgq")
        co = get_custom_objects()
        expected = {"QDense", "QBatchNormalization", "QMultiply",
                    "QGlobalAveragePooling1D"}
        missing = expected - set(co)
        assert not missing, f"get_custom_objects missing HGQ2 layers: {missing}"

    def test_synthesize_shim_returns_same_mapping(self):
        """``scripts/synthesize.py::load_custom_objects`` stays as a shim for
        existing callers and must delegate to the library implementation."""
        from synthesize import load_custom_objects

        assert set(load_custom_objects()) == set(get_custom_objects())


class TestLoadAnyModel:
    @pytest.fixture
    def tiny_fp32_model(self):
        """Minimal functional model exercising two custom layers the way
        the scalar-weight head does: Dense(1) -> BoundedWeight ->
        Multiply with pxpy -> SumOverParticles."""
        import keras

        rng = np.random.default_rng(42)
        pxpy = keras.Input(shape=(128, 2), name="momentum_inputs")
        raw = keras.layers.Dense(1, name="met_weight")(pxpy)
        w = BoundedWeight(name="bounded")(raw)
        weighted = keras.layers.Multiply(name="weighted")([w, pxpy])
        out = SumOverParticles(name="met")(weighted)
        model = keras.Model(pxpy, out)
        batch = rng.standard_normal((32, 128, 2)).astype(np.float32)
        return model, batch

    def test_round_trip_identical_predictions(self, tmp_path, tiny_fp32_model):
        model, batch = tiny_fp32_model
        path = tmp_path / "tiny.keras"
        model.save(path)

        restored = load_any_model(path)
        assert restored.count_params() == model.count_params()
        np.testing.assert_allclose(
            model.predict(batch, verbose=0),
            restored.predict(batch, verbose=0),
            atol=1e-6,
        )

    def test_load_uncompiled_by_default(self, tmp_path, tiny_fp32_model):
        model, _ = tiny_fp32_model
        path = tmp_path / "tiny.keras"
        model.save(path)
        restored = load_any_model(path)
        # Keras 3: an uncompiled model has no optimizer (attribute absent).
        assert getattr(restored, "optimizer", None) is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((OSError, ValueError)):
            load_any_model(tmp_path / "does_not_exist.keras")


class TestIsHgq2Model:
    def test_false_for_plain_keras_model(self):
        import keras

        inp = keras.Input(shape=(4,))
        model = keras.Model(inp, keras.layers.Dense(2)(inp))
        assert is_hgq2_model(model) is False

    def test_true_for_hgq2_model(self):
        pytest.importorskip("hgq")
        from l1deepmet.quantization import build_hgq2_model

        model = build_hgq2_model(width=8, depth=1, use_embeddings=True,
                                 bounded_weight=True)
        assert is_hgq2_model(model) is True
