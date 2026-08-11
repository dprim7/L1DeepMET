"""Tests for ``l1deepmet.estimation.ebops`` — HGQ2 EBOPs collection.

EBOPs (effective bit operations, ~ bits_in x bits_kernel accumulated over
the multiply structure) live in a non-trainable ``uint32`` Keras weight on
each Q-layer, populated only when the model is called with a truthy /
``'tracing'`` training flag. Consequences pinned here:

- a freshly *built* model reads all-zero;
- ``recompute_ebops`` (hgq's ``trace_minmax``) populates it without
  touching BatchNorm moving statistics;
- the values survive save/load, so checkpoints report their trained EBOPs
  with no forward pass.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

pytest.importorskip("hgq")

import keras  # noqa: E402

from l1deepmet.estimation.ebops import (  # noqa: E402
    collect_ebops,
    recompute_ebops,
    synthetic_calibration_batch,
)
from l1deepmet.models.serialization import load_any_model  # noqa: E402
from l1deepmet.quantization import build_hgq2_model  # noqa: E402


@pytest.fixture()
def tiny_hgq2_model():
    return build_hgq2_model(width=8, depth=1, use_embeddings=True,
                            bounded_weight=True)


class TestCollectEbops:
    def test_fresh_model_is_all_zero(self, tiny_hgq2_model):
        report = collect_ebops(tiny_hgq2_model)
        assert report.total == 0
        assert report.all_zero is True
        assert report.n_q_layers > 0
        assert report.source == "saved"
        names = [row.layer for row in report.per_layer]
        assert "qdense_0" in names

    def test_plain_keras_model_has_no_q_layers(self):
        inp = keras.Input(shape=(4,))
        model = keras.Model(inp, keras.layers.Dense(2)(inp))
        report = collect_ebops(model)
        assert report.n_q_layers == 0
        assert report.total == 0
        assert report.per_layer == []

    def test_to_dict_json_shape(self, tiny_hgq2_model):
        import json

        payload = json.dumps(collect_ebops(tiny_hgq2_model).to_dict())
        data = json.loads(payload)
        assert set(data) >= {"per_layer", "total", "n_q_layers", "all_zero",
                             "source"}


class TestRecomputeEbops:
    def test_recompute_populates_ebops(self, tiny_hgq2_model):
        batch = synthetic_calibration_batch(tiny_hgq2_model, n=32, seed=42)
        report = recompute_ebops(tiny_hgq2_model, batch, batch_size=16)
        assert report.source == "recomputed"
        assert report.total > 0
        assert report.all_zero is False
        by_name = {row.layer: row.ebops for row in report.per_layer}
        assert by_name["qdense_0"] > 0

    def test_recompute_does_not_touch_bn_moving_stats(self, tiny_hgq2_model):
        bn = tiny_hgq2_model.get_layer("qbn_0")
        mean_before = np.array(bn.moving_mean)
        var_before = np.array(bn.moving_variance)
        batch = synthetic_calibration_batch(tiny_hgq2_model, n=32, seed=42)
        recompute_ebops(tiny_hgq2_model, batch)
        np.testing.assert_array_equal(mean_before, np.array(bn.moving_mean))
        np.testing.assert_array_equal(var_before, np.array(bn.moving_variance))

    def test_ebops_survive_save_load(self, tiny_hgq2_model, tmp_path):
        batch = synthetic_calibration_batch(tiny_hgq2_model, n=32, seed=42)
        before = recompute_ebops(tiny_hgq2_model, batch)
        path = tmp_path / "traced.keras"
        tiny_hgq2_model.save(path)

        restored = load_any_model(path)
        after = collect_ebops(restored)
        assert after.source == "saved"
        assert after.total == before.total
        assert {r.layer: r.ebops for r in after.per_layer} == \
               {r.layer: r.ebops for r in before.per_layer}


class TestSyntheticCalibrationBatch:
    def test_shapes_match_model_inputs(self, tiny_hgq2_model):
        n = 16
        batch = synthetic_calibration_batch(tiny_hgq2_model, n=n)
        assert len(batch) == len(tiny_hgq2_model.inputs)
        for arr, tensor in zip(batch, tiny_hgq2_model.inputs):
            assert arr.shape == tuple([n] + [int(d) for d in tensor.shape[1:]])
            assert arr.dtype == np.float32

    def test_categorical_inputs_are_one_hot(self, tiny_hgq2_model):
        batch = synthetic_calibration_batch(tiny_hgq2_model, n=8)
        by_name = {t.name.split(":")[0]: a
                   for t, a in zip(tiny_hgq2_model.inputs, batch)}
        for key in ("pdgid_inputs", "charge_inputs"):
            arr = by_name[key]
            np.testing.assert_array_equal(arr.sum(axis=-1),
                                          np.ones(arr.shape[:-1]))
            assert set(np.unique(arr)) <= {0.0, 1.0}

    def test_deterministic_for_fixed_seed(self, tiny_hgq2_model):
        a = synthetic_calibration_batch(tiny_hgq2_model, n=4, seed=7)
        b = synthetic_calibration_batch(tiny_hgq2_model, n=4, seed=7)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
