"""Unit tests for ``l1deepmet.synthesis.config``."""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

pytest.importorskip("hls4ml")

import keras  # noqa: E402
from keras.layers import Dense, Embedding, Input  # noqa: E402
from keras.models import Model  # noqa: E402

from l1deepmet.synthesis.config import (  # noqa: E402
    CATEGORICAL_INPUT_NAMES,
    CATEGORICAL_PRECISION,
    DEFAULT_PART,
    DEFAULT_PRECISION,
    build_hls_config,
)


def _toy_model() -> keras.Model:
    """A tiny model exercising the L1DeepMET-flavored input layout
    (categorical pdgid + categorical charge + continuous + momentum), without
    depending on the real ``arch_search`` builder."""
    x_cont = Input(shape=(128, 5), name="continuous_inputs")
    x_pxpy = Input(shape=(128, 2), name="momentum_inputs")
    x_pdgid = Input(shape=(128,), dtype="int32", name="pdgid_inputs")
    x_charge = Input(shape=(128,), dtype="int32", name="charge_inputs")

    e_pdg = Embedding(input_dim=6, output_dim=4, name="emb_pdgid")(x_pdgid)
    e_chg = Embedding(input_dim=4, output_dim=2, name="emb_charge")(x_charge)
    # Use a tiny dense to keep the graph compilable.
    h = keras.layers.Concatenate()([x_cont, e_pdg, e_chg])
    h = Dense(4, name="dense_0")(h)
    out = keras.layers.GlobalAveragePooling1D(name="out_pool")(h)
    out = Dense(2, name="out")(out)
    # Tie the unused momentum_inputs into the graph so Keras accepts it.
    out = keras.layers.Add(name="tie")([
        out, keras.layers.GlobalAveragePooling1D()(Dense(2, name="mom_proj")(x_pxpy))
    ])
    return Model(inputs=[x_cont, x_pxpy, x_pdgid, x_charge], outputs=out)


def test_config_has_top_level_strategy_and_layer_names():
    cfg = build_hls_config(_toy_model())
    assert "Model" in cfg and "Strategy" in cfg["Model"]
    assert cfg["Model"]["Strategy"] == "Latency"
    assert "LayerName" in cfg
    # The four named inputs should all appear as LayerName entries.
    for name in ("continuous_inputs", "momentum_inputs",
                  "pdgid_inputs", "charge_inputs"):
        assert name in cfg["LayerName"], f"missing layer {name!r} in config"


def test_categorical_inputs_get_tight_precision():
    cfg = build_hls_config(_toy_model())
    for cat in CATEGORICAL_INPUT_NAMES:
        assert cat in cfg["LayerName"]
        assert cfg["LayerName"][cat]["Precision"]["result"] == CATEGORICAL_PRECISION


def test_tracing_enabled_by_default():
    cfg = build_hls_config(_toy_model())
    for ln in cfg["LayerName"].values():
        assert ln.get("Trace") is True


def test_tracing_can_be_disabled():
    cfg = build_hls_config(_toy_model(), trace_layers=False)
    for ln in cfg["LayerName"].values():
        # Some hls4ml versions may not set Trace at all when disabled;
        # either missing or False both indicate "not tracing".
        assert ln.get("Trace") in (False, None)


def test_strategy_override():
    cfg = build_hls_config(_toy_model(), strategy="Resource")
    assert cfg["Model"]["Strategy"] == "Resource"


def test_default_precision_recorded_at_model_level():
    """``default_precision`` ends up in ``cfg["Model"]["Precision"]["default"]``;
    individual layers get ``'auto'`` and hls4ml falls back to the model default.
    This documents the hls4ml 1.3 behavior so changes here are caught."""
    cfg = build_hls_config(_toy_model(), default_precision="ap_fixed<24,8>")
    assert cfg["Model"]["Precision"]["default"] == "ap_fixed<24,8>"


def test_per_layer_override_wins_over_default():
    """If a per-layer precision is supplied, it should appear on that layer
    instead of the auto-fallback."""
    cfg = build_hls_config(
        _toy_model(),
        default_precision="ap_fixed<24,8>",
        layer_precisions={"dense_0": "ap_fixed<48,20>"},
    )
    assert cfg["LayerName"]["dense_0"]["Precision"]["result"] == "ap_fixed<48,20>"


def test_per_layer_overrides_apply():
    cfg = build_hls_config(_toy_model(),
                            layer_precisions={"dense_0": "ap_fixed<48,20>"})
    assert cfg["LayerName"]["dense_0"]["Precision"]["result"] == "ap_fixed<48,20>"


def test_per_layer_override_for_unknown_layer_raises():
    with pytest.raises(KeyError, match="not in model"):
        build_hls_config(_toy_model(),
                        layer_precisions={"definitely_not_a_layer": "ap_fixed<32,16>"})


def test_defaults_constants_match_l1_phase2():
    """Sanity-check that the documented CMS L1 Phase-2 defaults haven't
    drifted. Changing these would silently change the synthesis target."""
    assert DEFAULT_PART == "xcvu13p-flga2577-2-e"   # CMS L1 Phase-2 VU13P
    assert DEFAULT_PRECISION == "ap_fixed<32,16>"
