"""Tests for HGQ2 → hls4ml conversion (the QAT-to-firmware path).

The HGQ2-trained model is the intended QAT route: its learned per-channel
bit-widths feed directly into hls4ml's ap_fixed precision config, giving
firmware without the precision-degradation that post-training quantization
of a float-trained model suffers. This pipeline is currently blocked by a
shape-broadcast error inside HGQ2's quantizer during hls4ml's keras_v3
parsing.

These tests anchor the failure and (when the fix lands) guarantee it stays
fixed. Marked ``xfail`` while the bug is open.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

# Skip the whole module if the toolchain isn't installed.
pytest.importorskip("hls4ml")
pytest.importorskip("hgq")

import keras  # noqa: E402
import numpy as np  # noqa: E402

from l1deepmet.quantization import build_hgq2_model  # noqa: E402


def _build_smallest_qat_model() -> keras.Model:
    """Smallest variant of the HGQ2 builder that's compatible with hls4ml.

    Note: ``use_embeddings=False`` uses zero-init dummy QDense + Add layers
    to tie the unused pdgid/charge Inputs into the graph (Keras requires it).
    hls4ml's parser doesn't handle that topology ("unexpected input layer
    chain" error), so the no-embeddings path is currently training-only.
    For conversion testing we use the smallest *with-embeddings* variant.
    """
    return build_hgq2_model(
        width=8, depth=1,
        use_embeddings=True,
        bounded_weight=False,
    )


def _build_full_qat_model() -> keras.Model:
    """The production-shaped HGQ2 model (matches the residual-ablation
    winner architecturally)."""
    return build_hgq2_model(
        width=64, depth=3,
        use_embeddings=True,
        bounded_weight=True,
    )


def _try_convert(model: keras.Model) -> Exception | None:
    """Attempt hls4ml conversion. Returns the exception on failure, ``None``
    on success."""
    import hls4ml
    try:
        config = hls4ml.utils.config_from_keras_model(
            model, granularity="name",
            default_reuse_factor=1,
            default_precision="ap_fixed<32,16>",
        )
        hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, io_type="io_parallel",
            output_dir="/tmp/hgq2_hls_test_output",
            part="xcvu13p-flga2577-2-e",
            clock_period=5, project_name="test",
        )
    except Exception as e:  # noqa: BLE001 — we want to see anything
        return e
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Failing tests (will be un-xfailed when the bug is fixed)
# ─────────────────────────────────────────────────────────────────────────────

def test_full_qat_model_converts_to_hls():
    """Production-shape HGQ2 model converts to HLS C++ without error.

    Was blocked by ``QUnaryFunctionLUT`` activations triggering a
    rank-mismatch in hls4ml's keras_v3 parser. Fixed by switching to
    ``hgq.layers.activation.Activation`` in :func:`build_hgq2_model`.
    """
    model = _build_full_qat_model()
    err = _try_convert(model)
    assert err is None, f"Conversion failed: {type(err).__name__}: {err}"


def test_smallest_qat_model_converts_to_hls():
    """Minimal HGQ2 config (no embeddings, no bounded head, single dense
    block) converts cleanly. Anchors against regressions in the bare
    QDense/QBN/activation chain."""
    model = _build_smallest_qat_model()
    err = _try_convert(model)
    assert err is None, f"Conversion failed: {type(err).__name__}: {err}"


# ─────────────────────────────────────────────────────────────────────────────
# Currently passing — diagnostic tests that document what we know.
# ─────────────────────────────────────────────────────────────────────────────

def test_hgq2_forward_pass_works_independent_of_hls_conversion():
    """The HGQ2 model itself trains and predicts fine; conversion is the
    blocker, not the model. This is the existence proof that we don't need
    to rebuild the QAT architecture."""
    model = _build_full_qat_model()
    rng = np.random.default_rng(0)
    B, N = 2, 128
    from l1deepmet.quantization import one_hot_encode_features
    pdg = rng.integers(0, 6, size=(B, N))
    chg = rng.integers(0, 4, size=(B, N))
    pdg_oh, chg_oh = one_hot_encode_features(pdg, chg)
    inputs = {
        "continuous_inputs": rng.standard_normal((B, N, 5)).astype(np.float32),
        "momentum_inputs":   rng.standard_normal((B, N, 2)).astype(np.float32),
        "pdgid_inputs":      pdg_oh,
        "charge_inputs":     chg_oh,
    }
    y = model.predict(inputs, verbose=0)
    assert y.shape == (B, 2)
    assert np.isfinite(y).all()


def test_no_embeddings_variant_does_not_convert():
    """The ``use_embeddings=False`` HGQ2 variant uses dummy zero-init QDense
    + Add layers to keep the unused pdgid/charge Inputs in the graph. hls4ml
    1.3's parser can't handle this topology and raises "unexpected input
    layer chain". Documented as a training-only path for now; if we ever
    need it on FPGA, either rework the dummy plumbing or train without
    declaring the categorical Inputs at all.
    """
    model = build_hgq2_model(width=8, depth=1, use_embeddings=False,
                              bounded_weight=False)
    err = _try_convert(model)
    assert err is not None, "no-embeddings variant unexpectedly converts now"
    assert "unexpected input layer chain" in str(err)


def test_quaryfunctionlut_still_breaks_conversion():
    """Regression guard for the original bug: ``QUnaryFunctionLUT`` is
    still incompatible with hls4ml 1.3's keras_v3 parser. If this test
    ever passes (e.g. after an HGQ2 or hls4ml upgrade), revisit the
    fix in :func:`build_hgq2_model` — we may be able to drop the
    workaround and use ``QUnaryFunctionLUT`` again.
    """
    import keras
    from keras.layers import Input
    from hgq.layers import QUnaryFunctionLUT

    x = Input(shape=(128, 8))
    h = QUnaryFunctionLUT(keras.activations.relu)(x)
    m = keras.Model(x, h)
    err = _try_convert(m)
    if err is None:
        pytest.fail(
            "QUnaryFunctionLUT now converts under hls4ml — the workaround "
            "in build_hgq2_model can probably be removed; see commit history."
        )
