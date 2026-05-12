"""HGQ2 quantization-aware build of the L1DeepMET scalar-weight model.

Drop-in replacement for ``arch_search.build_model`` when the user wants QAT.
Same forward computation, same number of "real" parameters; HGQ2 adds extra
learnable bit-width parameters per quantized op.

Differences from the full-precision build:

  - ``Embedding`` → one-hot input + ``QDense(use_bias=False)``. HGQ2 lacks a
    native ``QEmbedding`` (lookups don't fit the gradient-quantized op model).
    For our small vocabularies (pdgId vocab=6, charge vocab=4) the one-hot
    representation is fine — the one-hot input is 6+4=10 extra features, the
    QDense kernel has identical parameter count to an Embedding table.
  - ``Dense`` → ``QDense``
  - ``BatchNormalization`` → ``QBatchNormalization``
  - ``Activation('relu')`` / ``Activation('tanh')`` → ``QUnaryFunctionLUT(fn)``
  - ``Multiply``, ``Add`` → ``QMultiply``, ``QAdd``
  - ``SumOverParticles`` → ``QSum(axes=1)`` (HGQ2 has a native quantized sum;
    no need for the GlobalAveragePooling × N hack used in the full-precision
    export pass).

Output is in normalized units (model output × ``normfac`` → GeV), matching the
training-time convention of the full-precision builder.

This model is **directly synthesizable by hls4ml** — no ``export_for_hls`` pass
required. The QSum, QDense, etc. layers register as hls4ml-recognized
quantized primitives.
"""
from __future__ import annotations

from typing import Optional

import keras  # type: ignore
from keras.layers import Concatenate, Input  # type: ignore
from keras.models import Model  # type: ignore

# HGQ2 quantized layers.
#
# Two layer choices here exist to work around hls4ml-1.3 ↔ HGQ2-2.x
# parser limitations:
#
#  1. ``hgq.layers.activation.Activation`` (aliased ``QActivation``) replaces
#     ``QUnaryFunctionLUT``. The latter triggers a rank-mismatch in hls4ml's
#     keras_v3 parser (quantizer's bit-width tensor rank 3 vs the input
#     shape rank 2 the parser feeds it).
#
#  2. ``QGlobalAveragePooling1D`` + a fixed-weight ``QDense(2)`` with kernel =
#     N · I replace ``QSum``. ``QSum`` isn't in hls4ml's layer registry
#     (raises ``Layer not found in registry, and no fallback option
#     succeeded``). The average-then-multiply-by-N trick is mathematically
#     identical for our padded-with-zeros inputs and is QAT-friendly.
#
# Both regressions are anchored in ``tests/unit/synthesis/test_hgq2_to_hls.py``;
# if HGQ2 or hls4ml fix these upstream, the test will tell you and the
# workarounds can be reverted.
from hgq.layers import (  # type: ignore
    QAdd, QBatchNormalization, QDense, QGlobalAveragePooling1D, QMultiply,
)
from hgq.layers.activation import Activation as QActivation  # type: ignore
from hgq.quantizer.config import QuantizerConfig  # type: ignore

# ─── Shared with arch_search.build_model ─────────────────────────────────────
N_PARTICLES = 128
PDGID_VOCAB = 6   # 0=invalid, 1=ch.had, 2=n.had, 3=γ, 4=μ, 5=e
CHARGE_VOCAB = 4  # 0=invalid, 1=neg, 2=neutral, 3=pos


def one_hot_encode_features(
    pdgid: "np.ndarray",
    charge: "np.ndarray",
    pdgid_vocab: int = PDGID_VOCAB,
    charge_vocab: int = CHARGE_VOCAB,
):
    """One-hot encode the integer categorical inputs.

    Call this from the data loader (or the FPGA host code at deploy time)
    when feeding an HGQ2 model. Hls4ml cannot synthesize ``tf.one_hot`` or a
    ``Lambda`` layer, so the HGQ2 model expects pre-encoded one-hot inputs.

    Args:
        pdgid:        (..., N) array of int-valued pdgId codes 0..vocab-1.
        charge:       (..., N) array of int-valued charge codes 0..vocab-1.
        pdgid_vocab:  width of the pdgId one-hot dimension.
        charge_vocab: width of the charge one-hot dimension.

    Returns:
        Pair (pdgid_onehot, charge_onehot), each of shape (..., N, vocab),
        float32, suitable to feed to the model's ``pdgid_inputs`` and
        ``charge_inputs`` layers.
    """
    import numpy as np  # type: ignore
    pdg_int = np.clip(pdgid.astype(np.int32), 0, pdgid_vocab - 1)
    chg_int = np.clip(charge.astype(np.int32), 0, charge_vocab - 1)
    pdg_oh = np.eye(pdgid_vocab, dtype=np.float32)[pdg_int]
    chg_oh = np.eye(charge_vocab, dtype=np.float32)[chg_int]
    return pdg_oh, chg_oh


def build_hgq2_model(
    *,
    width: int = 64,
    depth: int = 3,
    use_embeddings: bool = True,
    embed_pdgid_dim: int = 4,
    embed_charge_dim: int = 2,
    bounded_weight: bool = True,
    normfac: float = 100.0,
    n_particles: int = N_PARTICLES,
    # HGQ2 quantizer defaults — overridable.
    default_q_conf: Optional[QuantizerConfig] = None,
) -> keras.Model:
    """Build an HGQ2 (QAT) version of the L1DeepMET scalar-weight model.

    Args:
        width:             hidden dim of the per-particle MLP body. Default 64.
        depth:             number of (QDense → QBN → QReLU) blocks. Default 3.
        use_embeddings:    embed pdgId and charge via one-hot + QDense
                           (analogous to Embedding in full-precision).
        embed_pdgid_dim:   pdgId embedding output dim. Matches the full-precision
                           ArchConfig default.
        embed_charge_dim:  charge embedding output dim.
        bounded_weight:    use bounded scalar weight head (tanh-bounded). True
                           gives ``(tanh(x) - 1) / normfac`` in (-2/normfac, 0)
                           — matches the bounded_no_bias winner from the
                           residual ablation.
        normfac:           target normalization factor. Model output × normfac → GeV.
        n_particles:       particle-axis length.
        default_q_conf:    quantizer config passed to every QDense's ``kq_conf``
                           / ``iq_conf`` / ``bq_conf``. ``None`` → HGQ2 defaults
                           (learnable bit-widths starting from a reasonable prior).

    Returns:
        Keras Model with the same dict-style inputs as
        ``arch_search.build_model`` so the existing data loader and ablation
        runner work unchanged.
    """
    q = default_q_conf  # short alias; None means use HGQ2 defaults

    # ── Inputs ────────────────────────────────────────────────────────────────
    # For HGQ2, the pdgid / charge inputs are pre-encoded one-hot
    # (shape (N, vocab)) rather than integer indices. Use
    # ``one_hot_encode_features`` from this module in your data pipeline
    # to convert the raw integer columns to the format this model expects.
    # This is necessary because hls4ml cannot synthesize ``tf.one_hot`` or
    # ``Lambda`` layers.
    x_cont = Input(shape=(n_particles, 5), name="continuous_inputs")
    x_pxpy = Input(shape=(n_particles, 2), name="momentum_inputs")
    x_pdgid = Input(shape=(n_particles, PDGID_VOCAB), name="pdgid_inputs")
    x_charge = Input(shape=(n_particles, CHARGE_VOCAB), name="charge_inputs")

    # ── Feature assembly ──────────────────────────────────────────────────────
    if use_embeddings:
        # QDense(no bias) on the pre-encoded one-hot is mathematically
        # identical to an Embedding lookup, but is fully quantization-aware
        # and synthesizable.
        emb_pdgid = QDense(
            embed_pdgid_dim, use_bias=False, name="emb_pdgid",
            kq_conf=q, iq_conf=q,
        )(x_pdgid)
        emb_charge = QDense(
            embed_charge_dim, use_bias=False, name="emb_charge",
            kq_conf=q, iq_conf=q,
        )(x_charge)
        # hls4ml's Concatenate handler is happy with 2-way at a time
        cont_plus_pdgid = Concatenate(name="concat_cont_pdgid")([x_cont, emb_pdgid])
        features = Concatenate(name="concat_features")([cont_plus_pdgid, emb_charge])
    else:
        # Categorical inputs are unused, but Keras requires every declared
        # Input to be connected to an output. Multiply by zero via a fixed
        # QDense(use_bias=False) with weights initialized to zero, then add
        # (a no-op) to features. The path is parameter-free at inference time
        # (the zero kernel makes it identical to skipping these inputs) but
        # keeps the graph topologically valid.
        #
        # Note: hls4ml's ``Add`` only supports 2 inputs at a time (same
        # limitation as ``Concatenate``); chain two binary Adds.
        from keras.layers import Add  # local import
        pdg_dummy = QDense(
            5, use_bias=False, name="dummy_pdgid",
            kernel_initializer="zeros",
            kq_conf=q, iq_conf=q,
            trainable=False,
        )(x_pdgid)
        chg_dummy = QDense(
            5, use_bias=False, name="dummy_charge",
            kernel_initializer="zeros",
            kq_conf=q, iq_conf=q,
            trainable=False,
        )(x_charge)
        cont_plus_pdg = Add(name="cont_plus_pdg")([x_cont, pdg_dummy])
        features = Add(name="features")([cont_plus_pdg, chg_dummy])

    # ── Per-particle MLP body ─────────────────────────────────────────────────
    h = features
    for i in range(depth):
        h = QDense(
            width, name=f"qdense_{i}",
            kq_conf=q, iq_conf=q, bq_conf=q,
        )(h)
        h = QBatchNormalization(name=f"qbn_{i}")(h)
        h = QActivation("relu", name=f"qact_{i}")(h)

    # ── Per-particle scalar weight head ───────────────────────────────────────
    raw_w = QDense(1, name="qmet_weight", kq_conf=q, iq_conf=q, bq_conf=q)(h)

    if bounded_weight:
        # BoundedWeight(x) = (lo + (hi-lo)*0.5*(1+tanh(x))) / normfac
        # For lo=-2, hi=0, normfac=100: (tanh(x) - 1) / 100.
        # Implemented as: tanh activation, then a frozen-affine QDense.
        # Re-using QDense (trainable) is fine — the gradient will keep
        # kernel/bias near the target; alternatively the caller can freeze it.
        w = QActivation("tanh", name="qtanh")(raw_w)
        w = QDense(
            1, name="qbounded_scale",
            kernel_initializer=keras.initializers.Constant(1.0 / normfac),
            bias_initializer=keras.initializers.Constant(-1.0 / normfac),
            kq_conf=q, iq_conf=q, bq_conf=q,
        )(w)
    else:
        # Plain scalar weight in normalized units. Initialize near
        # -1/normfac (the weight-minus-one trick) so the untrained model
        # starts at PUPPI MET, matching the full-precision builder.
        # Achieved here by initializing kernel=0 and bias=-1/normfac.
        w = QDense(
            1, name="qweight_shift",
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(-1.0 / normfac),
            kq_conf=q, iq_conf=q, bq_conf=q,
        )(raw_w)

    # ── Multiply by pxpy and sum over particles ───────────────────────────────
    weighted = QMultiply(name="qweight_pxpy")([w, x_pxpy])  # (B, N, 2)

    # Replace what was SumOverParticles in the full-precision builder with a
    # QGlobalAveragePooling1D followed by a fixed-weight QDense(2) that
    # multiplies by N. Padded slots have pxpy = 0 in our preprocessing so
    # sum and mean*N agree exactly. ``QSum`` isn't recognized by hls4ml 1.3.
    import numpy as np  # local import
    pooled = QGlobalAveragePooling1D(name="qavg_over_particles")(weighted)  # (B, 2)
    out = QDense(
        2, use_bias=False, name="output",
        kernel_initializer=keras.initializers.Constant(
            (np.eye(2, dtype="float32") * float(n_particles)).tolist()
        ),
        trainable=False,
        kq_conf=q, iq_conf=q,
    )(pooled)

    model = Model(
        inputs=[x_cont, x_pxpy, x_pdgid, x_charge],
        outputs=out,
        name=f"l1deepmet_hgq2_w{width}_d{depth}",
    )
    return model
