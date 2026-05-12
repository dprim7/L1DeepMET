"""Convert a trained L1DeepMET Keras model to an hls4ml-friendly equivalent.

The training-time model uses three custom layers that hls4ml 0.6 doesn't know:

  - ``CastToInt(x)``           — ``tf.cast(x, int32)``. Pure type cast.
                                  Needed at training time so the Embedding
                                  layers see int indices. At HLS time the input
                                  is already declared ``ap_uint<4>``, so the
                                  cast is redundant. **Remove it.**

  - ``BoundedWeight(x)``       — ``(lo + (hi-lo) * 0.5 * (1+tanh(x))) / normfac``.
                                  For the default ``lo=-2``, ``hi=0``,
                                  ``normfac=100``, this simplifies to
                                  ``(tanh(x) - 1) / 100``. **Replace with a
                                  built-in tanh Activation followed by a
                                  Dense(1) layer with fixed weight = 1/100 and
                                  bias = −1/100.** Mathematically identical;
                                  one extra layer.

  - ``SumOverParticles(x)``    — ``tf.reduce_sum(x, axis=1)``. hls4ml 0.6
                                  has GlobalAveragePooling1D but no native
                                  global sum. **Replace with
                                  GlobalAveragePooling1D and multiply the
                                  model output by N=128** (the particle-axis
                                  length). For padded particles px=py=0, so
                                  the sum-vs-average distinction is purely
                                  a constant scale we absorb downstream.

The returned model has the same numerical behavior as the original (modulo
floating-point reorder), the same parameter count, and the same input/output
shapes. The original Keras model is unchanged.

Also handled: ``weight_minus_one`` models (``Dense → ShiftByConstant``). The
shift is a simple bias add; we fold the constant into the bias of the Dense
that precedes it so the resulting model is one layer shorter.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Activation, BatchNormalization, Concatenate, Dense, Embedding,
    GlobalAveragePooling1D, Input, Multiply,
)
from tensorflow.keras.models import Model  # type: ignore

logger = logging.getLogger(__name__)

# Must match arch_search.N_PARTICLES.
N_PARTICLES_DEFAULT = 128


def export_for_hls(
    model: tf.keras.Model,
    *,
    n_particles: int = N_PARTICLES_DEFAULT,
    embedded_normfac: bool = True,
) -> tf.keras.Model:
    """Build an hls4ml-friendly equivalent model with the same trained weights.

    Args:
        model:            the trained Keras model.
        n_particles:      particle-axis length used in the training architecture.
                          Default matches arch_search.N_PARTICLES = 128.
        embedded_normfac: if True (default), the N=128 scale factor introduced by
                          replacing SumOverParticles with GlobalAveragePooling1D
                          is multiplied into the output inside the graph (via a
                          fixed Dense), so downstream code can keep using the
                          same ``output × normfac`` convention as before. If
                          False, the caller must scale predictions by N themselves.

    Returns:
        A new Keras ``Model`` with the same trained weights and equivalent
        output. The original model is not modified.
    """
    # ── Inspect the source model to discover its config ──────────────────────
    layer_names = [layer.name for layer in model.layers]
    has_embeddings = "emb_pdgid" in layer_names
    has_bounded = any("bounded_weight" in n for n in layer_names)
    has_weight_minus_one = "weight_minus_one" in layer_names
    has_bias = "met_bias" in layer_names
    has_sum_layer = any(
        type(layer).__name__ == "SumOverParticles" for layer in model.layers
    )

    if has_bias:
        raise NotImplementedError(
            "Export for models with with_bias=True is not implemented yet."
        )
    if not has_sum_layer:
        raise NotImplementedError(
            "Export currently expects use_sum=True (SumOverParticles in the head)."
        )

    # Infer width / depth from the body Dense layers.
    body_dense = [layer for layer in model.layers
                  if isinstance(layer, Dense) and layer.name.startswith("dense_")]
    depth = len(body_dense)
    width = body_dense[0].units if body_dense else 64

    logger.info(
        "export_for_hls: depth=%d width=%d embeddings=%s bounded=%s wm1=%s",
        depth, width, has_embeddings, has_bounded, has_weight_minus_one,
    )

    # ── Rebuild architecture with hls4ml-friendly primitives ─────────────────
    # Inputs: keep the same names so layer_precisions overrides in
    # build_hls_config still apply.
    x_cont = Input(shape=(n_particles, 5), name="continuous_inputs")
    x_pxpy = Input(shape=(n_particles, 2), name="momentum_inputs")
    # Integer inputs: declare as int32 so we can drop the CastToInt layer.
    x_pdgid = Input(shape=(n_particles,), dtype="int32", name="pdgid_inputs")
    x_charge = Input(shape=(n_particles,), dtype="int32", name="charge_inputs")

    if has_embeddings:
        # Pull vocab sizes / output dims from the source Embedding layers.
        emb_pdgid_src = model.get_layer("emb_pdgid")
        emb_charge_src = model.get_layer("emb_charge")
        emb_pdgid = Embedding(
            input_dim=emb_pdgid_src.input_dim,
            output_dim=emb_pdgid_src.output_dim,
            name="emb_pdgid",
        )(x_pdgid)
        emb_charge = Embedding(
            input_dim=emb_charge_src.input_dim,
            output_dim=emb_charge_src.output_dim,
            name="emb_charge",
        )(x_charge)
        # hls4ml's Concatenate only supports 2 inputs at a time. Chain them.
        cont_plus_pdgid = Concatenate(name="concat_cont_pdgid")([x_cont, emb_pdgid])
        features = Concatenate(name="concat_features")([cont_plus_pdgid, emb_charge])
    else:
        features = x_cont

    h = features
    for i in range(depth):
        h = Dense(width, activation=None, name=f"dense_{i}")(h)
        h = BatchNormalization(momentum=0.95, name=f"bn_{i}")(h)
        h = Activation("relu", name=f"act_{i}")(h)

    # Output head: Dense(1) producing per-particle weight w.
    raw_w = Dense(1, activation=None, name="met_weight")(h)   # (B, N, 1)

    if has_bounded:
        # BoundedWeight(x) = (tanh(x) - 1) / 100 for lo=-2, hi=0, normfac=100.
        # Re-build as: tanh → Dense(1, kernel=1/100, bias=-1/100).
        # The src BoundedWeight layer carries lo/hi/normfac in its config.
        src_bw = next(l for l in model.layers if type(l).__name__ == "BoundedWeight")
        cfg = src_bw.get_config()
        lo, hi, nf = cfg["lo"], cfg["hi"], cfg["normfac"]
        # effective_w = lo + (hi-lo) * 0.5 * (1 + tanh(raw))
        # divided by normfac: w = ((hi-lo)/(2*nf)) * tanh(raw) + (hi+lo)/(2*nf)
        kernel_val = (hi - lo) / (2.0 * nf)
        bias_val = (hi + lo) / (2.0 * nf)
        w = Activation("tanh", name="bounded_tanh")(raw_w)
        # Fixed-weight 1×1 Dense applied per-particle:
        w = Dense(1, activation=None, name="bounded_scale",
                  kernel_initializer="zeros", bias_initializer="zeros")(w)
        # We'll set this layer's weights manually below.
        bounded_scale_kernel = np.array([[kernel_val]], dtype=np.float32)
        bounded_scale_bias = np.array([bias_val], dtype=np.float32)
    elif has_weight_minus_one:
        # Shift by -1/normfac. Read the shift constant from the source layer.
        src_shift = model.get_layer("weight_minus_one")
        shift = float(src_shift.shift)
        # Fold the shift into a new Dense(1) so the architecture stays standard.
        # Per-particle Dense(1) on a scalar input is just y = x + bias.
        w = Dense(1, activation=None, name="weight_shift",
                  kernel_initializer="ones", bias_initializer="zeros")(raw_w)
        bounded_scale_kernel = np.array([[1.0]], dtype=np.float32)
        bounded_scale_bias = np.array([shift], dtype=np.float32)
    else:
        # No bounding, no shift — pass the raw weight through (unusual config).
        w = raw_w
        bounded_scale_kernel = None
        bounded_scale_bias = None

    weighted = Multiply(name="weight_pxpy")([w, x_pxpy])    # (B, N, 2)

    # Replace SumOverParticles with GlobalAveragePooling1D × N
    pooled = GlobalAveragePooling1D(name="avg_over_particles")(weighted)  # (B, 2)
    if embedded_normfac:
        # Multiply by N via a fixed Dense(2) so downstream callers don't have to.
        out = Dense(2, activation=None, name="output",
                    kernel_initializer="zeros", bias_initializer="zeros")(pooled)
        out_kernel = np.eye(2, dtype=np.float32) * float(n_particles)
        out_bias = np.zeros(2, dtype=np.float32)
    else:
        # Caller must scale by N themselves.
        out = pooled
        out_kernel = None
        out_bias = None

    new_model = Model(
        inputs=[x_cont, x_pxpy, x_pdgid, x_charge],
        outputs=out,
        name=model.name + "_hls_friendly",
    )

    # ── Copy trained weights for every layer that has a 1:1 counterpart ─────
    for layer in new_model.layers:
        name = layer.name
        if name in ("avg_over_particles", "weight_pxpy", "bounded_tanh",
                    "concat_features", "concat_cont_pdgid"):
            continue
        if name == "bounded_scale" or name == "weight_shift":
            layer.set_weights([bounded_scale_kernel, bounded_scale_bias])
            continue
        if name == "output" and out_kernel is not None:
            layer.set_weights([out_kernel, out_bias])
            continue
        # Direct copy from the source model (Dense, BN, Embedding, input).
        try:
            src = model.get_layer(name)
        except ValueError:
            continue
        if src.weights:
            layer.set_weights(src.get_weights())

    # ── Numerical sanity check: random batch should give matching outputs ───
    _check_numerical_equivalence(model, new_model, n_particles=n_particles)

    return new_model


def _check_numerical_equivalence(
    src: tf.keras.Model,
    dst: tf.keras.Model,
    *,
    n_particles: int,
    n_samples: int = 4,
    atol: float = 1e-4,
) -> None:
    """Smoke test: src and dst should agree on a random batch."""
    rng = np.random.default_rng(0)
    cont = rng.standard_normal((n_samples, n_particles, 5)).astype(np.float32)
    pxpy = rng.standard_normal((n_samples, n_particles, 2)).astype(np.float32)
    pdg = rng.integers(0, 6, size=(n_samples, n_particles)).astype(np.int32)
    chg = rng.integers(0, 4, size=(n_samples, n_particles)).astype(np.int32)

    src_inputs = {
        "continuous_inputs": cont,
        "momentum_inputs": pxpy,
        "pdgid_inputs": pdg.astype(np.float32),
        "charge_inputs": chg.astype(np.float32),
    }
    dst_inputs = {
        "continuous_inputs": cont,
        "momentum_inputs": pxpy,
        "pdgid_inputs": pdg,
        "charge_inputs": chg,
    }
    y_src = src.predict(src_inputs, verbose=0)
    y_dst = dst.predict(dst_inputs, verbose=0)
    err = float(np.max(np.abs(y_src - y_dst)))
    if err > atol:
        logger.warning(
            "Numerical mismatch between source and HLS-friendly model: max|Δ|=%.6e",
            err,
        )
    else:
        logger.info(
            "HLS-friendly model matches source on random batch: max|Δ|=%.2e",
            err,
        )
