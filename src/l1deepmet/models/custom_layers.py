"""Custom Keras layers used by the full-precision L1DeepMET models.

Promoted verbatim from ``scripts/arch_search.py`` so that library modules
(synthesis, estimation) can reference them without importing from
``scripts/``. All classes are registered under the ``"l1deepmet"``
serialization package, so previously saved ``.keras`` checkpoints load
unchanged — the registry key ``"l1deepmet>Name"`` does not depend on the
defining module. ``arch_search`` re-imports these names; do not redefine
them elsewhere or the duplicate registration will shadow one class object.
"""
from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="l1deepmet")
class CastToInt(tf.keras.layers.Layer):
    """Cast input to int32 — Keras 3 compatible."""
    def call(self, x):
        return tf.cast(x, tf.int32)


@tf.keras.utils.register_keras_serializable(package="l1deepmet")
class ZeroReduce(tf.keras.layers.Layer):
    """Reduce input to zeros of shape (B, dim) — dummy connection for unused inputs."""
    def __init__(self, output_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.zeros((batch_size, self.output_dim), dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config["output_dim"] = self.output_dim
        return config


@tf.keras.utils.register_keras_serializable(package="l1deepmet")
class ShiftByConstant(tf.keras.layers.Layer):
    """Add a fixed constant to input. Used for weight-minus-one trick.

    When initialized with shift=-1.0, the per-particle weight starts near -1
    (since Dense init produces values near 0), so the initial MET prediction
    is approximately -sum(-1 * px_i) = sum(px_i), which is the raw PF MET.
    The network then learns small corrections from this baseline.
    """
    def __init__(self, shift=-1.0, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift

    def call(self, x):
        return x + self.shift

    def get_config(self):
        config = super().get_config()
        config["shift"] = self.shift
        return config


@tf.keras.utils.register_keras_serializable(package="l1deepmet")
class PaddingMask(tf.keras.layers.Layer):
    """Build a (B, 1, N) attention mask from a (B, N) pdgid tensor.

    Mask is True where pdgid != 0 (real particles); False on padding slots.
    Adds an identity-diagonal OR so no row is fully masked (which would
    NaN MHA). Output shape (B, 1, N) broadcasts across queries in MHA.
    """
    def __init__(self, n_particles: int, **kwargs):
        super().__init__(**kwargs)
        self.n_particles = n_particles

    def call(self, pdgid):
        valid = tf.not_equal(tf.cast(pdgid, tf.int32), 0)        # (B, N) bool
        attn_mask = valid[:, None, :]                            # (B, 1, N)
        # Identity-diagonal fallback so every query attends to at least itself.
        self_mask = tf.eye(self.n_particles, dtype=tf.bool)[None, :, :]  # (1, N, N)
        return tf.logical_or(attn_mask, self_mask)

    def get_config(self):
        config = super().get_config()
        config["n_particles"] = self.n_particles
        return config


@tf.keras.utils.register_keras_serializable(package="l1deepmet")
class BoundedWeight(tf.keras.layers.Layer):
    """Map a raw Dense output to a bounded per-particle weight (in normalized
    units). Default range gives effective weight ∈ (-2, 0) with init at -1
    when raw=0 — matching the weight_minus_one convention so untrained model
    output equals PUPPI MET.

    effective_w = lo + (hi - lo) * (1 + tanh(raw)) / 2
    stored_w    = effective_w / normfac

    The "effective weight" is the multiplier on pxpy you'd see in the MET
    sum after un-normalization. With lo=-2, hi=0 → range (-2, 0), init -1.
    """
    def __init__(self, lo=-2.0, hi=0.0, normfac=100.0, **kwargs):
        super().__init__(**kwargs)
        self.lo, self.hi, self.normfac = lo, hi, normfac

    def call(self, x):
        eff = self.lo + (self.hi - self.lo) * 0.5 * (1.0 + tf.tanh(x))
        return eff / self.normfac

    def get_config(self):
        config = super().get_config()
        config.update({"lo": self.lo, "hi": self.hi, "normfac": self.normfac})
        return config


@tf.keras.utils.register_keras_serializable(package="l1deepmet")
class SumOverParticles(tf.keras.layers.Layer):
    """Sum over the particle axis (axis=1). Drop-in replacement for
    GlobalAveragePooling1D that doesn't divide by N."""
    def call(self, x):
        return tf.reduce_sum(x, axis=1)

    def get_config(self):
        return super().get_config()
