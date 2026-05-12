# L1DeepMET quantization

Quantization-aware training (QAT) infrastructure for L1DeepMET. Currently
supports the **HGQ2** (Heterogeneous Gradient-based Quantization v2) path —
quantization bit-widths are learned per-parameter via gradient descent jointly
with the weights, then exported to hls4ml.

The motivation: post-training fixed-point conversion (the
``scripts/synthesize.py`` path on a float-trained model) showed ~2× resolution
degradation because the network's float weights were never trained to be
robust to limited precision. QAT trains the model with the precision
constraints baked into the loss surface, so the converted FPGA model has the
same physics performance as the simulator.

## Status

| Capability                                | Status      |
|-------------------------------------------|-------------|
| Build HGQ2 model with our architecture    | ✅ working  |
| Forward pass                              | ✅ working  |
| Training (``model.fit``)                  | ✅ working  |
| hls4ml conversion (with embeddings)       | ✅ working  |
| hls4ml conversion (no embeddings)         | ❌ blocked (parser topology issue) |
| Integration with ``scripts/ablation.py``  | 🚧 pending  |
| QKeras (alternative manual-bit-width QAT) | 🚧 not implemented |

## Quick start

```python
from l1deepmet.quantization import build_hgq2_model, one_hot_encode_features

# Build a QAT model with the same shape as scalar_xybal10
model = build_hgq2_model(
    width=64, depth=3,
    use_embeddings=True,
    bounded_weight=True,   # tanh-bounded output head (matches residual ablation winner)
)

# Note: HGQ2 model expects ONE-HOT pdgid / charge inputs (hls4ml cannot
# synthesize tf.one_hot / Lambda). Encode them yourself in the data loader:
pdgid_oh, charge_oh = one_hot_encode_features(pdgid_int, charge_int)

inputs = {
    "continuous_inputs": cont_features,          # (B, 128, 5)
    "momentum_inputs":   pxpy,                   # (B, 128, 2)
    "pdgid_inputs":      pdgid_oh,               # (B, 128, 6) — pre-encoded one-hot
    "charge_inputs":     charge_oh,              # (B, 128, 4) — pre-encoded one-hot
}

model.compile(optimizer="adam", loss=...)
model.fit(inputs, targets, epochs=30)
```

Trained model can be saved with ``model.save("hgq2_model.keras")`` and reloaded
with the standard HGQ2 custom-objects registration.

## Architectural choices

| Component                  | Full-precision (arch_search) | HGQ2 (this module) |
|----------------------------|------------------------------|--------------------|
| Categorical inputs         | int + ``Embedding``          | pre-encoded one-hot + ``QDense(use_bias=False)`` |
| Per-particle Dense         | ``Dense``                    | ``QDense``         |
| Normalization              | ``BatchNormalization``       | ``QBatchNormalization`` |
| ReLU / tanh                | ``Activation('relu'/'tanh')``| ``QUnaryFunctionLUT(activations.relu / .tanh)`` |
| Multiply (weight × pxpy)   | ``Multiply``                 | ``QMultiply``      |
| Sum over particles         | custom ``SumOverParticles``  | ``QGlobalAveragePooling1D`` + ``QDense(2, kernel=N·I)`` (``QSum`` unsupported by hls4ml 1.3) |
| Bounded weight head        | custom ``BoundedWeight``     | ``QUnaryFunctionLUT(tanh)`` + ``QDense(1)`` with fixed init |

The HGQ2 model **does not need the** ``export_for_hls`` **rewrite step** — every
layer is already an hls4ml-recognized quantized primitive, modulo the
conversion bug below.

## Workarounds applied to make hls4ml conversion work

hls4ml 1.3 has two compatibility gaps with HGQ2 layers; we route around both:

1. **``QUnaryFunctionLUT`` triggers a rank mismatch.** The original
   ``InvalidArgumentError: Rank of input (3) must be no greater than rank of
   output shape (2)`` from inside ``FixedPointQuantizer.bw_to_x``. Use
   ``hgq.layers.activation.Activation`` (aliased ``QActivation``) instead —
   it doesn't go through that code path. Verified for both ReLU and tanh.

2. **``QSum`` is not in hls4ml's layer registry.** Raises ``Layer not found
   in registry, and no fallback option succeeded``. Replace with
   ``QGlobalAveragePooling1D`` + a fixed-weight ``QDense(2)`` with kernel
   = N·I (mathematically identical for our zero-padded inputs).

Both workarounds are regression-anchored in
``tests/unit/synthesis/test_hgq2_to_hls.py``. If a future HGQ2 / hls4ml
release fixes either upstream, the dedicated "still broken" tests will tell
you and the workarounds can be removed.

2. **No integration with ``scripts/ablation.py`` yet.** A new recipe key
   (``quantization: "hgq2" | "none"``) needs to be wired into
   ``arch_search.build_model`` to dispatch, plus a training-time data-loader
   shim that one-hot-encodes the categorical features.

## Layer parameter inflation

A note on what HGQ2 adds to the model size at training time:

```
Full-precision (scalar_xybal10):    9,953 params
HGQ2 (same architecture):         196,565 params total
                                   81,333 trainable (weights + learned bit-widths)
                                  115,232 non-trainable (quantizer state, running averages)
```

The bit-width parameters do NOT contribute to FPGA resources — they're a
training-time auxiliary. The deployed model is whatever fixed-point precision
the bit-widths converged to. The 9k of "real" parameters is unchanged.

## Dependencies

The HGQ2 path requires:
- ``HGQ2`` (already in ``environment.yaml``).
- ``hls4ml>=1.0`` (for the conversion attempt, when fixed).
