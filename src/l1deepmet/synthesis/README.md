# L1DeepMET synthesis pipeline

End-to-end Keras → hls4ml fixed-point HLS C++ project for L1 trigger deployment.

## Quick start

```bash
# Activate the environment (hls4ml>=1.3, da4ml>=0.5.2, etc.)
micromamba activate l1deepmet

# Convert + bit-accurate validate any saved best_model.keras
python scripts/synthesize.py \
    --model /path/to/best_model.keras \
    --output-dir hls_output/my_run \
    --n-validation-events 500
```

Produces under ``hls_output/my_run/``:

- ``firmware/`` — synthesizable HLS C++ (top-level ``l1deepmet.cpp``, per-layer
  weight files, ``parameters.h``, ``ap_types/``, ``nnet_utils/``).
- ``firmware/l1deepmet-*.so`` — compiled bit-accurate simulation library.
- ``validation.json`` — full physics card for both Keras and hls4ml predictions,
  plus per-key delta.
- ``profiling_MET.png`` / ``profiling_MET_x.png`` / ``profiling_MET_y.png`` —
  pairplots (gen, PUPPI, Keras, hls4ml).
- ``response_MET.png`` — response histograms for the three reco strategies.
- ``model_hls4ml.png`` — rendered model graph.
- ``hls4ml_config.yml`` — resolved per-layer precision config.
- ``synthesize_meta.json`` — CLI args + part / precision / clock / rf for provenance.
- ``build_prj.tcl`` + ``vivado_synth.tcl`` — Vivado driver scripts (run later
  for real LUT/DSP numbers).

## Architecture

```
scripts/synthesize.py
       │
       ▼
load_custom_objects()  ─► CastToInt, BoundedWeight, SumOverParticles, …
       │
       ▼
tf.keras.models.load_model(model_path, custom_objects=…)
       │
       ▼
synthesis.export.export_for_hls(model)
       │   - rewrites trained model with hls4ml-friendly primitives:
       │       CastToInt          → integer-typed Input layer
       │       BoundedWeight      → tanh + Dense(1) with fixed kernel/bias
       │       SumOverParticles   → GlobalAveragePooling1D + Dense(2) × N
       │       weight_minus_one   → Dense(1) with fixed bias = -1/normfac
       │   - copies trained weights for every layer with a 1:1 counterpart
       │   - sanity-checks: numerical equivalence on a random batch
       ▼
synthesis.config.build_hls_config(model_friendly)
       │   - default precision: ap_fixed<32,16>
       │   - default reuse_factor: 1
       │   - strategy: Latency
       │   - per-layer Trace=True
       │   - pdgid_inputs / charge_inputs: ap_uint<4>
       │   - optional per-layer overrides via --layer-precisions
       ▼
synthesis.convert.convert_to_hls(model_friendly, config)
       │   - part: xcvu13p-flga2577-2-e (CMS L1 Phase-2 VU13P)
       │   - clock_period: 5 ns (200 MHz)
       │   - io_type: io_parallel
       │   - emits the HLS C++ project + compiles bit-accurate .so
       ▼
synthesis.validate.compare_keras_vs_hls(keras, hls)
       │   - predicts on test.h5
       │   - computes l1deepmet.metrics.physics.full_physics_card
       │     for both (resolution, response, AUC, PUPPI deltas)
       │   - generates pairplots + response histograms
       │   - optional per-layer trace scatter plots (--trace-layers)
       ▼
hls_output/my_run/  (all artifacts above)
```

## Why ``export_for_hls`` exists

The training-time model uses three custom Keras layers that hls4ml doesn't
support out of the box:

| Layer              | What it does                                  | Replacement                              |
|--------------------|-----------------------------------------------|------------------------------------------|
| ``CastToInt``      | ``tf.cast(x, int32)`` — Embedding index cast  | Declare input ``dtype="int32"`` directly |
| ``BoundedWeight``  | ``(tanh(x) − 1) / 100`` (for defaults)        | ``Activation('tanh')`` then ``Dense(1)`` with kernel = 0.01, bias = −0.01 |
| ``SumOverParticles`` | ``tf.reduce_sum(x, axis=1)``                | ``GlobalAveragePooling1D`` + ``Dense(2)`` with kernel = N·I |
| ``ShiftByConstant`` | ``x + shift``                                | Fold into the preceding ``Dense`` bias  |

The export pass rebuilds the architecture with these substitutions and copies
the trained weights for every layer that has a 1:1 counterpart. A numerical
equivalence check on a random batch validates the rebuild (typical
``max|Δ|`` ≈ 1e-8).

Disable with ``--no-export-pass`` if you've already trained a model using only
hls4ml-friendly primitives (e.g. a QAT model authored without the custom layers).

## Default HLS config

| Knob                  | Default                          | Why                                   |
|-----------------------|----------------------------------|---------------------------------------|
| ``part``              | ``xcvu13p-flga2577-2-e``         | CMS L1 Phase-2 Serenity ATCA blade (VU13P) |
| ``clock_period``      | 5 ns                             | 200 MHz, standard CMS L1 target       |
| ``io_type``           | ``io_parallel``                  | Lowest latency; can switch to ``io_stream`` for pipelined |
| ``strategy``          | ``Latency``                      | Vs ``Resource`` (slower, fewer LUTs)  |
| ``default_precision`` | ``ap_fixed<32,16>``              | Wide starting point; sweep down for resource savings |
| ``reuse_factor``      | 1                                | 1 = full parallel, more LUTs           |
| ``trace``             | ``True`` per layer               | Enables per-layer numerical profiling. Free at synthesis time. |
| Categorical inputs    | ``ap_uint<4>``                   | pdgId 0–5, charge 0–3 — 4 bits suffice |

Override anything via CLI flags (``--precision``, ``--reuse-factor``,
``--strategy``, ``--clock-period-ns``, ``--part``, …).

## Known issues

1. **Bit-accurate validation shows ~2× precision degradation.** First run on
   ``bounded_no_bias_seed42`` gave Keras X resolution 33.0 GeV but hls4ml
   X resolution 70.7 GeV; AUC dropped 0.986 → 0.644; response shifted only
   slightly (0.65 → 0.85). Likely culprits: the tiny ``bounded_scale`` weights
   (kernel = 0.01, bias = −0.01) at the edge of ``ap_fixed<32,16>`` precision,
   or BatchNorm fold-into-Dense precision interaction. Diagnostic plots are
   produced; precision sweep is the next investigative step.

2. **Vivado HLS synthesis not yet wired.** ``--synth`` is reserved as a CLI
   flag but raises ``NotImplementedError``. Conversion + compile + bit-accurate
   sim is the supported flow (Q3a in the design discussion).

3. **No QAT (quantization-aware training) yet.** All models converted so far
   were trained in full precision and quantized only at conversion time. The
   ``DenseQuantized`` / ``DenseHGQ`` stubs in ``src/l1deepmet/models/`` are
   placeholders.

## Dependencies

Pinned in ``environment.yaml``:

- ``hls4ml>=1.0`` — earlier versions (0.6) lack Embedding support and
  Keras 3 handlers.
- ``da4ml>=0.5.2,<0.6`` — required by ``hls4ml>=1.3`` for the distributed-arithmetic
  optimization path. Without it you get "Keras v2 handler used" fallback warnings
  (non-fatal but ugly).
- ``seaborn`` — pairplots in validation.
- ``qkeras``, ``HGQ2`` — installed for the planned QAT path; not yet used here.
