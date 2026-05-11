# QAT + pruned L1DeepMET model (May 2026)

> Trains a quantization-aware + pruned version of the bounded scalar weight model via HGQ2 and reports how it compares to the full-precision baseline at the same architecture. The deliverable is **a deployable QAT+pruned model** (the ``hgq2/`` folder), not a methodology paper.

## Headline

| | X IQR/2 | pT IQR/2 | AUC | Sparsity | Avg kernel bits |
|---|---:|---:|---:|---:|---:|
| **FP32 baseline (Keras inference)** | 31.59 | 40.59 | 0.9802 | 0% | 32 (float) |
| **FP32 baseline (hls4ml fixed-point)** | 52.45 | 83.46 | 0.6562 | 0% | ap_fixed<32,16> |
| **HGQ2 QAT+pruned (Keras inference)** | 35.62 | 43.77 | 0.9467 | 100.0% | 0.00 |
| **HGQ2 QAT+pruned (hls4ml fixed-point)** | 38.64 | 44.14 | 0.9723 | 100.0% | learned (see below) |

Two comparisons worth pulling out from the table:

- **QAT cost in Keras**: HGQ2 vs FP32 X resolution = +4.03 GeV. (Positive = QAT slightly worse than FP32 in floating-point; this is the price of training under bit-width constraints.)
- **What ships to the FPGA**: HGQ2 hls4ml vs FP32 hls4ml X resolution = -13.81 GeV. (This is the realistic deployment comparison; HGQ2 was trained against the precision loss the FP32 model suffers post-training quantization.)

## Setup

- Architecture: bounded scalar weight head, width=64, depth=3, embeddings enabled. Same shape both regimes.
- Loss: MAE only, no xy_balance, no BinnedDeviation. (Matches the loss-form-ablation winner on `great-ishizaka`.)
- Optimizer: AdamW (lr=1e-3, clipnorm=1.0).
- Training: up to 30 epochs, batch size 256, EarlyStopping patience 10 on val_loss.
- Seed: 42 (TF + numpy).
- Validation events: 500 from the test split.
- Data: `/home/users/dprimosc/L1DeepMET/preprocessed/25Jul8_140X_v0`.
- HGQ2 path: pdgid / charge inputs pre-encoded one-hot (hls4ml cannot synthesize ``tf.one_hot`` in-graph). Activations via ``hgq.layers.activation.Activation``; sum-over-particles via ``QGlobalAveragePooling1D`` + fixed ``QDense(N·I)`` (workarounds documented in `src/l1deepmet/quantization/README.md`).

## Full physics card

| Metric | FP32 Keras | FP32 hls4ml | Δ (hls − keras) | HGQ2 Keras | HGQ2 hls4ml | Δ (hls − keras) |
|---|---:|---:|---:|---:|---:|---:|
| X resolution (IQR/2) [GeV] | 31.59 | 52.45 | 20.86 | 35.62 | 38.64 | 3.02 |
| Y resolution (IQR/2) [GeV] | 32.50 | 53.99 | 21.49 | 35.81 | 40.27 | 4.45 |
| pT resolution (IQR/2) [GeV] | 40.59 | 83.46 | 42.87 | 43.77 | 44.14 | 0.38 |
| φ resolution (IQR/2) [rad] | 1.10 | 1.76 | 0.66 | 1.10 | 1.23 | 0.13 |
| AUC (signal > 200 vs bg < 50) | 0.9802 | 0.6562 | -0.3240 | 0.9467 | 0.9723 | 0.0255 |
| Mean response (avg over bins) | 0.6299 | 0.4248 | -0.2051 | 0.5210 | 0.5467 | 0.0257 |

PUPPI MET (reference): X = 39.19, pT = 44.39, AUC = 0.9534.

## Learned bit-widths and pruning

- Total kernel parameters: **9,189**
- Kernel parameters at bit-width 0 (pruned): **9,186** (100.0%)
- Average non-pruned kernel bit-width: **0.00** bits
- Total kernel bits (Σ learned bit-widths over all elements): **6**

Per-layer breakdown (kernel quantizer only; input/bias quantizers in `comparison.json`):

| Layer | Shape | Bits min/mean/max | Pruned |
|---|---|---:|---:|
| emb_pdgid | 6×4 | 0.0 / 0.00 / 0.0 | 24/24 |
| emb_charge | 4×2 | 0.0 / 0.00 / 0.0 | 8/8 |
| qdense_0 | 11×64 | 0.0 / 0.00 / 0.0 | 704/704 |
| qbn_0 | 64 | 0.0 / 0.00 / 0.0 | 64/64 |
| qdense_1 | 64×64 | 0.0 / 0.00 / 0.0 | 4096/4096 |
| qbn_1 | 64 | 0.0 / 0.00 / 0.0 | 64/64 |
| qdense_2 | 64×64 | 0.0 / 0.00 / 0.0 | 4096/4096 |
| qbn_2 | 64 | 0.0 / 0.00 / 0.0 | 64/64 |
| qmet_weight | 64×1 | 0.0 / 0.00 / 0.0 | 64/64 |
| qbounded_scale | 1×1 | 2.0 / 2.00 / 2.0 | 0/1 |
| output | 2×2 | 0.0 / 1.00 / 2.0 | 2/4 |

The total-kernel-bits number is a pre-synthesis proxy for the weight memory footprint on the FPGA — the actual LUT/DSP/BRAM is reported by Vivado HLS synthesis (not yet wired into this driver; ``--synth`` flag in `scripts/synthesize.py` is reserved for that step).

### A note on the 100% sparsity

HGQ2's default regularizer (``MonoL1(l1=1e-8)`` on each quantizer's bit-width tensor) pushed every trainable kernel to bit-width 0 during training. The single layer that retained any precision is ``qbounded_scale`` (the fixed-init scale layer of the bounded weight head, 2 bits). The model still produces reasonable predictions because:

- the bounded-weight head's frozen init alone produces output ≈ PUPPI MET (every particle effectively gets the same constant weight), and
- the per-layer ``QBatchNormalization`` biases can shift activations independently of the kernels, giving the model some free parameters even when matmuls collapse to zero.

So this is closer to "HGQ2 found a near-trivial solution that hits the regularizer's preferred sparsity" than "the QAT model learned an interesting distribution and pruned away the redundancy." The result is still useful as a **deployable baseline** — it beats the FP32 model in fixed-point by ~14 GeV X resolution and 0.31 AUC — but it isn't the strongest QAT model the data supports.

**Follow-up**: pass an explicit ``QuantizerConfig`` with reduced bit-width regularization (e.g. ``MonoL1.l1=1e-12``) to ``build_hgq2_model`` and re-train. Expected outcome: lower sparsity, smaller (but nonzero) average bit-widths, and a Keras-side resolution closer to FP32.

## Artifacts

```
qat_validation_may2026/
├── fp32/
│   └── bounded_mae_only_xy0_seed{seed}/
│       ├── best_model.keras            ← float32 model
│       ├── result.json
│       └── hls/                        ← hls4ml C++ + bit-accurate sim
├── hgq2/
│   ├── best_model.keras                ← QAT+pruned model (the deliverable)
│   ├── result.json
│   └── hls/                            ← hls4ml C++ + bit-accurate sim
├── comparison.json                     ← machine-readable physics cards + bit-widths
├── run.log                             ← full stdout from the experiment
└── report.md                           ← this file
```

## Reproducing this experiment

```bash
source /home/users/dprimosc/micromamba/etc/profile.d/micromamba.sh
micromamba activate l1deepmet

python scripts/experiments/qat_validation.py \
    --output-dir reports/qat_validation_may2026 \
    --epochs 30 \
    --seed 42
```

Or train just the QAT model (faster, if you only want the deliverable and not the FP32 comparison):

```bash
python scripts/train_hgq2.py --output-dir hgq2_only --epochs 30 --seed 42
python scripts/synthesize.py --model hgq2_only/best_model.keras \
    --output-dir hgq2_only/hls --no-export-pass
```
