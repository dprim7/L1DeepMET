# l1deepmet.estimation — synthesis-free FPGA resource estimation

Estimate the hardware cost of a model **without running Vivado/Vitis
HLS**. Three estimators, each honest about what it is and is not:

| Estimator | Models | What the number means | What it is NOT |
|---|---|---|---|
| `analytic.analyze_model` | any Keras | Per-layer params / MACs / elementwise ops from symbolic shapes. Technology-independent workload, rank-3 aware (a body `Dense(w)` over 128 particle slots counts `128·C·w` MACs). | Not LUTs, not DSPs. A float model's MACs say nothing about post-quantization cost. |
| `ebops.collect_ebops` / `recompute_ebops` | HGQ2 only | EBOPs ≈ Σ bits_in × bits_kernel over each layer's multiply structure — HGQ's differentiable proxy that tracks LUT+DSP usage well. Stored in the checkpoint; `collect` reads, `recompute` traces calibration data through the model (mutates model state). | Not calibrated to absolute LUT counts. All-zero on models whose quantizers collapsed (see below) or were never traced. |
| `tracing.trace_costs` | HGQ2 only | da4ml symbolic replay: LUT-equivalent cost and (min, max) latency in adder stages for a **fully-parallel distributed-arithmetic** implementation (`io_parallel`, no DSPs, no reuse). | Not a Vivado report; a different HLS strategy (DSP-mapped multiplies, reuse > 1) lands elsewhere. Compare configurations with it; don't quote it as utilization. |

**There is no honest synthesis-free DSP number.** DSP usage depends on
HLS strategy/precision thresholds that only synthesis resolves; the
report records reasons for skipped sections instead of inventing values.

## Quick start

```bash
python scripts/estimate_resources.py \
    --model reports/qat_validation_may2026/hgq2/best_model.keras \
    --output-dir outputs/estimation/hgq2_baseline
```

Writes `estimate.json` + `estimate.txt`. Family is auto-detected: fp32
checkpoints get the analytic table only. Useful flags: `--no-trace`
(the full w64/d3 trace takes minutes), `--per-layer-trace`,
`--calibration-h5 <preprocessed>.h5` for real-data EBOPs recompute
(default is a synthetic batch — fine for plumbing, not for
physics-meaningful WRAP ranges), `--inputs-kif K I F` for models that
do arithmetic on raw inputs before any Q-layer
(`build_hgq2_model(use_embeddings=False)`'s dummy path needs it).

Python API: `estimate_resources(model, calibration=...)` →
`ResourceEstimate`; individual estimators are importable from
`l1deepmet.estimation`.

## Training integration

`scripts/train_hgq2.py` logs total EBOPs per epoch into `history.csv`
(hgq `FreeEBOPs` callback, on by default, `--no-ebops-log` to disable)
and writes the final per-layer breakdown into `result.json["ebops"]`.

## Caveats and sharp edges

- **The May-2026 QAT checkpoint's EBOPs (total 10,036, body layers ≈ 0)
  are degenerate**: HGQ2's default `MonoL1` regularizer drove the kernel
  bit-widths to zero (`reports/qat_validation_may2026/report.md`).
  The number is pinned in the integration test as a loading-semantics
  anchor, not as a physics result. Fixing the collapse (e.g. weaker
  `MonoL1.l1`, `beta0` tuning) is a separate experiment.
- `recompute_ebops` **mutates** the model: `_ebops` weights always,
  WRAP-mode quantizer min/max registers when `reset=True`. BatchNorm
  moving statistics are untouched (pinned by test). The CLI loads its
  own instance, so this only matters for in-process use.
- `register_extra_handlers()` patches two da4ml-0.5.2 gaps (exact-class
  registry): plain `keras.layers.Activation` (hgq's "QActivation" is
  just an alias of it; tanh & co. go through the deferred-LUT `apply`
  mechanism) and plain `Concatenate` (upstream broadcasts before
  concatenating). Revisit on a da4ml upgrade.
- Per-layer trace costs are cumulative-from-inputs and use da4ml's
  private unflattened dump (`trace_model(dump=True)` crashes on
  deferred LUT tensors upstream). Opt-in; degrades to `None`.
- hgq2 ≥ 0.1.9 is required for the da4ml converter (pinned in
  `environment.yaml`); on incompatible envs the tracing section reports
  `TracingUnavailableError` and everything else still works.
- Wall time, single CPU node: tiny models (w8/d1, 16 slots) trace in
  ~1 min; the production-size w64/d3 128-slot model takes several
  minutes (see the `slow`-marked integration test).
