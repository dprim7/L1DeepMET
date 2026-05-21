# Event-count justification — pre-registered experiment design

**Date.** 2026-05-21
**Branch.** `claude/jolly-lalande-3e5006`
**Author.** Claude (experimenter), human-in-the-loop dprimosc

## Question (concrete, falsifiable)

> *On the ~10k-event Phase2Spring24 extended-features dataset (`26May21_142_extended_v0`), fixing the production-baseline Dense recipe (Dense w64 d3 mode-1, MAE-only loss, normfac=100), at what training event count N does the val MAE stop falling steeply enough to matter — i.e., the marginal improvement from doubling N is within seed-to-seed std?*

This is a scaling-law question, not an absolute-performance question. We're sizing the production budget, not picking the winning architecture.

## Hypothesis (with stated uncertainty, written before measurement)

- **Prior**: prior production used ~148k events on the 9-feature legacy data; baseline reached X IQR/2 ≈ 33 GeV with seed std ≈ 0.04 GeV; a ~42 GeV pT-resolution floor was identified as data/architecture-limited.
- **Prediction (extended variant)**: the learning curve should plateau between 5k–15k events for this ~10k-param model. Specifically:
  - I expect val MAE at N=8000 to be within 0.5 GeV of val MAE at N=4000 (i.e., marginal improvement from doubling ≤ 1 σ_seed).
  - I expect val MAE at N=400 to be 2–5 GeV worse than at N=8000.
  - I'd be **surprised** if N=8000 still shows large gains from more data (would mean the curve hasn't plateaued, and 10k is genuinely insufficient).
  - I'd be **surprised** if N=400 is already saturated (would mean this task is trivially easy and our extended features aren't doing much).
- **Prediction (extended vs baseline)**: at fixed N=8000, the extended-features model should beat the baseline on at least the tail bins (gen MET > 200 GeV) by ≥ 1 σ_seed. If it doesn't, the new features aren't paying their way.

## Design

### Dataset

- **Source**: `outputs/preprocessed/26May21_142_extended_v0/{train,val,test}.h5` (to be produced by `scripts/preprocess.py --feature-layout extended` on the 5-sample × ~2k-event Phase2Spring24 production).
- **Schema**: `features` (N, 128, 26), `event_features` (N, 13) [unused in this experiment], `targets` (N, 2).
- **Note**: We use the per-candidate features only. Event-level features (vertex, alt-MET algos) are saved on disk but not consumed — they'd require a model architecture change that's out of scope for this experiment.

### Splits (deterministic, frozen across all runs)

- After preprocessing, the existing splitter in `combine_shuffle_split_extended` already creates train/val/test at 0.8/0.1/0.1 ratios with shuffle seed 42.
- For this experiment: **use the existing val/test as our frozen held-out sets**, subsample the existing train.
- Sub-pool sizes: N ∈ {400, 800, 2000, 4000, 8000}. If train pool < 8000, cap at pool size.
- Subsample seed = experiment seed (≠ split seed).

### Variants

- **`extended`**: all 26 per-candidate continuous features fed into `continuous_inputs`.
- **`baseline`**: same model, same input shape (B, 128, 26), but slots 4..25 (everything except pt/eta/phi/puppi_weight) are **zeroed before the model sees them**. This keeps the architecture identical, isolating the effect of the new features. Dead-column dynamics in BN + Dense make zeroed columns effectively unseen by the model.

### Seeds

- **3 seeds per cell**: 42, 123, 456 (matches prior reports' convention).
- Used for: (i) train subsampling, (ii) weight init, (iii) per-epoch batch shuffling.

### Recipe (fixed)

- Architecture: Dense, mode 1 (per-particle weight × px/py, summed). w64 d3 (units=[64,64,64]), tanh activation, BatchNorm momentum=0.95, no per-particle bias.
- Loss: MAE only (the `mae_only` config from `final_baseline_apr2026`, with `binned_weight=0`, `xy_balance_weight=0`).
- Optimizer: Adam, lr=1e-3.
- normfac=100 on targets.
- Batch size: 256.
- Epochs: 50 (with no early stopping — the question is "what's the final val MAE at this N?", not "what's the best epoch?"; fixing epochs removes a confound).
- TF: CPU only (`CUDA_VISIBLE_DEVICES=`), 1 intra-op thread per worker so we can run many in parallel.

### Cells

5 fractions × 2 variants × 3 seeds = **30 training runs**.

### Metrics (computed on the test set after training)

Per run, write to JSON:
- Overall val MAE, val MSE
- Per-gen-MET-bin MAE for bins `[0,50,100,200,300,400,∞]` GeV (matches BinnedDeviation convention)
- Per-bin bias = mean(pred_x − true_x), mean(pred_y − true_y)
- X IQR/2 and Y IQR/2 (the metric prior reports use)
- Per-bin event count (to flag any bin with < 50 events as low-confidence)

### Decision rule (written before measuring)

We compute σ_seed_per_cell = std over 3 seeds of overall val MAE.

- **Data-limited rule**: if for the extended variant
  `val_MAE(8000) > val_MAE(4000) − 1·σ_seed`
  → data is **not** the binding constraint at 10k events; we can ship with 10k.

- **Data-bottleneck rule**: if the marginal improvement from doubling N at N=8000 is **larger** than σ_seed
  → fit `MAE ≈ A + B·N^{−α}` and report N* such that doubling at N* yields ≤ σ_seed improvement. That's our recommendation.

- **Feature-value check**: at N=8000, compare extended vs baseline by bin. The new features earn their keep iff `MAE_extended(bin) < MAE_baseline(bin) − 1·σ_seed` in at least one tail bin (gen MET > 200 GeV).

## Sanity checks before scaling (don't skip)

1. **One-cell smoke**: train (N=400, extended, seed=42, 5 epochs). Check loss decreases, no NaN, output JSON has the expected fields. Cost: ≤ 1 min.
2. **Baseline zeroing**: verify that for the baseline variant, slots 4..25 of `continuous_inputs` are actually zero at training time (not just in the preprocessor).
3. **Split disjointness**: assert train/val/test indices have empty intersection.

## Time budget

- Per run: ~3 min (8k events × 50 epochs × ~ms/batch, Dense feedforward only).
- 30 runs × 1 core each, 16 cores available, 2 batches → ~10 min wall.
- Plus preprocess (~5 min), analysis + plots + report (~30 min).
- **Total: ≈ 1 hour** wall once TT retry finishes.

## Outputs (committed under `reports/event_count_justification/`)

```
PLAN.md                    # this document (pre-registered)
report.md                  # results, interpretation, recommendation
scripts/
  train_one.py             # single (N, variant, seed) → results JSON
  run_sweep.py             # parallel multiprocessing driver
  analyze.py               # CSV aggregation, plots, learning-curve fit
results/
  raw/<run_id>.json        # per-run metrics + config + git SHA
  aggregated.csv           # one row per (N, variant, seed)
  fits.json                # learning-curve fit parameters + N*
plots/
  learning_curve.png       # val MAE vs N, both variants, with seed-band
  per_bin_mae.png          # per-pT-bin MAE at multiple N
  extended_vs_baseline.png # variant comparison at N=8000
  extrapolation.png        # fit + extrapolation to target SEM
LAUNCH.md                  # exact reproduction commands
```

## What this experiment does NOT answer

- Whether the event-level features (vertex, alt-MET) help — they're saved but unused here.
- Whether the optimal architecture is Dense vs Mixer vs DeepSets — fixed architecture by design.
- Whether the model fits the FPGA latency/resource budget — architectural decision, not data-budget.
- Whether the standard pT-resolution floor (~42 GeV) can be broken — that requires the FULL feature set + architecture exploration, this is just the data-quantity prerequisite.

## After this experiment

If the conclusion is "10k is sufficient", the next experiment should be **architecture** (does Mixer / DeepSets beat Dense given the new features?) or **event-level features** (does adding a global head that consumes `event_features` help?).

If the conclusion is "we need N* > 10k", the next experiment is **scaling production** to N*, then re-running this ablation at the new size to confirm the plateau.
