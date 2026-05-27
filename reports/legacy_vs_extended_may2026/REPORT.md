# Legacy vs extended feature subset — direct comparison

**Date:** 2026-05-27
**Question:** does the extended 22-continuous-slot block do anything
detectable on top of the 4-slot legacy-equivalent baseline at the same
architecture and training protocol?

## Setup

Same H5, same architecture, same protocol — only the continuous-input
slot list differs.

|                                | Legacy-equivalent                 | Extended                                                                       |
|---|---|---|
| H5                             | `outputs/preprocessed/26May27_142_extended_sanitized_v0/` (post-FLT_MAX sanitization, 77,380 train events) | same |
| Continuous slots               | `0,1,2,3` (pt, eta, phi, puppi_weight) — 4 features | `0,1,2,3,8,9,…,25` (4 base + dxy + z0 + 5 hw* + 5 track* + 2 calo* + 4 cl*) — 22 features |
| Momentum / categoricals        | extended-layout positions: momentum=4-5, cats=6-7 (cats unused by mode-1 architecture) | same |
| Architecture                   | mode-1 per-particle weight MLP, 3 × Dense64 + BatchNorm + tanh, weight × pxpy → GlobalAvgPool | same |
| Hyperparameters                | epochs=20, batch=1024, AdamW lr=1e-3 clipnorm=1, normfac=100, seed=42 | same |
| Output dir                     | `outputs/models/legacy_eq_w64d3_seed42/` | `outputs/models/extended_w64d3_seed42/` |
| Trainable params               | 9,089                             | 10,241                                                                          |

The 1,152-param delta is the first dense layer growing from `(4 → 64)` to
`(22 → 64)`. Everything else is identical.

## Results

`scripts/experiments/compare_legacy_vs_extended.py` on the test split
(9,673 events). Raw JSON: `comparison.json` (same directory).

```
┌──────────────────────────────┬────────────────┬────────────────┬─────────────┐
│ metric                       │     legacy (4) │  extended (22) │            Δ │
├──────────────────────────────┼────────────────┼────────────────┼─────────────┤
│ mae_xy                       │        34.1487 │        26.4340 │      -7.7147 │  ← extended better
│ mse_xy                       │      2266.5881 │      1600.4215 │    -666.1666 │  ← extended better
│ met_x_resolution (IQR/2)     │        38.7898 │        27.9221 │     -10.8677 │  ← extended better
│ met_y_resolution (IQR/2)     │        38.8802 │        28.4356 │     -10.4446 │  ← extended better
│ met_pt_resolution (IQR/2)    │        43.4328 │        59.4232 │     +15.9904 │  ← extended worse
│ phi_resolution               │         1.5169 │         1.7164 │      +0.1995 │  ← extended worse
│ mean_response                │         1.3475 │         0.4690 │      -0.8785 │  ← extended SEVERELY under-shrinks
│ response_50_100              │         1.3585 │         0.6101 │      -0.7484 │
│ response_100_200             │         1.2504 │         0.4709 │      -0.7795 │
│ response_200_300             │         1.3432 │         0.4383 │      -0.9048 │
│ response_300_400             │         1.4378 │         0.3565 │      -1.0813 │
│ response_400_inf             │         1.0000 │         1.0000 │      +0.0000 │  ← too few events for this bin
└──────────────────────────────┴────────────────┴────────────────┴─────────────┘
```

## Verdict — INCONCLUSIVE on features, DECISIVE on architecture/loss

**Neither wins cleanly.** The extended model improves X / Y resolution
substantially (~10 GeV reduction in each per-axis IQR/2) and the joint
MAE / MSE on (px, py), but it pays for this by **collapsing the
response toward zero** — mean predicted pT is only **47% of mean gen
pT**, vs the legacy model's 135% over-shoot. The collapsed response
destroys pT resolution and degrades φ. This is the over-shrinkage
failure mode already documented in
`reports/output_head_overshrink_apr2026/`.

What this tells us:

1. **The extended features ARE being used.** The X / Y IQR/2 drop of
   ~10 GeV is much larger than any seed-to-seed variance the
   over-shrinkage report observed (~1-2 GeV) at this protocol. Random
   weight initialization can't explain a 10 GeV move. The model is
   genuinely consuming z0 / hw* / track quality / cluster ID and
   reducing per-axis residuals.

2. **But more features at this loss + arch worsens the failure mode.**
   The legacy 4-feature model has too little capacity to over-shrink;
   it under-shrinks (response 1.35). Give it the extra 18 features and
   it has the capacity to shrink hard — and the BinnedDeviation +
   MAE + MSE composite loss doesn't penalize that strongly enough to
   stop it. Same loss-tilting issue as the over-shrinkage report.

3. **Feature ablation is NOT the right next experiment.** With this
   loss / architecture, the model converges to a degenerate response
   regardless of which feature subset it has. Running 24-cell LOGO at
   the current protocol would produce table entries dominated by the
   response collapse, not the feature contributions.

## The real next step

Fix the response-collapse failure mode FIRST, then re-do the comparison.
Three protocols from `output_head_overshrink_apr2026` are candidates:

| Fix | Mechanism | Cost |
|---|---|---|
| **Asymmetric MAE / Huber on |MET| with bias term** | Penalises under-prediction directly | ~30 min retrain |
| **Bias-corrected output head** | Add a learnable scalar that re-scales the pooled output, regularised to keep response ≈ 1 | ~30 min retrain |
| **Mode 2 (single event-level weight on Σpxpy)** | Architecturally limits the over-shrink — the model can only scale, not zero | already exists in `build_dense` |

The cheapest one to try is mode 2 — flip `--mode 2` and re-run both
trainings on the same H5s. If mode-2 extended ≥ mode-2 legacy on the
physics card, that's the green light for ablation. If mode-2 still
collapses on the extended side, the loss is the binding constraint.

## Caveats (in addition to those in §Setup above)

- **Single seed.** 1 seed at 20 epochs is a sniff test. The conclusions
  are robust to the qualitative pattern (extended changes the failure
  mode); the absolute Δ numbers should not be quoted as a final result.
- **20 epochs is below convergence.** Both models are still moving on
  val_loss in their final epochs. Plateau winners in the over-shrink
  report needed 100+ epochs. The trajectories at this protocol are
  representative of "what does the model converge toward" but not of
  "what's the best each can do."
- **No event-level features.** The 13 extra event features in the H5
  (alt-MET algos, vertex info) are NOT consumed by `build_dense` —
  they sit in the `event_features` dataset waiting for an architecture
  that can use them (concat after pool, or as bias on the weight head).
  Including them is the obvious next-but-one experiment.
- **dxy is dead in slot 8 of the extended config but included anyway.**
  Dropping it would change zero of the numbers above; it's there for
  layout-stability + forward-compat.

## Caveats

- Single seed. A 3-seed repeat is the next thing if this result is
  inconclusive — at 20 min/run it's a ~1-hour follow-up.
- 20 epochs is below the convergence horizon for these per-particle-weight
  architectures (~100 epochs in the over-shrinkage report). So absolute
  numbers will be worse than the published baselines; only the relative
  comparison is meaningful.
- The extended path uses 18 new features that are NOT all created equal —
  `dxy` (slot 8) is structurally zero in CMSSW_14_2_X, several track
  features carry `-1` on neutral candidates by design, `clPuId/clEmId`
  carry `-1` on charged candidates by design. The model effectively sees
  a mixture of real and sentinel-marker values. A future-tier ablation
  could remove these "designed-zero" groups to isolate where the gain
  actually comes from — but that's only worth doing if THIS comparison
  shows a real gain in the first place.

## What this experiment is NOT

- Not a feature-importance / SHAP-style analysis. Drop-column / grouped
  ablation only becomes useful once both models converge to a
  non-degenerate response.
- Not a physics-card evaluation against PUPPI. The PUPPI baseline + AUC
  trigger metrics need an architecture that doesn't shrink the response;
  the verdict above says we don't have one yet.
- Not a recipe-fix campaign. The caloEta/caloPhi recipe bug (reads
  `pfTrack.caloEta` → -999 on neutrals instead of `pfCluster.eta`) and
  the slot-22 / slot-23 FLT_MAX sentinels both live upstream of this
  comparison. FLT_MAX is already handled by the preprocessor's
  sanitization (`feat(preprocess)` commit `3a309b7`). The
  caloEta/caloPhi recipe fix would need re-production (~26 h wall) —
  see the open-follow-ups list in `reports/ntuple_production_26May20/`.

## Reproduce

From the worktree root, with the patched submodule applied and the
CMSSW area not required (preprocessing + training only):

```bash
micromamba activate l1deepmet
export TF_NUM_INTRAOP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2

# (already done in this branch — listed for completeness)
# python scripts/preprocess.py \
#     --config params.yaml --tag 26May27_142_extended_sanitized_v0 \
#     --data-root /ceph/cms/store/user/dprimosc/l1deepmet/26May22_142_extended_20k_v0 \
#     --output-root outputs/preprocessed --feature-layout extended

PYTHONPATH=src python scripts/train.py \
    --data-dir outputs/preprocessed/26May27_142_extended_sanitized_v0 \
    --output-dir outputs/models/legacy_eq_w64d3_seed42 \
    --epochs 20 --batch-size 1024 --normfac 100 --mode 1 \
    --units 64,64,64 --continuous-slots 0,1,2,3

PYTHONPATH=src python scripts/train.py \
    --data-dir outputs/preprocessed/26May27_142_extended_sanitized_v0 \
    --output-dir outputs/models/extended_w64d3_seed42 \
    --epochs 20 --batch-size 1024 --normfac 100 --mode 1 \
    --units 64,64,64 \
    --continuous-slots 0,1,2,3,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25

PYTHONPATH=src python scripts/experiments/compare_legacy_vs_extended.py \
    --h5-dir outputs/preprocessed/26May27_142_extended_sanitized_v0 \
    --legacy-model   outputs/models/legacy_eq_w64d3_seed42/model.keras \
    --extended-model outputs/models/extended_w64d3_seed42/model.keras \
    --legacy-slots   0,1,2,3 \
    --extended-slots 0,1,2,3,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 \
    --out reports/legacy_vs_extended_may2026/comparison.json
```

Two trainings × 20 epochs is ~40 min wall (CPU-only on UAF with 1
intra-op thread). The comparison itself takes <30 s.
