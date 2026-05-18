# Output-head over-shrinkage failure mode (May 2026)

**TL;DR.** Two unrelated architecture changes that both *relax the output structure* away from "weighted sum of input momenta" reproduce the same degenerate failure: the model collapses to predict-small. X/Y IQR/2 looks much better than PUPPI; pT IQR/2 is 12–18 GeV worse than PUPPI; AUC sometimes drops below PUPPI. The plateau MLP with mode-1 weight_minus_one output is the only configuration we have that **simultaneously beats PUPPI on every physics axis** at ~10k params.

## The failure mode

When the output is a *free regression* (a Dense(2) on aggregated features, or a transformer-derived per-particle weight × pxpy without strong initialization at PUPPI MET):

```
Symptoms:
  X IQR/2  ≪ PUPPI   ✓ ("better resolution")
  Y IQR/2  ≪ PUPPI   ✓
  pT IQR/2 ≫ PUPPI   ✗ (much worse!)
  AUC      ≈ PUPPI   sometimes better, sometimes worse
  φ res    > PUPPI's ✗ (worse direction reconstruction)
```

The pattern is unambiguous: **X/Y are sensitive to the **magnitude** of predictions, pT is sensitive to the **per-event scatter** of predictions**. A model that predicts near-zero on uncertain events has small |pred_X − gen_X| (because |pred| is small), even if its pT is wildly mis-scaled.

## Three independent observations of the same failure

### Observation 1 — Transformer recon (May 2026, earlier)

| Config | Params | X | Y | pT | AUC |
|---|---:|---:|---:|---:|---:|
| PUPPI ref | 0 | 38.30 | 38.38 | 44.55 | 0.9741 |
| MLP w64 d3 (production) | 9,953 | 33.17 | 32.80 | 41.56 | 0.9770 |
| **xformer_w64_d2 (recon)** | 9,913 | **31.59** | 31.67 | **52.80** ⚠️ | 0.9788 |
| **xformer_w64_d3 (recon)** | 14,437 | **30.07** | 30.30 | **56.35** ⚠️ | 0.9734 |

Diagnosed previously: transformer's pred_pT mean = 39.2 vs gen mean 57.2 (response 0.69, vs MLP's 0.80). Per-event std(reco/gen) at high pT is 0.25 vs MLP's 0.16. **Higher per-event scale variance is the actual pathology**; the bin-mean response is similar.

### Observation 2 — Deep Sets ρ-head (this week)

Same MLP φ as production; vary post-aggregation head. Loss is the same `mae_only` we use everywhere.

| Config | Params | X | Y | **pT** | AUC | Δ_pT vs PUPPI |
|---|---:|---:|---:|---:|---:|---:|
| **PUPPI MET (ref)** | 0 | 38.30 | 38.38 | **44.55** | 0.9741 | 0 |
| **mode1_w64_d3_ref** | 9,953 | 33.31 | 33.13 | **41.86** ✓ | 0.9782 | **−2.69** |
| deepsets_rho0 (mode 0, Dense(2) head) | 10,018 | 34.85 | 35.35 | 44.22 | 0.9597 ⚠️ | −0.33 |
| **deepsets_rho1_w64_d3** | 14,434 | **29.49** | 29.67 | **60.45** ⚠️ | 0.9828 | **+15.90** |
| **deepsets_rho2_w64_d3** | 18,850 | **29.20** | 29.13 | **60.05** ⚠️ | 0.9812 | **+15.50** |
| **deepsets_rho2_w32_d3** | 5,346 | **29.43** | 29.79 | **62.67** ⚠️ | 0.9808 | **+18.12** |

- The **mode-1 reference** (current production) beats PUPPI on every axis simultaneously.
- The **mode-0 baseline with no ρ MLP** doesn't beat PUPPI on AUC (0.9597 < 0.9741), but doesn't suffer the pT blow-up either — its `Dense(2)` head is too weak to over-shrink confidently.
- Every **ρ-head variant** (depths 1 or 2, widths 32 or 64) reproduces the over-shrinkage failure: X drops by ~4 GeV vs mode-1 (looking great) while pT IQR/2 *increases* by ~18 GeV vs PUPPI. AUC slightly better than PUPPI but worse than mode-1 production.
- Failure is independent of capacity — `deepsets_rho2_w32_d3` (5,346 params, *smaller* than production) has the worst pT among the ρ variants.

### Observation 3 (in progress) — Transformer with loss × head sweep

Test whether the over-shrinkage is fixable by changing the loss (MAE → MSE) or the output head (free → bounded). 4 cells × 3 seeds; first cell (`xformer_mae_wmo`, reproducing the recon) confirms the baseline over-shrinkage. Result on the other 3 cells expected ~9 hr from now.

## Why this happens — the median trap

For zero-mean targets (gen_MET_x is approximately N(0, 58)), the **median of the conditional distribution is zero** for any input. MAE-based loss minimises L1 distance to the median — so the global optimum is **always predict zero**, for any feature input.

What stops a model from collapsing to this trivial solution?

| Constraint | Production MLP | ρ-head | Transformer (recon) |
|---|---|---|---|
| Output structurally is "Σ w_i × pxpy_i" (cannot easily be 0 without weights → 0) | ✓ | ✗ — free regression | ✓ but more capacity in φ |
| `weight_minus_one` init: starts at PUPPI MET, not 0 | ✓ | n/a (mode 0) | ✓ |
| Bounded weight head (BoundedWeight: tanh-bound ∈ [-2, 0]) | not used | n/a | not used in recon |
| Limited per-particle capacity to "vary independently" | ✓ (1,761–9,953 params plenty for this task) | the ρ MLP can collapse the aggregated representation to 0 | attention can produce small queries/keys that yield small outputs |

The production MLP avoids the trap because:
1. The output **must** be `Σ w_i × pxpy_i` (mode-1 architectural constraint).
2. `weight_minus_one` initialises at PUPPI MET (w_i = -1/normfac for every i).
3. The MLP's per-particle factorisation makes it hard for the model to set all w_i ≈ 0 without losing on training-loss for the bulk of events.

Free-output architectures (ρ-head, attention with capacity to drive its own outputs small) can route around these guardrails.

## What this means for next steps

1. **Don't pursue ρ-head Deep Sets at this loss/output configuration.** The post-aggregation `Dense(2)` head reproduces the over-shrinkage failure. If we want a ρ MLP, we need to wire it as a residual correction *on top of* the mode-1 output, not as the output itself.

2. **The transformer_loss_head ablation is now the key remaining experiment.** Two hypotheses to test:
   - **MSE replaces MAE** → unique optimum at conditional mean, no zero-median attractor.
   - **Bounded weight head** → weights bounded in [-2, 0], cannot collapse to 0 freely.

   If either fixes the transformer, the same fix will fix the ρ-head. If neither does, the attention encoder is fundamentally too flexible for this task at L1-budget sizes.

3. **The plateau MLP at ~33 GeV X / 41.9 GeV pT / 0.978 AUC remains the strongest single result.** No architecture we've tried (including bigger MLPs from the arch_sweep) beats it on the full physics card. The path to further gains likely runs through **data** (more events; Spring24 once available) or **features** (vertex association, track quality) rather than architecture.

## Artifacts

```
outputs/deepsets_rho_apr2026/
├── ablation_results.csv               # training-time results
├── deepsets_rho0_seed{42,123,456}/    # mode-0 baseline
├── deepsets_rho1_w64_d3_seed*/        # 1-layer ρ
├── deepsets_rho2_w64_d3_seed*/        # 2-layer ρ
├── deepsets_rho2_w32_d3_seed*/        # 2-layer ρ, slim φ
└── mode1_w64_d3_ref_seed*/            # production reference (= the plateau)
```

15 saved models, each with `best_model.keras` + `result.json` + `history.csv`. Re-eval through `scripts/reeval_physics_card.py` once the transformer ablation lands will give a single CSV across both.

## Reproduce

```bash
source /home/users/dprimosc/micromamba/etc/profile.d/micromamba.sh
micromamba activate l1deepmet

# Deep Sets ρ-head ablation (this report) — ~3 hr:
python scripts/ablation.py --recipe deepsets_rho --epochs 30 --seeds 42 123 456

# In-progress comparison — Transformer × loss × head — ~9 hr:
python scripts/ablation.py --recipe transformer_loss_head --epochs 30 --seeds 42 123 456
```
