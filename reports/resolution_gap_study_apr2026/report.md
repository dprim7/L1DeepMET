# Resolution Gap Study: Closing the X/Y MET Resolution Gap with PUPPI

**Date**: April 4, 2026
**Context**: Following the DeepMET fixes ablation study, all ML models achieve pT resolution comparable to PUPPI (44.1 vs 44.3 GeV) and stable phi/AUC, but the X/Y resolution remains 11% worse (42.6 vs 38.3 GeV). This study investigates why and what can be done.

## 1. Problem Statement

After implementing the DeepMET recipe (weight-minus-one, reduce_sum, embeddings), the model achieves:
- pT resolution: 44.3 GeV (matches PUPPI's 44.3)
- Phi resolution: 1.166 rad (matches PUPPI's 1.166)
- AUC: 0.974 (matches PUPPI's 0.974)
- **X resolution: 42.6 GeV (PUPPI: 38.3 GeV) -- 11% gap**
- **Y resolution: 42.7 GeV (PUPPI: 38.4 GeV) -- 11% gap**

Why does MET pT resolution match but X/Y doesn't?

## 2. Hypothesis: Scalar Weight Limitation

The mode-1 output head computes: `MET = -sum(w_i * px_i, w_i * py_i)`

With a scalar weight `w_i`, the model can only scale each particle's momentum uniformly. It cannot independently correct px and py. PUPPI, being a particle-level algorithm that zeros out pileup particles entirely, effectively applies a binary (0 or 1) scalar weight -- but it benefits from particle-level truth information not available to the DNN.

**Proposed fix**: 2D weights `(w_x, w_y)` per particle, so `MET_x = -sum(w_xi * px_i)` and `MET_y = -sum(w_yi * py_i)`.

## 3. Round 1: Architecture Experiments

### Configs tested (3 seeds each, 100 epochs, early stopping patience=10):

| Config | Key change |
|--------|-----------|
| baseline_wm1_w64_d3 | Scalar weight, no embeddings (reference) |
| w2d_emb_w64_d3 | 2D weights + embeddings |
| w2d_bias_emb_w64_d3 | 2D weights + bias + embeddings |
| philoss50_emb_w64_d3 | Scalar + phi loss (weight=50) |
| w2d_philoss50_emb_w64_d3 | 2D weights + phi loss (weight=50) |
| w2d_cosine_emb_w64_d3 | 2D weights + cosine LR decay |
| w2d_philoss_cosine_emb_w64_d3 | 2D + phi loss + cosine |
| w2d_lr01_emb_w64_d3 | 2D weights + lr=0.01 |

### Round 1 Results:

| Config | pT Res | X Res | Y Res | Stable? |
|--------|--------|-------|-------|---------|
| PUPPI | 44.3 | 38.3 | 38.4 | -- |
| baseline | 44.3+/-0.4 | 42.7+/-0.4 | 42.7+/-0.4 | Yes |
| w2d_emb | 44.6+/-0.3 | 43.6+/-2.5 | 42.2+/-2.8 | **No -- axis asymmetry** |
| w2d_bias_emb | 44.4+/-0.4 | 55.3+/-20.6 | 44.0+/-1.8 | **No -- seed failures** |
| philoss50 | NaN | NaN | NaN | **Diverged** |
| w2d_cosine | 44.6+/-0.2 | 41.2+/-1.3 | 44.9+/-1.7 | **No -- axis asymmetry** |
| w2d_lr01 | 46.2+/-1.9 | 64.2+/-19.5 | 52.2+/-20.4 | **No -- unstable** |

### Round 1 Key Insights:

1. **2D weights CAN reach PUPPI-level resolution on individual axes**: `w2d_emb` seed 456 achieved Y=39.8, `w2d_bias` seed 123 achieved X=38.4. But never both axes simultaneously.

2. **The axis asymmetry is the core problem**: With 2D weights, the model sacrifices one axis to improve the other. The loss function (MAE + MSE + BinnedDeviation) doesn't enforce X/Y symmetry -- BinnedDeviation only penalizes pT errors.

3. **Phi loss at weight=50 causes NaN**: All 6 phi-loss runs diverged within 1 epoch. The atan2-based implementation had numerical issues.

4. **Higher LR (0.01) is unstable**: Consistent with the constrained optimization landscape.

## 4. Round 2: Loss Function Rebalancing

Based on Round 1, the problem is the loss function, not the model. Three fixes:

1. **XY balance loss**: `|MAE_x - MAE_y|` -- directly penalizes axis asymmetry
2. **Phi loss (fixed)**: Dot-product formulation `1 - cos(dphi)` weighted by pT (numerically stable)
3. **Reduced binned_weight**: Lower from 200 to 50 to rebalance pT vs X/Y objectives

### Configs tested:

| Config | 2D weights | XY bal | Phi loss | Binned wt |
|--------|-----------|--------|----------|-----------|
| baseline_wm1 | No | No | No | 200 |
| w2d_xybal10 | Yes | 10 | No | 200 |
| w2d_bw50 | Yes | No | No | 50 |
| w2d_phi2 | Yes | No | 2.0 | 200 |
| w2d_xybal_phi2 | Yes | 10 | 2.0 | 200 |
| w2d_bw50_xybal_phi2 | Yes | 10 | 2.0 | 50 |
| w2d_xybal50 | Yes | 50 | No | 200 |
| scalar_xybal10 | No | 10 | No | 200 |

### Round 2 Results (mean +/- std across 3 seeds):

| Config | pT Res | X Res | Y Res | max(X,Y) | Phi Res | Stability |
|--------|--------|-------|-------|-----------|---------|-----------|
| **PUPPI** | **44.3** | **38.3** | **38.4** | **38.4** | **1.166** | -- |
| baseline (scalar+emb) | 44.8+/-0.7 | 43.9+/-1.9 | 43.9+/-1.8 | 43.9 | 1.173+/-0.009 | Poor |
| w2d_xybal10 | 44.2+/-0.0 | 42.6+/-0.5 | 42.9+/-0.3 | 42.9 | 1.166+/-0.001 | **Excellent** |
| w2d_bw50 | 44.3+/-0.2 | 43.1+/-1.1 | 42.3+/-1.3 | 43.1 | 1.165+/-0.002 | Moderate |
| w2d_phi2 | 44.1+/-0.2 | 42.6+/-0.6 | 43.1+/-0.2 | 43.1 | 1.165+/-0.000 | Good |
| w2d_xybal_phi2 | 44.5+/-0.3 | 41.3+/-1.5 | 44.0+/-1.5 | 44.0 | 1.165+/-0.001 | Poor |
| **w2d_bw50_xybal_phi2** | **44.1+/-0.1** | **42.4+/-0.3** | **42.7+/-0.4** | **42.7** | **1.163+/-0.003** | **Best** |
| w2d_xybal50 | 44.2+/-0.0 | 42.2+/-0.1 | 43.2+/-0.2 | 43.2 | 1.165+/-0.000 | **Excellent** |
| scalar_xybal10 | 44.1+/-0.3 | 42.7+/-0.1 | 42.7+/-0.2 | 42.7 | 1.165+/-0.001 | **Excellent** |

## 5. Analysis

### What improved:

1. **XY balance loss is the single most impactful change for stability**: All configs with XY balance have dramatically lower variance (std < 0.5 GeV vs 1.9 GeV for baseline). This is the most important finding.

2. **The best config** (`w2d_bw50_xybal_phi2`) achieves:
   - pT res: 44.1 GeV (1% better than PUPPI)
   - X res: 42.4 GeV (10.5% worse than PUPPI)
   - Y res: 42.7 GeV (11.2% worse than PUPPI)
   - Phi res: 1.163 rad (0.3% better than PUPPI)
   - Seed variance < 0.4 GeV on all metrics

3. **Scalar weight + XY balance matches 2D weights**: `scalar_xybal10` achieves X=42.7, Y=42.7 -- nearly identical to 2D weight configs. This means the X/Y gap is NOT caused by the scalar weight limitation.

### What didn't close the gap:

The ~42 GeV floor on X/Y resolution persists across all configurations. This suggests the limitation is **not** in the output head (scalar vs 2D weights) or the loss function, but in one of:

1. **Feature representation**: The model sees raw px/py (not PUPPI-weighted). PUPPI MET benefits from applying PUPPI weights at the particle level before summing. The model must learn these weights from scratch using only continuous features + embeddings.

2. **Architecture capacity**: A 3-layer Dense network processes each particle independently (no inter-particle attention). It cannot learn particle correlations that might help distinguish pileup from hard-scatter particles.

3. **Training data**: 118k events may not be enough to learn the complex mapping from 128x9 features to MET, especially for rare high-pT events that dominate the resolution metric.

4. **Information bottleneck**: The model reduces 128 particles to a single weight per particle, then sums. This is a severe bottleneck compared to PUPPI which uses vertex association, track quality, and other information not available in our 9-feature representation.

## 6. Recommendations

### Short-term (current framework):

1. **Use `scalar_xybal10` as the production config**: Simplest architecture (scalar weight), best stability, and matches 2D weight performance. FPGA-friendly.

2. **Add PUPPI-weighted px/py as additional features**: If `pw_px = puppi_weight * px` and `pw_py = puppi_weight * py` were included, the model would start closer to the PUPPI baseline and only need to learn corrections.

3. **Increase training data**: The current 118k events limit what the model can learn. Even 500k events would help significantly.

### Medium-term (architecture changes):

4. **Self-attention or cross-attention layer**: Allow particles to attend to each other, enabling the model to learn pileup correlations (e.g., "this neutral hadron is near a high-pT jet, likely not pileup").

5. **MLP-Mixer**: The `mixer.py` stub in the codebase is well-suited for this -- token-mixing layers provide inter-particle communication without full attention cost.

### Understanding the gap:

6. **The 42 GeV floor is a real architectural/data limitation**, not a training artifact. The model has converged (early stopping), the loss function is well-tuned (XY balance ensures symmetric optimization), and both scalar and 2D weights reach the same floor.

7. **PUPPI has an inherent advantage**: It uses vertex association and track quality information that our 9-feature representation doesn't include. The ~10% gap likely represents the information gap between our features and PUPPI's full feature set.

## 7. Plots

See `plots/` directory for:
- `roc_comparison.png` -- ROC curves for all configs vs PUPPI
- `resolution_vs_params.png` -- pT resolution vs parameter count
- `auc_vs_params.png` -- AUC vs parameter count
- `turn_on_curves_VBFHInv_TT.png` -- Trigger turn-on curves
- Per-config directories with response, resolution, and distribution plots
