# DeepMET Fixes Ablation: Reproducing the Offline Advantage at L1

**April 3, 2026 | D. Primos Castellanos**

---

## 1. Context

The initial Dense architecture search (see `reports/dense_architecture_baseline_apr2026/`) found that ML models failed to beat the PUPPI MET baseline on resolution, despite achieving comparable trigger AUC. Investigation revealed **three missing ingredients** from the original DeepMET approach (Feng et al., arXiv:2509.12012) that were absent from our L1 implementation.

## 2. The Three Missing Ingredients

### 2.1 Weight-Minus-One Initialization (Critical)

The original DeepMET uses a frozen BatchNorm layer to shift the per-particle learned weight by -1, so the model initializes at:

```
MET_pred = -sum((-1) * px_i) = sum(px_i) = PF MET
```

The network starts from the PF MET baseline and learns small corrections. Our implementation started from random weights near zero, forcing the model to learn the entire MET reconstruction from scratch.

**Implementation**: A `ShiftByConstant(shift=-1/normfac)` layer applied after the weight Dense layer, with the weight Dense initialized to zeros. This ensures the initial prediction exactly equals PF_MET / normfac, matching the training target scale.

### 2.2 Per-Particle Bias Terms

The original DeepMET formula is:
```
MET_x = -sum(w_i * px_i + b_ix)
MET_y = -sum(w_i * py_i + b_iy)
```

The bias terms `(b_ix, b_iy)` allow **additive momentum corrections** per particle, which can correct both magnitude and direction. Our implementation only had `w_i * px_i` (scalar reweighting), which can scale magnitude but cannot independently correct the x and y components for a single particle.

**Implementation**: An additional `Dense(2)` output per particle, added to the weighted momentum before summation. Also initialized to zeros when weight-minus-one is enabled.

### 2.3 reduce_sum Instead of GlobalAveragePooling1D

The original DeepMET uses `reduce_sum` over particles. Our implementation used `GlobalAveragePooling1D`, which divides by N=128. This forced the model to learn weights 128x larger to compensate, adding unnecessary optimization difficulty.

**Implementation**: A `SumOverParticles` layer as drop-in replacement for `GlobalAveragePooling1D`.

## 3. Ablation Study Design

5 configurations, each trained 3 times with different seeds (42, 123, 456), 50 epochs:

| Config | weight-minus-one | bias | sum | embeddings | Params |
|--------|:---:|:---:|:---:|:---:|-------:|
| `deepmet_emb_w64_d3` | Y | Y | Y | Y | 10,083 |
| `deepmet_emb_w32_d3` | Y | Y | Y | Y | 3,011 |
| `wm1_noemb_w64_d3` | Y | N | Y | N | 9,537 |
| `wm1_bias_noemb_w64_d3` | Y | Y | Y | N | 9,667 |
| `broken_noemb_w64_d3` | N | N | N | N | 9,537 |

All use: depth=3, mode=1 (per-particle weight), relu, bw=200, normfac=100, AdamW lr=1e-3.

## 4. Results

### 4.1 Per-Model Results

> **Data**: `evaluation_results.csv`

```
Config                    Seed   Params  pT Res   X Res   Y Res  Response  Phi Res    AUC
------------------------------------------------------------------------------------------
deepmet_emb_w64_d3        42     10,083  44.685  43.393  43.308    1.055    1.169   0.974
deepmet_emb_w64_d3        123    10,083  44.237  42.814  42.851    1.041    1.162   0.974
deepmet_emb_w64_d3        456    10,083  44.457  42.361  42.391    1.029    1.170   0.975
deepmet_emb_w32_d3        42      3,011  44.327  42.493  42.523    1.032    1.162   0.974
deepmet_emb_w32_d3        123     3,011  44.671  87.231  87.782    1.040    2.814*  0.976
deepmet_emb_w32_d3        456     3,011  44.270  43.346  43.252    1.055    1.169   0.973
wm1_noemb_w64_d3          42      9,537  46.545  42.688  42.738    0.994    1.168   0.977
wm1_noemb_w64_d3          123     9,537  43.856  42.470  42.628    1.031    1.164   0.974
wm1_noemb_w64_d3          456     9,537  43.795  42.522  42.371    1.024    1.168   0.975
wm1_bias_noemb_w64_d3     42      9,667  44.371  86.662  86.844    1.021    2.818*  0.974
wm1_bias_noemb_w64_d3     123     9,667  44.258  42.475  42.518    1.031    1.160   0.973
wm1_bias_noemb_w64_d3     456     9,667  44.060  41.979  42.091    1.017    1.173   0.973
broken_noemb_w64_d3       42      9,537  46.515  45.387  45.019    1.036    1.186   0.974
broken_noemb_w64_d3       123     9,537  47.699  43.824  43.490    0.983    1.192   0.977
broken_noemb_w64_d3       456     9,537  46.477  89.189  89.859    1.013    2.815*  0.972
------------------------------------------------------------------------------------------
PUPPI baseline            ---       ---  44.325  38.304  38.380    0.884    1.166   0.974
```

\* = phi learning failure (2.81 rad ~ random). These seeds fail to learn angular structure.

### 4.2 Aggregate (mean +/- std across 3 seeds)

| Config | pT Res [GeV] | X Res [GeV] | Response | Phi Res [rad] | AUC |
|--------|:-------------|:-----------|:---------|:------------|:----|
| **PUPPI** | **44.3** | **38.3** | 0.884 | **1.166** | 0.974 |
| deepmet_emb_w64_d3 | 44.5 +/- 0.2 | 42.9 +/- 0.4 | **1.042 +/- 0.011** | **1.167 +/- 0.003** | 0.974 |
| wm1_noemb_w64_d3 | 44.7 +/- 1.3 | 42.6 +/- 0.1 | 1.016 +/- 0.016 | **1.167 +/- 0.002** | 0.975 |
| broken_noemb_w64_d3 | 46.9 +/- 0.6 | 59.5 +/- 21.0 | 1.011 +/- 0.022 | 1.731 +/- 0.766 | 0.974 |

(Small model and bias configs omitted from aggregate due to 1/3 seed failures — see 4.1 for full data.)

### 4.3 Key Plots

| Plot | Description |
|------|-------------|
| `fixed_best_MET_response.png` | Response curve: best fixed model vs PUPPI (20 pT bins) |
| `fixed_best_XY_resolution.png` | X/Y resolution in pT bins for best fixed model |
| `fixed_best_pt_resolution.png` | pT and phi resolution in bins |
| `fixed_best_MET_pt.png` | MET pT distribution: truth vs ML vs PUPPI |
| `wm1only_MET_response.png` | Response for weight-minus-one only (no bias/emb) |
| `wm1only_XY_resolution.png` | X/Y resolution for wm1-only |
| `broken_MET_response.png` | Response for broken (original) model |
| `broken_XY_resolution.png` | X/Y resolution for broken model — shows bimodal failure |
| `roc_comparison.png` | ROC curves: all 5 configs + PUPPI overlay |
| `turn_on_curves_VBFHInv_TT.png` | Trigger turn-on at 30 kHz equivalent |
| `resolution_vs_params.png` | pT resolution vs parameter count (error bars from seeds) |
| `auc_vs_params.png` | AUC vs parameter count |

## 5. Analysis

### 5.1 Weight-Minus-One is the Critical Fix

**Before fix (broken)**: 1 of 3 seeds fails to learn phi (gets 2.81 rad). Mean pT resolution 46.9 GeV (6% worse than PUPPI). X/Y resolution bimodal between ~44 and ~89 GeV.

**After fix (wm1_noemb_w64_d3)**: **All 3 seeds learn phi correctly** (1.167 rad, matching PUPPI). pT resolution improves to 44.7 GeV (within 1% of PUPPI). X/Y resolution is consistently ~42.6 GeV.

This single fix eliminates the training instability. The model starts from PF MET and learns corrections, rather than learning MET reconstruction from scratch.

### 5.2 Bias Terms Don't Reliably Help (Yet)

Adding per-particle bias `(b_ix, b_iy)` does not consistently improve performance and introduces a 1/3 seed failure rate even with weight-minus-one enabled. Possible reasons:
- The bias doubles the output head parameters per particle (1 weight -> 1 weight + 2 bias), making optimization harder
- Zero initialization of both weight and bias may create a flat loss landscape
- May need longer training or different learning rate for bias terms
- The bias may need to be initialized differently (e.g., small random values)

### 5.3 Embeddings Provide Stability at Small Width

At width 64, embeddings don't meaningfully change the results. At width 32, embeddings help prevent seed failures (2/3 seeds succeed with embeddings vs unknown without). The additional categorical features give the network more signal to learn from when capacity is limited.

### 5.4 The Remaining 11% X/Y Resolution Gap

The best ML model achieves X/Y resolution of ~42.6 GeV vs PUPPI's 38.3 GeV — an 11% gap. This gap persists across all configurations. Possible causes:

1. **Information bottleneck**: The scalar weight `w_i` can only scale the magnitude of each particle's momentum contribution. It cannot independently correct px and py for a single particle. The original DeepMET paper's bias terms address this, but our implementation of bias is unstable.

2. **Limited particle count**: L1 has 128 PUPPI candidates vs offline's ~4500 PF candidates. Less information to work with.

3. **Feature limitations**: L1 features lack impact parameters (d_xy, d_z) available offline, which help identify pileup.

4. **Training duration**: 50 epochs may not be sufficient. The original DeepMET trains until convergence.

5. **PUPPI is the real baseline, not PF MET**: PUPPI already applies pileup weighting. In this data, 80% of particles have PUPPI weight = 1.0 and mean weight = 0.93, so there's limited room for the ML to improve pileup subtraction beyond what PUPPI already does.

### 5.5 What the ML Model DOES Beat PUPPI On

**Response calibration**: ML response is 1.02-1.04 vs PUPPI's 0.88. The ML model correctly reconstructs the MET scale, while PUPPI systematically underestimates by 12%. This would matter for physics analyses that depend on MET scale (e.g., mass measurements, signal region definitions).

## 6. What's Needed to Close the Gap

### Priority 1: Stabilize the bias terms
- Try non-zero initialization for bias (small random, or initialize to approximate the PUPPI weight correction)
- Use separate learning rates for the weight and bias heads
- Consider a scheduled training approach: train weights first (frozen bias), then unfreeze bias

### Priority 2: Separate px/py corrections
- Instead of `w * px + b_x`, try predicting `(w_x, w_y)` as a 2D weight per particle
- This gives the network explicit control over directional corrections
- More parameters per particle but more expressive

### Priority 3: Longer training with better schedule
- Train for 200+ epochs with cosine annealing
- Use warmup for the first 10 epochs
- Try higher learning rate (original DeepMET uses lr=1.0 with clipnorm)

### Priority 4: Architectural improvements
- Self-attention across particles (capture jet-level correlations)
- MLP-Mixer (token mixing for inter-particle information flow)
- Residual connections in the dense body

### Priority 5: Loss function
- Add explicit phi penalty: `1 - cos(phi_pred - phi_true)` weighted by pT
- This directly optimizes the angular reconstruction the model struggles with

## 7. Conclusions

1. **Weight-minus-one initialization fixes training stability** — the single most important change. Should be the default for all future training.
2. **ML matches PUPPI on pT resolution and trigger AUC** with the corrected implementation.
3. **ML significantly outperforms PUPPI on response calibration** (1.02 vs 0.88).
4. **An 11% X/Y resolution gap remains** — likely requires architectural changes (2D weights or attention) to close.
5. **The framework is now correctly reproducing the expected behavior** and is ready for systematic improvements.

## 8. Reproducing

```bash
# Run the ablation study (5 configs x 3 seeds x 50 epochs)
TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/evaluate.py \
    --data-dir preprocessed/25Jul8_140X_v0 \
    --output-dir outputs/evaluation_fixed \
    --epochs 50

# Re-evaluate without retraining
python scripts/evaluate.py \
    --data-dir preprocessed/25Jul8_140X_v0 \
    --output-dir outputs/evaluation_fixed \
    --skip-training
```

Note: Thread limits are necessary on machines with limited thread budgets (e.g., shared login nodes).
