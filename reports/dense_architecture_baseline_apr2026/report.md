# L1DeepMET Dense Architecture Baseline Study

**April 2026 | D. Primos Castellanos**

---

## 1. Objective

Establish baseline performance for Dense neural network architectures for L1 Missing Transverse Energy (MET) reconstruction at the HL-LHC, and determine whether the current framework is ready for systematic architecture exploration. This study sweeps over network width, depth, categorical embeddings, training mode, and loss hyperparameters, then evaluates the top candidates with proper physics metrics against the PUPPI MET baseline.

## 2. Setup

### Data
- **Samples**: TT_PU200 (100k events) + VBFHToInvisible_PU200 (48k events)
- **Splits**: 118.4k train / 14.8k val / 14.8k test
- **Input**: 128 PUPPI particle candidates per event, 9 features each
  - 5 continuous: pT, eta, phi, PUPPI weight, HGCal depth
  - 2 momentum: px, py
  - 2 categorical: encoded pdgId (6 classes), encoded charge (4 classes)
- **Targets**: Generator-level MET (px, py) in GeV

### Architecture Search (Phase 1)
25 configurations trained for 30 epochs each:

| Axis | Values | Rationale |
|------|--------|-----------|
| Width | 32, 64 | FPGA resource constraints |
| Depth | 2, 3, 4 layers | Capacity vs latency tradeoff |
| Embeddings | yes / no | pdgId encodes particle type affecting detector response |
| Training mode | 0 (direct regression), 1 (per-particle weight) | Mode 1 exploits MET = -sum(w * p) |
| Activation | ReLU, ELU | Dead neuron avoidance for sparse data |
| BinnedDeviation weight | 100, 200 | Physics-informed loss term strength |

**Loss**: MAE + MSE + BinnedDeviation (asymmetry in pT bins [50, 100, 200, 300, 400, inf] GeV)

**Training**: AdamW (lr=1e-3, clipnorm=1.0), ReduceLROnPlateau, EarlyStopping (patience 10), normfac=100.

### Rigorous Evaluation (Phase 2)
5 representative architectures retrained 3x each (seeds 42, 123, 456) for 50 epochs, evaluated on the held-out test set with physics metrics and compared against PUPPI baseline.

## 3. Architecture Search Results

**Full results**: `arch_search_results.csv` (25 configs)

### Key finding: Per-particle weighting (mode 1) strongly dominates direct regression (mode 0)

Every mode-1 configuration outperformed every mode-0 configuration on training val_loss. This validates the physics-informed inductive bias: learning per-particle correction weights and summing weighted momenta (MET = -sum(w_i * p_i)) is fundamentally better than trying to directly regress global MET from pooled features.

### Search ranking was dominated by noise

Epoch-to-epoch val_loss variance within a single training run was large enough that the ranking across configs was unreliable. For example, `noemb_w32_d2_m1` fluctuated between val_loss 0.060 and 0.139 in its last 5 epochs alone (2.3x range). This motivated the multi-seed evaluation in Phase 2.

## 4. Physics Evaluation Results

### 4.1 Summary Table

> **Plots**: `resolution_vs_params.png`, `auc_vs_params.png`

| Config | Params | pT Res [GeV] | X Res [GeV] | Y Res [GeV] | Response | Phi Res [rad] | AUC |
|--------|--------|:-------------|:-----------|:-----------|:---------|:------------|:----|
| **PUPPI baseline** | -- | **44.3** | **38.3** | **38.4** | 0.88 | **1.17** | **0.974** |
| noemb/w32/d3/m1 | 2,721 | 47.3 +/- 0.2 | 87.8 +/- 0.5 | 87.4 +/- 0.2 | 1.01 | 2.82 +/- 0.00 | 0.972 +/- 0.001 |
| emb/w32/d4/m1 | 4,129 | 47.2 +/- 0.7 | 87.9 +/- 0.3 | 88.0 +/- 0.7 | 1.02 | 2.82 +/- 0.00 | 0.971 +/- 0.001 |
| noemb/w64/d2/m1 | 5,121 | 46.8 +/- 0.4 | 57.8 +/- 19.6 | 57.8 +/- 20.0 | 0.99 | 1.73 +/- 0.77 | 0.974 +/- 0.001 |
| emb/w64/d4/m1 | 14,369 | 47.5 +/- 0.4 | 73.2 +/- 20.9 | 73.6 +/- 21.1 | 0.98 | 2.27 +/- 0.78 | 0.968 +/- 0.006 |
| noemb/w64/d4/m0 | 14,018 | 55.1 +/- 13.1 | 63.9 +/- 13.2 | 84.9 +/- 6.9 | 0.98 | 2.14 +/- 0.01 | 0.923 +/- 0.068 |

Resolution = (p84 - p16)/2 of response-corrected residuals, for events with gen MET pT > 50 GeV.

### 4.2 What the Models Learn Well

> **Plots**: `best_model_MET_response.png`, `roc_comparison.png`

**Response calibration is excellent.** All mode-1 models achieve mean response ~1.0 across pT bins (50-400 GeV), compared to PUPPI's systematic 12% underestimation (response = 0.88). The network successfully learns per-particle correction weights that compensate for detector effects and pileup.

**Trigger-level AUC matches PUPPI.** Mode-1 models achieve AUC 0.968-0.974, comparable to PUPPI (0.974), for separating high-MET signal (VBFHToInvisible, gen MET > 200 GeV) from background (TT, gen MET < 50 GeV).

### 4.3 What the Models Fail to Learn

> **Plots**: `best_model_XY_resolution.png`, `fpga_candidate_XY_resolution.png`, `turn_on_curves_VBFHInv_TT.png`

**Directional (phi) reconstruction is unreliable.** This is the critical failure mode:
- PUPPI achieves phi resolution of 1.17 rad consistently
- ML models get either ~1.17 rad (good seeds) or ~2.82 rad (bad seeds, essentially random)
- The variance is **not across architectures but across random seeds**: the same architecture trained with different initialization can either learn or not learn angular structure
- Small models (width 32) *never* learn phi, always getting ~2.82 rad across all seeds

**X/Y resolution is consequently worse.** Since MET_x = MET_pT * cos(phi) and MET_y = MET_pT * sin(phi), poor phi resolution directly degrades X/Y resolution. Models that fail on phi get X/Y resolution ~88 GeV (vs PUPPI's 38 GeV), while models that succeed get ~44 GeV (close to PUPPI).

**pT resolution does not beat PUPPI.** Best ML: 46.8 GeV vs PUPPI: 44.3 GeV. The per-particle weighting learns the magnitude well but not better than PUPPI's direct momentum sum.

### 4.4 Training Stability

**Mode 0 is highly unstable.** The direct regression baseline (`noemb_w64_d4_m0`) shows pT resolution varying from 45.6 to 73.7 GeV across seeds, and AUC from 0.83 to 0.97. One in three seeds essentially fails to train.

**Mode 1 is stable on pT but not on direction.** Per-particle weighting consistently achieves ~47 GeV pT resolution regardless of seed, but angular learning is bimodal (works or doesn't).

**Width 64 is needed for directional learning.** Width-32 models consistently fail on phi (2.82 rad across all seeds). Width-64 models succeed on some seeds (~1.17 rad) but not all. This suggests the directional signal requires sufficient network capacity to capture.

## 5. Diagnosis: Why Does Phi Learning Fail?

The per-particle weighting approach (mode 1) predicts a scalar weight w_i per particle, then computes MET = sum(w_i * [px_i, py_i]). The weight is always positive-or-negative uniformly for both px and py of the same particle. This means:

1. **The weight can scale magnitude but not rotate direction.** To correct angular information, the model would need to learn different corrections for px vs py, which a single scalar weight cannot do.
2. **The loss function (MAE + MSE on px, py) doesn't explicitly penalize angular errors.** A model that gets the pT scale right but the angle wrong can still achieve reasonable loss values.
3. **Width-32 models lack capacity** to represent the complex mapping from 5 continuous features to a weight that even partially captures directional corrections.

## 6. Recommendations and Roadmap

### Immediate (next 2 weeks)
1. **Add phi-aware loss term** -- explicitly penalize angular error (e.g., 1 - cos(dphi) weighted by pT) to force directional learning
2. **Try mode with separate px/py weights** -- predict w_x, w_y per particle instead of a single scalar, allowing directional corrections
3. **Increase training epochs** -- 50 epochs with early stopping (patience 10) may be insufficient; try 200 epochs with patience 30
4. **Investigate PUPPI weight as direct feature** -- the model may not be effectively using PUPPI weights for pileup subtraction

### Medium-term (1-2 months)
5. **Implement attention mechanism** -- self-attention across particles can capture inter-particle correlations (e.g., back-to-back jets) that a per-particle dense model cannot
6. **Implement MLP-Mixer** -- the mixer architecture (channel mixing + token mixing) is a middle ground between dense and attention, potentially more FPGA-friendly
7. **Quantization-aware training** -- begin HGQ/QKeras studies to understand how quantization degrades the angular learning (which is already fragile)

### Longer-term (3+ months)
8. **Multi-sample training** -- add more physics processes to the training set for better generalization
9. **HLS4ML synthesis** -- resource/latency estimates for the promising architectures
10. **Comparison with other L1 MET approaches** -- benchmark against existing CMS L1 MET algorithms

## 7. Files in This Report

### Plots
| File | Description |
|------|-------------|
| `roc_comparison.png` | ROC curves: all 5 architectures + PUPPI |
| `turn_on_curves_VBFHInv_TT.png` | Trigger turn-on efficiency at 30 kHz equivalent |
| `resolution_vs_params.png` | pT resolution vs parameter count (3 seeds, error bars) |
| `auc_vs_params.png` | Trigger AUC vs parameter count (3 seeds, error bars) |
| `best_model_MET_response.png` | Response curve for best model (noemb/w64/d2/m1) |
| `best_model_XY_resolution.png` | X/Y resolution in pT bins, ML vs PUPPI |
| `best_model_pt_resolution.png` | pT and phi resolution in bins |
| `best_model_MET_pt_distribution.png` | MET pT distribution: truth vs ML vs PUPPI |
| `fpga_candidate_MET_response.png` | Response curve for FPGA-sized model (noemb/w32/d3/m1) |
| `fpga_candidate_XY_resolution.png` | X/Y resolution for FPGA-sized model |
| `mode0_MET_response.png` | Response for direct regression (mode 0) baseline |
| `mode0_XY_resolution.png` | X/Y resolution for mode 0 (shows instability) |

### Data
| File | Description |
|------|-------------|
| `evaluation_summary.csv` | Mean +/- std across 3 seeds per architecture |
| `puppi_baseline.csv` | PUPPI baseline metrics |
| `arch_search_results.csv` | Full 25-config architecture search results |

### Reproducing
```bash
# Architecture search (25 configs x 30 epochs)
python scripts/arch_search.py --data-dir preprocessed/25Jul8_140X_v0 --output-dir outputs/arch_search

# Full evaluation (5 configs x 3 seeds x 50 epochs + physics metrics + plots)
python scripts/evaluate.py --data-dir preprocessed/25Jul8_140X_v0 --output-dir outputs/evaluation --epochs 50

# Re-evaluate without retraining
python scripts/evaluate.py --data-dir preprocessed/25Jul8_140X_v0 --output-dir outputs/evaluation --skip-training
```
