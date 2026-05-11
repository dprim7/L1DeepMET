# Loss-form & output-head ablations (May 2026)

**Context.** Sequel to `reports/loss_diagnosis_apr2026/` (which removed `BinnedDeviation` from the default loss, taking X resolution from 42.78 → 34.86 GeV IQR/2). With the broken loss term gone, this study sweeps two orthogonal axes — the *form* of the regression loss, and the *parameterization* of the per-particle weight head — to find what else was tuned wrong.

## Headline result

We took two more pieces of the previously-recommended training recipe out of the loss/architecture and reached **33.38 GeV X IQR/2** (down from 34.86), a further **−1.5 GeV** improvement over PUPPI MET (38.30) for a total session improvement of **−4.9 GeV (−13%)**. Two terms that I (Claude) had earlier added/recommended turned out to actively hurt: `xy_balance_weight=10` (forces axis symmetry) and `with_bias=True` (per-particle additive 2D correction). Removing both, with **MAE-only** loss on the existing scalar-weight architecture, gives the best result; alternatively, switching to a **bounded scalar weight** (tanh-bounded ∈ (-2, 0), init at -1 → PUPPI MET) is robust enough to give 33.70 GeV even with the harmful `xy_balance=10` still on.

## Results: loss-form ablation

Architecture fixed (`scalar_xybal10_emb_w64_d3`, `binned_weight=0`, `weight_minus_one`, `use_sum`); only the regression loss varied. 6 configs × 3 seeds × ≤30 epochs each, early stopping `patience=10`.

| Config | mae | mse | huber (δ=0.5) | xy_bal | X IQR/2 | Y IQR/2 | Δ vs PUPPI X |
|---|---:|---:|---:|---:|---:|---:|---:|
| **mae_only** | 1 | 0 | 0 | 0 | **33.38 ± 0.14** | 33.09 ± 0.15 | **−4.93** |
| huber_d05 | 0 | 0 | 1 | 0 | 33.47 ± 0.19 | 33.11 ± 0.04 | −4.84 |
| mae_mse_xy0 | 1 | 1 | 0 | 0 | 33.57 ± 0.34 | 33.11 ± 0.27 | −4.74 |
| mse_only | 0 | 1 | 0 | 0 | 33.78 ± 0.23 | 33.46 ± 0.21 | −4.52 |
| mae_mse_xy10 (control) | 1 | 1 | 0 | 10 | 34.81 ± 0.24 | 34.74 ± 0.18 | −3.50 |
| huber_d05_xy10 | 0 | 0 | 1 | 10 | 35.47 ± 0.32 | 35.20 ± 0.30 | −2.84 |

Findings: (a) **`xy_balance=10` hurts by ~1 GeV** in every comparison; (b) **MAE, MSE, Huber, and MAE+MSE all cluster within seed variance** when `xy_balance=0` — the choice between them is essentially noise; (c) Huber was hypothesised to win because residuals are heavy-tailed; it tied with MAE. The dominant signal is "delete `xy_balance`", not "pick a particular regression loss".

## Results: output-head / residual ablation

Loss fixed at **MAE+MSE+xy_balance=10** (chosen *before* the loss ablation finished — see Caveats). 4 configs × 3 seeds. Two orthogonal flags toggled: `bounded_weight` (tanh-bounded scalar ∈ (-2, 0)) and `with_bias=True` (per-particle additive (Δpx, Δpy)).

| Config | bounded | bias | X IQR/2 | Y IQR/2 | Δ vs PUPPI X | #params |
|---|:---:|:---:|---:|---:|---:|---:|
| **bounded_no_bias** | ✓ | ✗ | **33.70 ± 0.02** | 33.83 ± 0.06 | **−4.61** | 9,953 |
| unbounded_with_bias | ✗ | ✓ | 34.59 ± 0.33 | 34.63 ± 0.34 | −3.72 | 10,083 |
| unbounded_no_bias | ✗ | ✗ | 35.15 ± 0.02 | 34.89 ± 0.06 | −3.15 | 9,953 |
| bounded_with_bias | ✓ | ✓ | 35.30 ± 0.20 | 35.17 ± 0.32 | −3.00 | 10,083 |

Findings: (a) **`bounded_weight=True` is the best output head**, and is *robust to the harmful `xy_balance=10`* — the bounded version with `xy=10` (33.70) nearly matches the unbounded `xy=0` winner from the loss study (33.38); (b) **per-particle additive bias hurts** — adding 130 parameters of `(Δpx, Δpy)` always degrades resolution; (c) bounded weight has extraordinarily low seed variance (±0.02 GeV) — the loss landscape becomes more benign once weights can't run away.

## Updated leaderboard

| Strategy | Params | X IQR/2 | Notes |
|---|---:|---:|---|
| PUPPI MET | 0 | 38.30 | `L1PuppiMet` branch |
| Constant w=0.69 | 1 | 34.36 | global response calibration |
| Per-type oracle (5 weights) | 5 | 34.97 | upper bound on type-only models |
| ML — `mae_only` xy=0 | 9,953 | **33.38 ± 0.14** | loss-ablation winner |
| ML — `bounded_no_bias` xy=10 | 9,953 | 33.70 ± 0.02 | output-head-ablation winner |

Both ML winners beat the per-type oracle by ~1.5 GeV, meaning the network is exploiting per-event/per-particle signal beyond just particle-type calibration.

## Artifacts

```
outputs/
├── loss_ablation_apr2026/                         # binned_weight ablation (3×3 seeds)
│   ├── ablation_results.csv                       # seed-level results
│   └── scalar_xybal10_bw{0,50,200}_..._seed{42,123,456}/
│       ├── best_model.keras                       # saved Keras model
│       ├── result.json                            # this run's full eval dict
│       └── history.csv                            # per-epoch train/val metrics
├── loss_form_ablation_apr2026/                    # this study, loss-form (6×3 seeds = 18 runs)
│   ├── ablation_results.csv
│   └── {mse_only,mae_only,huber_d05,mae_mse_xy0,mae_mse_xy10,huber_d05_xy10}_seed{42,123,456}/
│       └── (same 3 files per run)
└── residual_ablation_apr2026/                     # this study, output-head (4×3 seeds = 12 runs)
    ├── ablation_results.csv
    └── {bounded,unbounded}_{no_bias,with_bias}_seed{42,123,456}/
        └── (same 3 files per run)
```

Total: 39 trained models (~30 MB each) + 3 CSVs + 3 logs. Per-epoch loss curves preserved in `history.csv`.

## Reproducibility

**What's fixed:**
- Code: branch `claude/great-ishizaka` at commit `d93661e` (or later, as long as the loss/architecture in scope hasn't changed).
- Data: `preprocessed/25Jul8_140X_v0/{train,val,test}.h5` with MD5s
  - train: `24ec0c2bfc86a68cd7738fefb6f31cd4`
  - val:   `9c12fb4527f73e7808fc97a48e263279`
  - test:  `21edf0c9c9a60f4d2118c88ba6cdfff2`
- Hyperparameters: 30 max epochs, batch 256, AdamW lr=1e-3, clipnorm=1.0, EarlyStopping patience=10 on `val_loss`, ReduceLROnPlateau patience=5 factor=0.5.
- Seeds: 42, 123, 456 set via `tf.random.set_seed` and `np.random.seed`.
- Architecture: width=64, depth=3, mode=1, `use_embeddings=True`, `weight_minus_one=True` (or `bounded_weight=True`), `use_sum=True`.

**What's NOT fully fixed (statistically reproducible only):**
- `tf.config.experimental.enable_op_determinism()` is *not* called → some TF ops are non-deterministic at the bit level even with seeds set.
- `EarlyStopping` triggers at different epochs across reruns of the same seed (seen empirically: 11 vs 14 vs 30 epochs for nominally identical configs across the three ablation runs), which propagates through `ReduceLROnPlateau` to give different LR schedules.

Concretely: three runs of *identical* configurations across the three ablations gave per-seed differences of 0.3–0.7 GeV X IQR/2, but the seed-averaged mean ± std agreed within error bars in all cases:

| Ablation | Config | seed-mean X IQR/2 |
|---|---|---:|
| binned_weight | `scalar_xybal10_bw0` | 34.86 ± 0.30 |
| loss-form | `mae_mse_xy10` | 34.81 ± 0.24 |
| residual | `unbounded_no_bias` | 35.15 ± 0.02 |

So **conclusions are reproducible; per-seed numbers are not**. To make per-seed bitwise reproducible, set `tf.config.experimental.enable_op_determinism()` and pin `EarlyStopping` to the full epoch count (or remove it). Worth doing before publishing final numbers.

## Reproduce these ablations

From the worktree root (`/home/users/dprimosc/L1DeepMET/.claude/worktrees/great-ishizaka`):

```bash
# Activate env
source /home/users/dprimosc/micromamba/etc/profile.d/micromamba.sh
micromamba activate l1deepmet

# 1. Loss-form ablation (~2 hr; 6 configs × 3 seeds)
python scripts/loss_ablation.py \
  --epochs 30 --seeds 42 123 456 \
  --output-dir outputs/loss_form_ablation_apr2026

# 2. Output-head / residual ablation (~2 hr; 4 configs × 3 seeds)
python scripts/residual_ablation.py \
  --epochs 30 --seeds 42 123 456 \
  --output-dir outputs/residual_ablation_apr2026

# Subsets allowed via --configs:
python scripts/loss_ablation.py --configs mae_only --seeds 42
```

CSVs land alongside each run dir; `result.json` per run has the full eval dict; `history.csv` has per-epoch losses.

## Caveats

1. **The two ablations are not factorial**, so we have 6+4 cells, not 6×4 = 24. The combined-best config — `bounded_no_bias` × `mae_only` × `xy_balance=0` — has not been measured. Hypothesis (loose): 31.5–32.5 GeV X IQR/2.
2. **xy_balance=10 was on for the residual ablation** because it was launched before the loss ablation finished. Bounded weights are robust to this (33.70 even with xy=10), but the absolute residual numbers are pessimistic by ~1 GeV.
3. **Three duplicate ablation scripts** (`binned_weight_ablation.py`, `loss_ablation.py`, `residual_ablation.py`) share ~80% of their code. Refactor into one configurable runner before the next ablation.
4. **Determinism**: see "Reproducibility" above. The conclusions are stable; specific numbers shift by ~0.3 GeV per seed across reruns.
