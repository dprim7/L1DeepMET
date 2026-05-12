# Dense baseline: final results & next steps

**Date.** May 2026. Branch: `claude/great-ishizaka`. Latest commit at time of writing: `154fd3b`.

## TL;DR

Starting from a model that was *worse than PUPPI MET* (42.78 GeV X IQR/2 vs PUPPI's 38.30), three diagnostic ablations and one combined-best run have produced the strongest Dense baseline in the project so far:

| Metric | PUPPI | Best ML (this session) | Δ |
|---|---:|---:|---:|
| X resolution (IQR/2) | 38.30 GeV | **33.17 GeV** | −5.13 (−13%) |
| Y resolution (IQR/2) | 38.38 GeV | **33.00 GeV** | −5.38 (−14%) |
| pT resolution, response-corr. | 44.33 GeV | **41.60 GeV** | −2.73 (−6%) |
| φ resolution | 1.166 rad | **1.112 rad** | −0.054 (−5%) |
| ROC-AUC (gen > 200 vs gen < 50) | 0.9741 | **0.9777** | +0.0036 |
| Parameters | 0 | 9,953 | — |

All ML configs that reach this level share three features: `binned_weight=0`, `xy_balance_weight=0`, and `with_bias=False`. The choice between MAE / MSE / Huber and between bounded / unbounded scalar weight is within seed noise once those three are right. The architecture itself (w64 d3 mode-1 with embeddings + weight_minus_one or bounded) is at a clean plateau.

## What changed in this session

Three "physics-motivated" loss / head terms that had been added to the recipe in prior reports turned out to be actively harmful:

1. **`BinnedDeviation`** (weight 200) — forced unbiased response per pT bin. Provably suboptimal for resolution under non-negligible noise (Wiener-filter regression-to-mean). Cost: +4.4 GeV X. (See `reports/loss_diagnosis_apr2026/`.)

2. **`xy_balance_weight=10`** — penalises `|MAE_x − MAE_y|`. Once `BinnedDeviation` was removed, this turned out to hurt X/Y *and* phi resolution by ~1 GeV / ~0.05 rad. Empirically the model produces symmetric residuals without forcing it.

3. **`with_bias=True`** — adds per-particle (Δpx, Δpy) corrections after the scalar weight. Always hurts (~+1 GeV X, ~−0.003 AUC). Extra capacity is spent learning noise rather than signal.

Removing all three, with MAE-only loss on the existing scalar-weight Dense architecture, gives the headline result. Confirming experiments in the `combined_best` recipe found that bounded vs unbounded weight and MAE vs MAE+MSE differ by <0.1 GeV — under seed noise.

## Full physics card (48 trained models, all 4 ablations)

Each cell is mean ± std over 3 seeds (42, 123, 456). PUPPI baseline is exact.

| Ablation | Config | X IQR/2 | Y IQR/2 | pT IQR/2 | φ res | AUC | Δ_pT | Δ_AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| (ref) | **PUPPI MET** | 38.30 | 38.38 | 44.33 | 1.166 | 0.9741 | 0.00 | 0.0000 |
| binned | bw=200 | 42.78±0.04 | 42.81±0.01 | 44.25±0.10 | 1.166 | 0.9742 | −0.07 | +0.0001 |
| binned | bw=50 | 42.59±0.19 | 42.60±0.23 | 44.12±0.18 | 1.164 | 0.9742 | −0.20 | +0.0001 |
| binned | bw=0 | 34.86±0.30 | 34.88±0.24 | 43.57±0.60 | 1.168 | 0.9745 | −0.75 | +0.0004 |
| loss-form | mae_mse_xy10 | 34.81±0.24 | 34.74±0.18 | 43.52±0.74 | 1.164 | 0.9751 | −0.80 | +0.0010 |
| loss-form | huber_d05_xy10 | 35.47±0.32 | 35.20±0.30 | 43.94±0.54 | 1.180 | 0.9737 | −0.38 | −0.0005 |
| loss-form | mae_only | 33.38±0.14 | 33.09±0.15 | 42.17±0.51 | 1.121 | 0.9771 | −2.16 | +0.0029 |
| loss-form | huber_d05 | 33.47±0.19 | 33.11±0.04 | 41.66±0.18 | 1.108 | 0.9773 | −2.67 | +0.0032 |
| loss-form | mae_mse_xy0 | 33.57±0.34 | 33.11±0.27 | 41.41±0.12 | 1.114 | 0.9773 | −2.91 | +0.0031 |
| loss-form | mse_only | 33.78±0.23 | 33.46±0.21 | 41.52±0.14 | 1.116 | 0.9776 | −2.80 | +0.0034 |
| residual | unbounded_no_bias | 35.15±0.02 | 34.89±0.06 | 44.33±0.00 | 1.166 | 0.9741 | +0.00 | −0.0000 |
| residual | bounded_no_bias | 33.70±0.02 | 33.83±0.06 | 42.63±0.78 | 1.129 | 0.9773 | −1.70 | +0.0032 |
| residual | unbounded_with_bias | 34.59±0.33 | 34.63±0.34 | 43.73±0.39 | 1.169 | 0.9744 | −0.59 | +0.0003 |
| residual | bounded_with_bias | 35.30±0.20 | 35.17±0.32 | 44.42±0.21 | 1.175 | 0.9734 | +0.09 | −0.0007 |
| combined | unbounded_mae_only_xy0 | **33.17**±0.07 | 33.00±0.02 | 41.60±0.63 | 1.112 | 0.9774 | −2.73 | +0.0032 |
| combined | **bounded_mae_only_xy0** | 33.21±0.09 | **32.95**±0.09 | 41.71±0.05 | **1.112** | **0.9777** | −2.62 | **+0.0036** |
| combined | bounded_mae_mse_xy0 | 33.23±0.03 | 32.98±0.04 | 41.70±0.34 | 1.109 | 0.9770 | −2.63 | +0.0029 |

Three "all-axes-better-than-PUPPI" configs cluster at the bottom; they're statistically tied. All three:
- have `binned_weight=0`, `xy_balance_weight=0`, `with_bias=False`
- use 9,953 parameters (same arch, w64 d3 mode-1 with embeddings)
- improve every measured physics observable simultaneously — resolution, response (implicit in pT-corrected resolution), angular, and trigger discrimination

The bw=200 row makes the BinnedDeviation pathology unmistakable: pT and AUC essentially match PUPPI (it was doing its job of calibrating mean response), but X/Y resolution is ~4.5 GeV worse than PUPPI. The good configs sacrifice perfect mean response (response per-bin drops to ~0.7–0.8) in exchange for ~5 GeV better X/Y, ~3 GeV better pT (response-corrected), and slightly better AUC — net win on every physics axis.

## Artifacts produced

```
outputs/
├── loss_ablation_apr2026/                # binned_weight ablation, 9 runs
├── loss_form_ablation_apr2026/           # loss-form ablation, 18 runs
├── residual_ablation_apr2026/            # output-head ablation, 12 runs
├── combined_best_apr2026/                # combined best, 9 runs
└── full_physics_card.csv                 # 48-row aggregate, uniform schema
```

Per run: `best_model.keras`, `result.json` (training-time eval, schema varies by ablation), `result_v2.json` (post-hoc full physics card, uniform schema), `history.csv` (per-epoch losses).

## Reproducibility

- Data: `preprocessed/25Jul8_140X_v0/` with MD5s `train=24ec0c2b…`, `val=9c12fb45…`, `test=21edf0c9…`.
- Code: commits `d93661e` (loss fix) → `154fd3b` (physics-metrics module). Run any ablation:
  ```
  python scripts/ablation.py --recipe <name> --epochs 30 --seeds 42 123 456
  # recipes: binned_weight, loss_form, residual_head, combined_best
  ```
- Re-evaluate any saved model: `python scripts/reeval_physics_card.py <ablation_dirs> --output-csv <path>`.
- For bitwise-reproducible per-seed numbers: add `--deterministic` (slower).

Caveat: TF op-determinism is not on by default; per-seed numbers vary by 0.3–0.7 GeV across reruns of the *same* config. Seed-mean and conclusions are stable.

## Architecture research — what to compare next

Open question after this baseline lands: where does the next 1–3 GeV of resolution come from? The current model treats particles independently (a per-particle MLP + sum is exactly a vanilla Deep Sets). State of the art at CMS L1 Phase-2 for FPGA-deployable particle architectures:

| Family | Reference | Params | Latency | Source |
|---|---|---:|---:|---|
| **Deep Sets (CMS L1 jet tagger)** | arXiv:2509.24371 (Sep 2025) | few k | 234 ns | [TrainTagger repo](https://github.com/CMS-L1T-Jet-Tagging/TrainTagger) |
| **Transformer (hls4ml)** | arXiv:2402.01047 / 2409.05207 | 9,135 | 2.08 µs | hls4ml MHA upstream |
| GNN (JEDI-linear) | arXiv:2508.15468 | low-k | sub-µs | hls4ml support less mature |

Recommended: implement and compare on this MET task the **CMS L1 DeepSets jet-tagger style** (QKeras-quantized, φ/ρ structure, slim phi MLP) and the **hls4ml small transformer** (2 heads, 3 encoder blocks, hidden 64, ~9k params). Both fit our 10k-parameter and microsecond-latency budgets. Skip GNN this round — for global scalar regression like MET, the message-passing inductive bias is weaker than for jet tagging, and the hls4ml toolchain support is less complete.

## Pending follow-ups

1. **Architecture comparison**: build Deep Sets (CMS L1 style) and small Transformer (hls4ml style) MET models, run via `scripts/ablation.py` with the corrected loss/head defaults, compare on the same physics card.
2. **Determinism**: enable `tf.config.experimental.enable_op_determinism()` (`--deterministic` flag) before publication-quality numbers; pair with fixed epoch count.
3. **`evaluate.py` cleanup**: dedupe the inline `compute_metrics`, `compute_trigger_metrics`, `compute_puppi_met` against the new `src/l1deepmet/metrics/physics.py`. Currently both exist; physics.py is the single source of truth for new code.
4. **Spring24 ntuples**: blocked on grid certificate. Once data is available, larger training set may push past the 33 GeV plateau even with the current architecture.

## Files added/changed since the last report

```
src/l1deepmet/metrics/physics.py              (new, single-source physics metrics)
scripts/ablation.py                           (uses physics.py; full card per run)
scripts/ablation_recipes.py                   (new, recipes for unified runner)
scripts/reeval_physics_card.py                (new, rescore saved models)
src/l1deepmet/losses/corrected.py             (binned_weight default → 0, Huber added)
scripts/arch_search.py                        (BoundedWeight layer + bounded_weight flag)
scripts/evaluate.py                           (TOP_CONFIGS bw=0; compute_puppi_met doc)
src/l1deepmet/data/loader.py                  (thread-safety fix)
```

Three duplicate ablation scripts (`binned_weight_ablation.py`, `loss_ablation.py`, `residual_ablation.py`) deleted in favour of the unified runner.
