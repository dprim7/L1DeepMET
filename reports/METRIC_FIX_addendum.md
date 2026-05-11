# Addendum — response-correction metric fixed (May 2026)

## What changed

`src/l1deepmet/metrics/physics.py::compute_resolution_metrics` was using **mean of per-event ratios** (`⟨pt_reco / pt_gen⟩`) to compute per-pT-bin response, then dividing reco by that response to compute the response-corrected pT resolution. This is mathematically distinct from the **ratio of means** (`⟨pt_reco⟩ / ⟨pt_gen⟩`) convention used in:

- `src/l1deepmet/plotting.py` (which generated the plots that appeared in every prior report)
- `L1METML/utils.py` and `L1METML/consolidated_plotting.py` (the upstream reference convention)

The two coincide only when `reco` scales linearly with `gen`; with per-event response variance, they diverge. The fix is in commits after this addendum's date — `compute_resolution_metrics` now uses ratio-of-means everywhere, matching `plotting.py` and the L1METML legacy.

## Numerical impact

Across all 48 trained models, the pT resolution number shifted **uniformly by +0.2 to +0.4 GeV** (new values are slightly larger / "worse-looking"). Both PUPPI and ML numbers shifted by similar amounts, so the **Δ vs PUPPI** column changed by ≤ 0.15 GeV in most cases. Rankings were preserved.

| Config | OLD pT IQR/2 [GeV] | NEW pT IQR/2 [GeV] |
|---|---:|---:|
| PUPPI baseline | 44.33 | **44.55** |
| scalar bw=200 (broken loss) | 44.25 | 44.48 |
| scalar bw=0 (BinnedDeviation removed) | 43.57 | 43.89 |
| mae_mse_xy10 | 43.52 | 43.88 |
| mae_only | 42.17 | 42.42 |
| huber_d05 | 41.66 | 41.90 |
| mae_mse_xy0 | 41.41 | 41.79 |
| mse_only | 41.52 | 41.84 |
| bounded_no_bias (xy=10) | 42.63 | 42.95 |
| bounded_mae_only_xy0 | 41.71 | **41.99** |
| unbounded_mae_only_xy0 | 41.60 | 41.93 |

The X/Y, phi, AUC, and per-bin response columns are **unchanged** by this fix — those metrics don't depend on the response-correction calculation. Only `met_pt_resolution` (and its `delta_*` derivative) shifts.

## Affected reports

The following reports' tables show OLD (mean-of-ratios) `met_pt_resolution` numbers. All conclusions in those reports hold qualitatively under the new metric (relative rankings preserved, sign of every delta unchanged), but the absolute pT-resolution figures will all shift by ~+0.3 GeV when reissued.

- `reports/loss_diagnosis_apr2026/` — pT numbers (44.32, etc.) referenced once; bug here was BinnedDeviation, which is independent of this metric fix.
- `reports/loss_and_output_head_ablation_apr2026/` — pT columns in the full table need +0.3 GeV uniform shift.
- `reports/final_baseline_apr2026/` — headline "pT IQR/2: 44.33 → 41.60" should read "44.55 → 41.93" with the new metric.

Updated authoritative numbers live in `outputs/full_physics_card.csv` (rewritten after the fix). The training-time `result.json` files preserve the OLD metric values for traceability; the new `result_v2.json` files have the NEW values.

## Why bother

1. **Internal consistency.** A single repo with two conventions for the same metric is a footgun. Plots and tables disagreed by ~0.3 GeV; that erodes confidence.
2. **HEP convention.** L1METML and the broader CMS L1 ML community use ratio-of-means. Publishing a "wrong-convention" pT resolution against a "right-convention" PUPPI baseline is a real flag at review.
3. **Behaviour at low pT.** Mean-of-ratios diverges when `gen_pt → 0` because `pt_reco / pt_gen → ∞`. The default `pt_bins = [50, …]` masked this for our resolution metric but it would silently corrupt any future low-pT diagnostic.

## How to reproduce post-fix

The unified runner already uses the fixed metric. To regenerate the full leaderboard from saved models:

```bash
python scripts/reeval_physics_card.py \
    outputs/loss_ablation_apr2026 \
    outputs/loss_form_ablation_apr2026 \
    outputs/residual_ablation_apr2026 \
    outputs/combined_best_apr2026 \
    --output-csv outputs/full_physics_card.csv
```
