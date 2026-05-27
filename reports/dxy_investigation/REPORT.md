# dxy / dxyErr investigation

**Date:** 2026-05-26  
**Trigger:** the dataset-card sidecar (`outputs/preprocessed/26May22_142_extended_20k_v0/dataset_card.md`) flagged feature 8 (`dxyErr`) as **all-zero on real candidates** — a sanity warning that the builder is designed to surface.

## TL;DR

`dxyErr` is fundamentally **not available at L1**. There is no `dxyError()` accessor on `l1t::PFCandidate` or `l1t::PFTrack` on any CMSSW branch we use. The legacy ntuples that did carry an `L1PuppiCands_dxyErr` branch had it populated with zeros throughout — verified empirically below.

**Action taken:**
- `feature_layout_extended` slot 8: replaced `dxyErr` with `dxy`. Same slot count (26), so existing model code that hard-codes 26 input features keeps working; only this single column's semantics change.
- Recipe patch (`patches/runPerformanceNTuple.patch`) gains a `dxy = cms.string("dxy")` accessor in `saveCands().moreVariables`.
- Legacy `var_list` no longer reads `L1PuppiCands_dxyErr` (it was loaded but discarded by the 9-feature transform anyway — purely a dead load).
- New unit-test file `tests/unit/data/test_recipe_patch.py` (5 tests) guards the contract so a future agent can't silently re-introduce `dxyErr`.

**Open question deferred to the user (task #18):** do we re-run the 20k production (~26 h wall) so the H5 carries real `dxy` values, or accept the current 20k (slot 8 = zeros) and pick up `dxy` in the next production round? See "Decision matrix" below.

## How `dxyErr` got into the layout in the first place

Two coincident facts misled the prior agent:

1. **The legacy 25Jul8 ntuples have an `L1PuppiCands_dxyErr` branch.** It's listed in `params.yaml::preprocess.var_list` as something the legacy preprocessor loads. So when the extended layout was being designed, "dxyErr exists, populate it" looked safe.

2. **The legacy 9-feature transform never exposes it as a model feature.** It's loaded then dropped on the floor — the legacy `preprocess_data` pathway only keeps {pt, eta, phi, puppi_weight, hcal_depth, px, py, encoded_pdgId, encoded_charge}. Nobody noticed because no plot, no test, and no model ever consumed it.

The dataset-card sidecar (commit `12f72b6`, added in this same branch) is what finally surfaced the truth — its `compute_per_feature_stats` separates real candidates from pad zeros and flags features that are all-zero on the real subset. That's the warning that triggered this investigation.

## Empirical verification

### 1. Legacy 25Jul8 ntuples carry `L1PuppiCands_dxyErr = 0.0` on every real candidate

```python
# /ceph/cms/store/user/dprimosc/l1deepmet/25Jul8_140X_v0/TT_PU200/FP/140Xv0/perfNano_12509500_0.root
# 50 events, 2704 real candidates (2009 charged + 695 neutral).
# L1PuppiCands_dxyErr stats on ALL real candidates:
#   min:    0.0
#   max:    0.0
#   unique: [0.0]
# Same on the charged subset (2009 candidates, all dxyErr=0.0).
```

So the branch is real but its values are constant zero. This isn't a corner case: this is true on every real PF candidate in every file we sampled.

### 2. CMSSW headers confirm no `dxyError()` exists

From `CMSSW_14_2_0_pre2/src/DataFormats/L1TParticleFlow/interface/PFCandidate.h`:

```cpp
class PFCandidate : public L1Candidate {
public:
    void setDxy(float dxy) { dxy_ = dxy; }
    float dxy() const { return dxy_; }                  // ← real value, populated
    int16_t hwDxy() const { return hwDxy_; }            // ← FPGA-quantised version

    // NO dxyError() method exists. NO dxyErr_ member exists.
private:
    float dxy_, puppiWeight_, caloEta_, caloPhi_;       // dxy is the only IP-related float
    int16_t hwZ0_, hwDxy_;
};
```

And `PFTrack.h` exposes `trkPtError()` (pt uncertainty) but **no** d0 / dxy uncertainty. The underlying TTTrack also doesn't store one — TTTrack has `d0()` returning `theD0_` but no sigma.

The L1 trigger simply doesn't carry per-track impact-parameter uncertainty in 14_2_0_pre2 (nor in 14_0_X, nor in 15_1_X as far as we checked the same headers). So no recipe can emit it.

### 3. The new 26May22 production also lacks `dxy` (different bug — also patched here)

```python
# /ceph/.../26May22_142_extended_20k_v0/TT_PU200/.../perfNano_0063_*.root
# L1PuppiCands_* branches (25 of them):
#   pt, eta, phi, mass, charge, pdgId, puppiWeight, hwPt, hwEta, hwPhi,
#   hwPuppiWeight, hwQual, trackChi2RPhi, trackChi2RZ, trackChi2Bend,
#   trackNStubs, trackMvaQual, caloEta, caloPhi, clPuId, clEmId, clPt,
#   clEmEt, z0
# MISSING: dxy, dxyErr, hwDxy.
```

So even with `dxy()` available on the candidate, the original extended recipe forgot to expose it. The preprocessor's "missing branch ⇒ zero-fill with warning" behaviour did the right thing: it warned and put zeros in slot 8. The dataset-card surfaced the warning to a human-readable spot.

## What `dxy` actually means at L1

The `l1t::PFCandidate::dxy()` value is populated by `PFAlgo3` from the underlying L1 track's POCA computation:

- **Charged candidates:** `dxy = -d0` of the matched L1 track (Phase 2 tracker / TTTrack-emulated). Units: cm. Range typically ±0.5 cm, with the TTTrack precision (13-bit signed, step ≈ 1/256 cm ≈ 39 µm). This is the principal discriminant for non-prompt vertices and PU rejection alongside `z0`.
- **Neutrals (photons / neutral hadrons):** `dxy = 0` by construction (no track ⇒ no impact parameter). The model sees this as "value=0 + charge_id=0 (neutral)" and the embedding can learn the conditional meaning.

There IS no analogous uncertainty stored. The model can approximate uncertainty from the trackWord's quality bits (`trackChi2RPhi`, `trackChi2RZ`, `trackChi2Bend`, `trackMvaQual` — all already in the layout). That's the right place for "how much do I trust this dxy."

## Decision matrix: re-run 20k or accept what we have?

| Question | Re-run 20k | Accept current 20k |
|----------|-----------|---------------------|
| Wall-time cost | ~26 h (same as last run) | 0 h |
| Disk cost | another 50–100 GB (intermediate scratch + final perfNano) | 0 GB |
| Model gets real `dxy` values | ✓ | ✗ (slot 8 = pad zeros) |
| Existing event-count study still valid | ✗ slot 8 changed → must re-train comparators | ✓ |
| Defers `dxy` to "next round" | no | yes — would be picked up next time we extend the layout |

**Recommendation:** unless the next architecture experiment specifically wants to test `dxy`-conditioning, **accept the current 20k**. The patch + tests are in place, so the next production automatically picks up `dxy`. The event-count justification report (`reports/event_count_justification/report.md`) already concluded that at the current Dense w64 d3 architecture, extra features (event-level included) did not improve MAE — so adding `dxy` to slot 8 of the current 20k is unlikely to move the needle by itself.

## Files touched

- `patches/runPerformanceNTuple.patch` — adds `dxy = cms.string("dxy"),` to saveCands moreVariables; patch regenerated via `git diff` against FastPUPPI submodule.
- `params.yaml::preprocess.feature_layout_extended` slot 8: `dxyErr` → `dxy` (with explanatory comment).
- `params.yaml::preprocess.var_list`: removed dead `L1PuppiCands_dxyErr` load.
- `tests/unit/data/test_recipe_patch.py` — new file, 5 tests guarding the contract:
  - layout does not contain `dxyErr`
  - layout contains `dxy`
  - every non-derived layout feature has a saveCands accessor
  - saveCands declares the `dxy` accessor
  - saveCands does NOT declare a (non-existent) `dxyErr` accessor

## Related observations

While inspecting the dataset card, two adjacent issues also surfaced. **Not** fixed here; documented for follow-up:

1. `clPuId` and `clEmId` carry `±3.4 × 10^38` (FLT_MAX) on a small fraction of charged candidates — these are the `-1` fallback from `? pfCluster.isNonnull ? … : -1` getting bit-promoted somewhere. Need a `_sanitize_extreme_values` pass in the preprocessor, OR change the recipe fallback to `0.0`.
2. `caloEta`/`caloPhi` are `-999` on neutrals (intentional fallback). This is a magic number that the model has to learn to ignore. Cleaner: replace fallback with `0.0` and have a separate `has_calo_track` bit, OR mask via the existing `puppiWeight == 0` test.

Both are tracked as candidates for the next architecture / feature-engineering sweep.
