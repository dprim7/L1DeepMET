# Ntuple production status — 26 May 2026

## TL;DR

The extended-features ntuple pipeline is fully **built and tested end-to-end**
through the cmsRun layer, but the actual production run is **blocked by data
availability**: every selected Phase2Spring23 PU200 input dataset is currently
100% tape-resident with no disk replicas anywhere in the CMS XRootD federation.

What you can do *today* without waiting:

- **A new H5** (`outputs/preprocessed/26May20_25Jul8_extendedH5_v0/`) re-preprocesses
  the existing 25Jul8 perfNano files into the new 26-feature + 13-event-feature
  schema. The 4 alternative MET algorithms (PuppiMet, PFMet, CaloMet, TKMet —
  including Central variants) become event-level features the model has never
  seen before. The new per-candidate slots (z0, hw*, track*, cl*) and the
  vertex / Layer2 event features stay zero-filled, because the 25Jul8 ntuples
  predate the extended saveCands recipe.

What needs to happen *before* re-producing ntuples with the full new branch set:

- A **Rucio tape recall** of at least one (preferably all) Phase2Spring23 PU200
  dataset(s). Typically 1–3 days. Once recalled, `scripts/ntuple_produce.py
  run --campaign Phase2Spring23 --sample TT_PU200 …` works as-is.

---

## 1. What shipped on `claude/jolly-lalande-3e5006`

```
84495ff fix(ntuple): PU200-only DAS queries + XRootD double-slash + --max-files
5302713 pivot: 14_0_X two-stage ntuple workflow (Phase2Spring23 inputs)
e535bd6 chore(catalog): refresh Phase2Spring24 from DAS
8961fd9 fix(ntuple): make cmsRun wrapper actually able to run
b3f7218 feat(preprocess): add event-level features to extended H5 schema
f8fd885 feat(recipe): add L1 vertex producer and clean Layer2 MET key
```

### Preprocessor (`b3f7218`)

- New `EVENT_FEATURE_BRANCHES` catalogue maps clean feature names to
  ROOT branches + aggregation mode (scalar / first-of-jagged).
- `load_samples_to_numpy_extended()` returns 3-tuple `(features,
  event_features, targets)` per sample.
- `combine_shuffle_split_extended()` threads event features through the
  shuffle+split with the same seed so X/EX/Y rows stay aligned per event.
- `save_h5_files()` writes an `event_features` dataset + `event_feature_layout`
  attr when the extended path is active. Schema is backward-compatible —
  legacy H5s without `event_features` keep working.
- Per-candidate layout grew from 22 → **26 features** (added `dxyErr`,
  `hwEta`, `hwPhi`, `trackChi2Bend`).
- Event-level layout: **13 features** (4 alternative MET algorithms × pt+phi,
  + Puppi Central pt+phi, + 3 vertex columns: lead_z0, lead_sumPt, n_vtx).
  Layer2 MET was originally planned but dropped — 14_0_X has no `addCTL2Met`
  helper.
- 9 unit tests in `tests/unit/data/test_event_features.py` cover the scalar /
  first-of-jagged helper, layout assembly, missing-branch zero-fill, and a
  consistency check that every feature in params.yaml is in the catalogue.

### Patches (`f8fd885`, `5302713`)

Submodule pin moved 15_1_X → 14_0_X after the 15_1_X recipe hit a fundamental
mismatch on Phase2Spring24 MINIAOD (missing L1T re-emulation modules like
`l1tKMTFMuonsGmt`). Two patches land in `external/FastPUPPI`:

- `patches/runPerformanceNTuple.patch` — extends saveCands moreVariables
  with z0/hw*/track*/cl* accessors (guarded `? pfTrack.isNonnull ? … : -1`
  so missing accessors fall back to -1 instead of crashing), adds an
  unconditional `VertexWordFlatTableProducer` for L1 primary vertices,
  and parses sys.argv directly so `cmsRun … inputFiles=… maxEvents=N
  outputFile=…` from our wrapper actually overrides the recipe.
- `patches/runInputs131X.patch` — same sys.argv override on the stage-1
  recipe so parallel workers don't collide on the hardcoded
  `inputs131X.root` output name.

`scripts/apply_ntuple_recipe.sh` now applies / reverts both patches; the
`check` subcommand reports `applied` / `partial` / `not-applied`.

### Wrapper (`5302713`, `84495ff`)

`scripts/ntuple_produce.py run` is now a 2-stage cmsRun driver:

```
Phase2Spring23 MINIAOD  (URL via XRootD)
  └─ stage 1: cmsRun runInputs131X.py inputFiles=… maxEvents=N outputFile=…
        ↓
   inputs131X_<job>.root  (large; lives under $L1DEEPMET_SCRATCH = /tmp/…)
        └─ stage 2: cmsRun runPerformanceNTuple.py inputFiles=file:… outputFile=…
              ↓
       perfNano_<job>.root  → /ceph/.../<tag>/<sample>/FP/<campaign>/
```

Per-job results record separate `stage1_wall_s`, `stage2_wall_s`, and
`stage1_intermediate_size`. New status values: `stage1_failed`,
`stage2_failed`. Intermediate is deleted after stage 2 regardless of
outcome. Resumable via `state.json` (a successful job is never re-run).

XRootD URL builder: `_input_url()` wraps `/store/...` LFNs with
`root://cmsxrootd.fnal.gov//store/...` (double slash is mandatory per
XRootD spec); overridable via `$L1DEEPMET_XROOTD_REDIRECTOR`.

New `--max-files N` flag caps the plan to N input files — for smoke
testing where the default planner would otherwise spread `--n-events 5`
across all 217 catalog files (1 event per file × 217 cmsRun startups).

### Catalog (`e535bd6`, `84495ff`)

`scripts/ntuple_catalog.py refresh` for Phase2Spring23 resolves 5 of 7
target samples (all gated on `*PU200*` in the campaign segment to avoid
silently catching the noPU variant):

| Sample | DAS dataset | n files | n events |
|---|---|---:|---:|
| TT_PU200             | `/TTToSemileptonic_TuneCP5_14TeV-powheg-pythia8/Phase2Spring23DIGIRECOMiniAOD-PU200_Trk1GeV_131X_mcRun4_realistic_v5-v1/GEN-SIM-DIGI-RAW-MINIAOD` | 217 | 299,000 |
| VBFHToInvisible_PU200 | `/VBFHToInvisible_M-125_TuneCP5_14TeV-powheg-pythia8/Phase2Spring23DIGIRECOMiniAOD-PU200_Trk1GeV_131X_mcRun4_realistic_v5-v1/GEN-SIM-DIGI-RAW-MINIAOD` | ~96 | ~100,000 |
| MinBias_PU200        | `/MinBias_TuneCP5_14TeV-pythia8/Phase2Spring23DIGIRECOMiniAOD-PU200_…/GEN-SIM-DIGI-RAW-MINIAOD` | 6631 | 1,988,040 |
| WJetsToLNu_PU200     | `/WJetsToLNu_TuneCP5_14TeV-amcatnloFXFX-pythia8/Phase2Spring23DIGIRECOMiniAOD-PU200_…/GEN-SIM-DIGI-RAW-MINIAOD` | 333 | 98,804 |
| DYToLL_PU200         | `/DYToLL_M-50_TuneCP5_14TeV-pythia8/Phase2Spring23DIGIRECOMiniAOD-PU200_…/GEN-SIM-DIGI-RAW-MINIAOD` | 1672 | 496,196 |
| SingleNeutrino_PU200 | NO Phase2Spring23 dataset matches `/SingleNeutrino*/Phase2Spring23*PU200*/…` | – | – |
| SMS_T1tttt_PU200     | NO Phase2Spring23 dataset matches `/SMS-T1tttt*/Phase2Spring23*PU200*/…` | – | – |

---

## 2. The blocker

Every resolved Phase2Spring23 PU200 dataset is currently **100% tape-resident**:

```
$ dasgoclient --query "site dataset=…TTToSemileptonic…"
T1_DE_KIT_Tape

$ dasgoclient --query "site file=…/ae14baad-…-…root"
T1_DE_KIT_Tape
```

cmsRun smoke-tested on the first TT_PU200 file via `root://cmsxrootd.fnal.gov//…`
and got back:

```
errno=3011, "No servers are available to read the file"
Disabled source: cms-xrd-global.cern.ch:1094
```

`/ceph/cms/store/mc/Phase2Spring23DIGIRECOMiniAOD/` exists at UCSD T2 but the
directory tree is empty (files pruned); the closest "real" Phase2Spring23
samples on local disk are `TT_TuneCP5` (inclusive instead of semileptonic),
`TTTo2L2Nu`, `VBFHToTauTau` — physics is wrong-for-MET for these.

---

## 3. Recommended path forward

### Track A — full new production (proper "last run")

1. Submit a Rucio recall for at least TT_PU200 and MinBias_PU200 (the two T1
   priority samples). Typical wait: 1–3 days.
2. Once on disk: `scripts/ntuple_produce.py run --campaign Phase2Spring23
   --sample TT_PU200 --tag 26MayXX_140X_extended_v0 --n-events 200000 --workers 16`.
   Expected wall: ~12 hr/sample × 5 = ~60h sequential or ~24h staggered if you
   alternate samples to hide stage-1 I/O wait.
3. Preprocess: `scripts/preprocess.py --tag 26MayXX_140X_extended_v0
   --feature-layout extended`. All 26 candidate + 13 event features populated.

### Track B — immediate win (no waiting)

The H5 at `outputs/preprocessed/26May20_25Jul8_extendedH5_v0/` (produced today)
gives the model the new alternative-MET-algorithm event features against the
existing TT + VBFHInv 25Jul8 data. Per-candidate is unchanged (still 8 useful
features padded to 26 with zeros). Good for a sanity-check ablation:

> If `puppi_met_pt / pf_met_pt / calo_met_pt / tk_met_pt` event-level features
> measurably move the MAE/MSE on validation, the underlying hypothesis (event-
> level context matters) is confirmed and worth the full re-production. If
> they don't, the per-candidate extensions can stay deferred too.

### Track C — pivot to a campaign with disk inputs

`/ceph/cms/store/mc/Phase2Spring24DIGIRECOMiniAOD/` has full local copies of
`DYToLL_M-50` and `MinBias_TuneCP5` (Spring24, D110 geometry, 141X GT). Using
these would need:
1. A `runInputs141X.py` (copy of runInputs131X.py with `D110` + `141X_…` GT).
2. The same geometry/GT bump in `runPerformanceNTuple.py`.
3. Acknowledging only 2 samples — TT and VBF still need recall.

Not recommended unless tape recall is a multi-week hold-up.

---

## 4. Track B output — `outputs/preprocessed/26May20_25Jul8_extendedH5_v0/`

```
train.h5  82.7 MB   features (118400, 128, 26)   event_features (118400, 13)   targets (118400, 2)
val.h5    11.4 MB   features (14800, 128, 26)    event_features (14800, 13)    targets (14800, 2)
test.h5   11.5 MB   features (14800, 128, 26)    event_features (14800, 13)    targets (14800, 2)
```

Per-candidate non-zero counts on the test split (14800 events × 128 candidates = 1.89M rows):

```
[ 0] pt              710,651   (~37% of slots are real candidates; rest pad-zeros)
[ 1] eta             708,851
[ 2] phi             710,257
[ 3] puppi_weight    710,651
[ 4] px              710,651
[ 5] py              710,257
[ 6] encoded_pdgId   710,651
[ 7] encoded_charge  710,651
[ 8] dxyErr        1,183,749   (the IP error sentinel 1000 leaks into pad slots — fine)
[ 9-25] z0, hw*, track*, calo*, cl*: 0/1,894,400   (expected — 25Jul8 predates extended saveCands)
```

Event-level features on the test split (14800 events):

```
[ 0] puppi_met_pt          14800   range +0.46  to +1254 GeV
[ 1] puppi_met_phi         14800   range −π     to +π
[ 2] puppi_met_central_pt  14800   range +0.18  to +1256 GeV  (|η| < 2.4 variant)
[ 3] puppi_met_central_phi 14800   range −π     to +π
[ 4] pf_met_pt             14800   range +1.44  to +1162 GeV
[ 5] pf_met_phi            14800   range −π     to +π
[ 6] calo_met_pt           14800   range +1.01  to +1187 GeV
[ 7] calo_met_phi          14800   range −π     to +π
[ 8] tk_met_pt             14800   range +0.75  to +4276 GeV  (note: long tail from bad
                                                              tracks; expect clipping
                                                              upstream in the model head)
[ 9] tk_met_phi            14800   range −π     to +π
[10-12] lead_vtx_*, n_vtx: 0/14800              (expected — needs the patched producer)
```

This is the immediate "Track B" deliverable. Drop it into the same model
recipe (Deep Sets ρ-head or whichever you're using); the model now sees
the 4 alternative-MET algorithm baselines as extra event-level inputs and
can learn to correct from them rather than reconstruct MET from scratch.

If a 1-day ablation on this H5 shows the event-level features measurably
improve the validation MAE/MSE, that motivates investing in the tape
recall + Track A re-production. If they don't, the recall isn't worth
the wait.

## 5. Inventory of what was actually verified end-to-end

- Preprocessor: 9 unit tests green; smoke-tested end-to-end on a 2-sample
  10k-event extract from the 25Jul8 perfNanos. H5 schema with `features`
  (10000, 128, 26), `event_features` (10000, 15), `targets` (10000, 2) and
  all expected attrs.
- DAS refresh: ran successfully against Phase2Spring23 with corrected
  PU200-only queries.
- cmsRun wrapper: dry-run + real smoke (1 file, 5 events). Sub-process
  spawn works, env propagation works, intermediate cleanup works, XRootD
  URL is well-formed and accepted by the redirector. Only fails on the
  upstream file_open due to tape-only storage.
- Patches: apply cleanly on FastPUPPI 14_0_X (d5ac584); verified by
  rebuilding the full CMSSW area (`/home/users/dprimosc/CMSSW_14_0_0_pre3_L1DeepMET/`)
  with our patched recipe and confirming the `VertexWordFlatTableProducer`
  plugin is registered.

## 6. Open follow-ups not addressed

- L1Layer2 MET column (dropped because 14_0_X has no `addCTL2Met`). If/when
  the pipeline moves back to 15_1_X (after upstream catches up), uncomment
  the two `layer2_met_*` entries in `params.yaml::event_feature_layout`.
- `SingleNeutrino_PU200` substitute: MinBias_PU200 covers PU-only baseline.
  True noise-floor requires gen-MET-≡0 events; pure neutrino-gun samples
  don't appear in Phase2Spring23 production.
- `SMS_T1tttt_PU200` substitute: high-MET tail under-represented relative
  to plan §3. VBFHToInvisible covers the genuine-MET-signal regime;
  acceptable for v0.
- HGCal3DCl ΔR join (the "broken legacy hcal_depth"): still zero-filled in
  the extended path. Defer to Phase 3 per the plan.
- HTCondor backend: still local-only. Phase 3.
