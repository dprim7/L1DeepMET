# Agent-accessible ntuple pipeline for L1DeepMET — design plan

**Goal.** A reproducible, agent-driven workflow that (re)produces L1DeepMET training ntuples from Phase-2 MC samples, emitting **every Layer-2-correlator variable** that could plausibly help MET reconstruction (not just the 9 features currently in `preprocessed/25Jul8_140X_v0/`).

**Reference upstream.** [`p2l1pfp/FastPUPPI` @ `15_1_X`](https://github.com/p2l1pfp/FastPUPPI/tree/15_1_X/NtupleProducer). All file/path references below are relative to that subtree unless stated otherwise.

**Hard prerequisite.** A working CMS grid certificate for the user (currently expired — see `project_grid_cert.md` memory file). Nothing below runs without `voms-proxy-init -voms cms -rfc` succeeding. **All planning here is engineering-ready; execution is blocked on the proxy.**

---

## 1. What we have today vs. what's missing

### 1.1 In `outputs/.../perfNano_*.root` today (156 branches)

Per-candidate (already in our preprocessing's 9-feature representation):

| Branch | Used? | Notes |
|---|---|---|
| `L1PuppiCands_pt` | ✓ | |
| `L1PuppiCands_eta` | ✓ | |
| `L1PuppiCands_phi` | ✓ | |
| `L1PuppiCands_pdgId` | ✓ | encoded → 1-5 |
| `L1PuppiCands_charge` | ✓ | encoded → 0-3 |
| `L1PuppiCands_puppiWeight` | ✓ | informational; pt is already weighted |
| `L1PuppiCands_dxyErr` | ✓ | tried — clipped to 0 in preprocessing |
| `L1PuppiCands_mass` | ✗ | always 0 for L1 |

Per-event MET (we use `pt`/`phi`, the others are unused):

| Branch | Used? | Notes |
|---|---|---|
| `L1PuppiMet_pt`, `_phi`, `_para`, `_perp` | ✓ pt/phi only | `_para`/`_perp` = projection along gen MET |
| `L1PuppiMetCentral_pt`, `_phi`, `_para`, `_perp` | ✗ | same in |η| < 2.4 only |
| `L1PFMet_*`, `L1CaloMet_*`, `L1TKMet_*` (+Central) | ✗ | other MET algorithms — never compared |

Optional per-cluster (45+ variables) — **all currently mis-indexed in our preprocessing**:

| Branch | Used? | Notes |
|---|---|---|
| `HGCal3DCl_pt/eta/phi` | ✗ | |
| `HGCal3DCl_firstHcal{1,3,5}layers`, `showerlength`, `coreshowerlength`, `hoe` | ✓ (broken) | preprocessing reads these but the index in our preprocessing is the candidate-index, not the cluster-index → all values come out 0. **Bug**. |
| `HGCal3DCl_pfPuIdScore`, `pfEmIdScore`, `egEmIdScore`, `pfPuIdPass`, `pfEmIdPass` | ✗ | the actual PUPPI/EM ID scores per cluster |
| `HGCal3DCl_{srrmax,srrmean,srrtot,szz,spptot,sppmax,seetot,seemax,varEtaEta,varPhiPhi,varRR,varZZ}` | ✗ | shower-shape moments |
| `HGCal3DCl_{ebm0,ebm1,emax{1,3,5}layers,emaxe,first{1,3,5}layers,last{1,3,5}layers,layer{10,50,90},maxlayer,firstlayer}` | ✗ | longitudinal shower profile |

### 1.2 Missing from the current ntuples (but available from L1 PUPPI candidate class)

These would need adding to the `saveCands()` `moreVariables` PSet in `runPerformanceNTuple.py`:

| Variable | Why it matters for MET |
|---|---|
| **`z0`** (`vz()`) | **Distance to primary vertex along beam.** The single most important variable PUPPI uses to reject pileup. Right now our model only sees the output `puppiWeight`, not the underlying decision input. |
| **`dz`** (`vz()` − PV.z) | Same as above, vertex-relative. |
| **`hwPuppiWeight`** | Quantised on-FPGA weight (more representative of the actual L1 decision). |
| **`hwQuality`** | Per-candidate quality bits (track-quality flag, etc.). |
| **`pfCluster.pfPuIdScore`** | For neutrals from HGCal: the upstream PU score. Joins HGCal3DCl info to the candidate. |
| **`pfCluster.pfEmIdScore`** | Same for EM ID. |
| **`pfTrack.chi2RPhi`, `chi2RZ`, `nStubs`, `mvaQual`** | For charged: track quality. Lets the model distinguish high-quality from low-quality charged candidates. |
| **`caloEta`, `caloPhi`** | The track's extrapolated position at the calorimeter — useful for matching. |
| **`encEta`, `encPhi`** | Encoded η/φ on the FPGA grid (more representative of what the L1 sees). |

### 1.3 Missing entirely from the ntuple (not in any current branch)

| Variable | Source | Why |
|---|---|---|
| **Vertex word** (`L1VertexWord`) | `VertexWordTableProducer.cc` plugin | PV position; lets the model compute z0_rel itself |
| **Per-region CTL1 input/output multiplicities** | `monitorPerf(..., makeInputMultiplicities="CTL1", makeOutputMultiplicities="CTL1")` | Per-η-region particle counts → event-level features the model currently has to re-derive from the cand list |
| **Decoded tracker (`DecTk*`)** + **decoded calo (`DecHadCalo*`/`DecEmCalo*`)** | `addDecodedTk()`, `addDecodedCalo()` helpers | Pre-PUPPI inputs to Layer-1 PF. Lets the model see what L1 PF rejected, not just what survived. |
| **Standalone muons** (`StaMu`) | `addStaMu()` | Muons not in PUPPI cands — relevant for W→μν events |
| **TkEm / TkEle L2** (`addTkEG(doL2=True)`) | `addTkEG()` | Layer-2 EG objects with isolation flags |
| **L1Layer2 / CTL2 MET** (`addCTL2Met()`) | `addCTL2Met()` | The actual Layer-2 correlator MET — currently NOT in our ntuples. Critical for the user's stated "Layer 2 correlator" framing. |

---

## 2. The "all-possible-variables" ntuple recipe

A patched `runPerformanceNTuple.py` that emits everything plausibly MET-relevant. Concrete diffs:

### 2.1 Extended `saveCands()` (the highest-payoff single change)

```python
def saveCands():
    process.l1pfcandTable = cms.EDProducer("L1PFCandTableProducer",
        commonSel = cms.string("pt > 0.0 && abs(eta) < 10.0"),
        cands = cms.PSet(),
        moreVariables = cms.PSet(
            # Already saved
            puppiWeight = cms.string("puppiWeight"),
            pdgId       = cms.string("pdgId"),
            charge      = cms.string("charge"),
            # NEW — vertex association
            z0          = cms.string("vz"),
            # NEW — quantised/hw values (what the FPGA sees)
            hwPt        = cms.string("hwPt"),
            hwEta       = cms.string("hwEta"),
            hwPhi       = cms.string("hwPhi"),
            hwPuppiWeight = cms.string("hwPuppiWeight"),
            hwQual      = cms.string("hwQual"),
            # NEW — track quality (charged candidates)
            trackChi2RPhi = cms.string("? pfTrack.isNonnull ? pfTrack.trackWord.getChi2RPhi : -1"),
            trackChi2RZ   = cms.string("? pfTrack.isNonnull ? pfTrack.trackWord.getChi2RZ : -1"),
            trackChi2Bend = cms.string("? pfTrack.isNonnull ? pfTrack.trackWord.getBendChi2 : -1"),
            trackNStubs   = cms.string("? pfTrack.isNonnull ? pfTrack.trackWord.getNStubs : -1"),
            trackMvaQual  = cms.string("? pfTrack.isNonnull ? pfTrack.trackWord.getMVAQuality : -1"),
            caloEta       = cms.string("? pfTrack.isNonnull ? pfTrack.caloEta : -999"),
            caloPhi       = cms.string("? pfTrack.isNonnull ? pfTrack.caloPhi : -999"),
            # NEW — cluster quality (neutral candidates from HGCal)
            clPuId   = cms.string("? pfCluster.isNonnull ? pfCluster.egVsPUMVAOut : -1"),
            clEmId   = cms.string("? pfCluster.isNonnull ? pfCluster.egVsPionMVAOut : -1"),
            clPt     = cms.string("? pfCluster.isNonnull ? pfCluster.pt : -1"),
            clEmEt   = cms.string("? pfCluster.isNonnull ? pfCluster.emEt : -1"),
        ),
    )
    monitorPerf("L1PF",    "l1tLayer1:PF",    saveCands=True)
    monitorPerf("L1Puppi", "l1tLayer1:Puppi", saveCands=True)
    process.p += process.l1pfcandTable
```

Caveat: these `pfTrack.isNonnull` / `pfCluster.isNonnull` `StringObjectFunction` accessors must be verified against the L1 `l1t::PFCandidate` class API for 15_1_X. They work in the existing `addPFLep()` helper (lines we already quoted), so the pattern is sound; specific field names may shift slightly between releases.

### 2.2 Unconditionally enable the optional helpers we want

Append at the bottom of `runPerformanceNTuple.py`, after the unconditional `monitorPerf` calls:

```python
saveCands()                      # per-candidate state with new moreVariables
saveGenCands()                   # genParticlesForMETAllVisible — ground-truth particles
addCTL2Met()                     # Layer-2 correlator MET (the user's framing)
addHGCalTPs()                    # HGCal cluster table (already mostly there, formalise it)
addDecodedCalo()                 # pre-PUPPI calo input
addDecodedTk()                   # pre-PUPPI track input
addStaMu()                       # standalone muons
addTkEG(doL2=True)               # Layer-2 EG objects
```

This roughly **doubles** the output ROOT file size per event, from ~50 kB → ~100 kB. For 100k events (our current TT sample size) that's 10 GB per sample — still fine for `/ceph`.

### 2.3 Per-event vertex word

Add to `process.p`:

```python
process.vertexWordTable = cms.EDProducer("VertexWordTableProducer",
    src = cms.InputTag("l1tVertexFinder", "L1Vertices"),
    name = cms.string("L1Vertex"),
)
process.p += process.vertexWordTable
```

Gives `L1Vertex_z0`, `L1Vertex_nTracks`, `L1Vertex_quality` per event. The model can then compute its own z0_rel.

### 2.4 (Optional, defer to phase 2) CTL1 input/output multiplicities

Modify the four default `monitorPerf` calls:

```python
monitorPerf("L1Puppi", "l1tLayer1:Puppi",
            makeInputMultiplicities="CTL1",
            makeOutputMultiplicities="CTL1")
```

Gives per-region counts (`L1Puppi_totNRegPuppi`, `L1Puppi_vecNRegPuppi`, ...) that summarise event activity. Useful as event-level inputs.

---

## 3. Sample plan

### 3.1 Phase-2 Spring24 (current campaign, CMSSW 14_0_X-equivalent)

The MET-relevant set, prioritised:

| Tier | Sample | DAS path (Phase2Spring24DIGIRECOMiniAOD) | Why |
|---|---|---|---|
| **1 (must have)** | `TTToSemileptonic_TuneCP5_14TeV-powheg-pythia8` | `/TTToSemileptonic_TuneCP5_14TeV-powheg-pythia8/Phase2Spring24DIGIRECOMiniAOD-PU200_Trk1GeV_140X_mcRun4_realistic_v4-v2/GEN-SIM-DIGI-RAW-MINIAOD` | Bulk MET training source. Already have. Re-do with the extended ntuple recipe. |
| **1** | `VBF_HToInvisible_M125_TuneCP5_14TeV_powheg-pythia8` | (need DAS query) | Clean genuine-MET signal. Already have. |
| **1** | `MinBias_TuneCP5_14TeV-pythia8` | (need DAS query) | Pure pileup, **no genuine MET** — establishes the false-positive rate floor. Currently missing from our training. |
| **1** | `SingleNeutrino_E-10-gun` | (need DAS query) | Zero-bias-like, **literally zero genuine MET**. Defines model noise floor. Currently missing. |
| **2 (nice to have)** | `WJetsToLNu_TuneCP5_14TeV-amcatnloFXFX-pythia8` | (need DAS query) | W→ℓν: broad MET spectrum, both real signal and pileup contributions. |
| **2** | `DYToLL_M-50_TuneCP5_14TeV-pythia8` | (need DAS query) | Z+jets, no genuine MET — control sample for response calibration. |
| **3 (signal-tail)** | `SMS-T1tttt_TuneCP5_14TeV-pythia8` or any SUSY signal MC | (need DAS query) | High-MET tail (gen MET > 300 GeV), currently under-represented in our training. |

DAS queries to enumerate file paths (run **after** `voms-proxy-init`):

```bash
dasgoclient --query="dataset=/TT*Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD instance=prod/global"
dasgoclient --query="dataset=/VBF*HToInv*Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD"
dasgoclient --query="dataset=/MinBias*Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD"
dasgoclient --query="dataset=/SingleNeutrino*Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD"
dasgoclient --query="dataset=/WJetsToLNu*Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD"
dasgoclient --query="dataset=/DYToLL*Phase2Spring24*/GEN-SIM-DIGI-RAW-MINIAOD"
```

Cache the results in `data/sample_catalog/<tag>.json` so subsequent runs are reproducible without grid access (next section).

### 3.2 Event counts

Match the existing ratio (TT:VBFHinv = 100k:48k). Suggested first-pass targets:

```
TT_PU200            200,000 events   (2× current; gives finer per-pT-bin stats)
VBFHToInvisible      80,000 events   (~current size)
MinBias_PU200        50,000 events   (new — PU-only background)
SingleNeutrino       30,000 events   (new — noise floor)
WJetsToLNu          100,000 events   (new — broad MET)
DYToLL               50,000 events   (new — Z+jets control)
─────────────────────────────────────
TOTAL                510,000 events  (~3.5× current; still 1 day of ntuple-production on grid)
```

---

## 4. Pipeline architecture

The pipeline has three layers: **catalog → produce → preprocess**. An agent can drive any single step or end-to-end.

### 4.1 Catalog layer — `scripts/ntuple_catalog.py` (new)

```bash
# Initialise sample catalog from DAS (requires grid proxy)
python scripts/ntuple_catalog.py refresh --campaign Phase2Spring24

# Inspect what's catalogued (no grid needed)
python scripts/ntuple_catalog.py list
python scripts/ntuple_catalog.py show TT_PU200

# Output: data/sample_catalog/Phase2Spring24.json
# Format: {sample_tag: {"dataset": "/store/...", "files": ["...", ...], "n_events": int, "campaign": "..."}}
```

The catalog file becomes the single source of truth for "which datasets/files belong to which named sample". Once cached, the rest of the pipeline runs without grid access (file paths are XRootD URLs that resolve via local TLS proxy or `cmsenv` + AAA).

### 4.2 Produce layer — `scripts/ntuple_produce.py` (new)

A thin wrapper around `cmsRun runPerformanceNTuple.py` that:

1. Pulls the file list for a named sample from the catalog
2. Splits work across local CPUs or HTCondor (configurable)
3. Stages outputs to `/ceph/cms/store/user/dprimosc/l1deepmet/<tag>/<sample>/FP/<campaign>/`
4. Writes a per-sample provenance JSON next to the output:
   - `cmssw_version`, `cmssw_release_base`, `git_sha` of the FastPUPPI checkout
   - `recipe_diff`: the textual diff of our patched runPerformanceNTuple.py vs upstream
   - `n_files_in`, `n_events_in`, `n_events_out`, `wall_time_s`
   - `voms_proxy_subject`, timestamp

```bash
python scripts/ntuple_produce.py \
    --sample TT_PU200 \
    --n-events 200000 \
    --tag 26May13_150X_extended_v0 \
    --recipe NtupleProducer/python/runPerformanceNTuple.py \
    --workers 16 \
    --output-root /ceph/cms/store/user/dprimosc/l1deepmet
```

Resumable: if a child cmsRun crashes, the wrapper re-queues just the failed input. State stored in `outputs/ntuple_produce/<tag>/<sample>/state.json`.

### 4.3 Preprocess layer — `scripts/preprocess.py` (already exists, needs extension)

Generalise `params.yaml::preprocess.var_list` to include the new branches:

```yaml
preprocess:
  data_root: /ceph/cms/store/user/dprimosc/l1deepmet/26May13_150X_extended_v0
  samples:
    TT_PU200: 200000
    VBFHToInvisible_PU200: 80000
    MinBias_PU200: 50000
    SingleNeutrino_PU200: 30000
    WJetsToLNu_PU200: 100000
    DYToLL_PU200: 50000
  var_list:
    # candidate-level (was 6; now ~18)
    L1PuppiCands_pt
    L1PuppiCands_eta
    L1PuppiCands_phi
    L1PuppiCands_pdgId
    L1PuppiCands_charge
    L1PuppiCands_puppiWeight
    L1PuppiCands_z0
    L1PuppiCands_hwPt
    L1PuppiCands_hwPuppiWeight
    L1PuppiCands_hwQual
    L1PuppiCands_trackChi2RPhi
    L1PuppiCands_trackChi2RZ
    L1PuppiCands_trackNStubs
    L1PuppiCands_trackMvaQual
    L1PuppiCands_caloEta
    L1PuppiCands_caloPhi
    L1PuppiCands_clPuId
    L1PuppiCands_clEmId
    # per-event
    L1Vertex_z0
    L1Vertex_nTracks
    L1Vertex_quality
    L1PuppiMet_pt
    L1PuppiMet_phi
    L1Layer2Met_pt          # if addCTL2Met enabled
    L1Layer2Met_phi
  targets:
    genMet_pt
    genMet_phi
```

The preprocessing then pads to (n_events, 128, N_features) and writes H5 as today. Existing `H5DataLoader` works unchanged.

**FIX** along the way: the current preprocessing reads `HGCal3DCl_first1layers` and indexes it by candidate-index, which is why those values are all 0. Either (a) drop the HGCal branches from `var_list` (they're not joinable to candidates), OR (b) write a small join routine using the per-cluster `pt/eta/phi` to candidate-match. Option (b) is more useful — gives the model per-candidate HGCal shower shape — but is a separate engineering ticket.

### 4.4 Provenance lookup — `scripts/ntuple_inspect.py` (new)

```bash
python scripts/ntuple_inspect.py /ceph/cms/store/user/dprimosc/l1deepmet/26May13_150X_extended_v0/TT_PU200/

# Prints:
#   tag:            26May13_150X_extended_v0
#   campaign:       Phase2Spring24
#   cmssw:          CMSSW_15_1_0_pre4
#   recipe sha:     <git sha of patched runPerformanceNTuple.py>
#   n_events:       200000  (across 200 files)
#   branches:       (full list of 200+ branches)
#   produced:       2026-05-XX by dprim7
```

---

## 5. The patched `runPerformanceNTuple.py` lives in our repo

```
external/FastPUPPI/                          # git submodule pointing at upstream
patches/runPerformanceNTuple.patch           # our diff (sections 2.1–2.4 above)
scripts/apply_ntuple_recipe.sh               # one-liner to apply the patch into the submodule
```

This way:
- Upstream changes are tracked (`git submodule update --remote`)
- Our extension is a clean, reviewable diff
- Reproducibility is `git submodule status` + the recipe sha

---

## 6. Phased delivery

### Phase 0 — Engineering ready, blocked on grid

(This document. Done.)

### Phase 1 — Plumbing (no grid needed)

Deliverable: catalog format, recipe patch, preprocess extension, three new scripts. Tested locally on the **existing** 25Jul8 ntuples (limited to the 9 features currently there).

Concrete steps (each ~half day):

- [ ] **P1.1** Add `external/FastPUPPI/` submodule pinned to 15_1_X.
- [ ] **P1.2** Write `patches/runPerformanceNTuple.patch` implementing §2.1–2.4. Verify it applies cleanly.
- [ ] **P1.3** `scripts/ntuple_catalog.py` — schema + `list`/`show` subcommands. Mock the `refresh` step (no DAS yet) with a JSON pre-populated from the data we already have.
- [ ] **P1.4** `scripts/ntuple_produce.py` — local-worker path only (HTCondor deferred). Smoke-test by re-producing one file from the existing 25Jul8 sample's MINIAOD input, if a local MINIAOD copy is available. If not, this step is a dry-run interface test.
- [ ] **P1.5** Extend `scripts/preprocess.py` + `params.yaml` to handle the new var_list (fall back to zeros when branches are missing). Run end-to-end on the existing ntuples to confirm no regression.
- [ ] **P1.6** `scripts/ntuple_inspect.py` — provenance reader.
- [ ] **P1.7** Fix the HGCal-cluster-to-candidate join bug as a side-task.

Output of Phase 1: the pipeline runs end-to-end on the **existing data**, producing the same H5 files we have today; the new branches will be empty until Phase 2 actually generates them. This proves the pipeline works without consuming grid.

### Phase 2 — Production run (requires grid)

- [ ] **P2.1** Renew CMS grid certificate.
- [ ] **P2.2** Run `ntuple_catalog.py refresh --campaign Phase2Spring24`. Cache result.
- [ ] **P2.3** Submit ntuple-production jobs for the 6 samples in §3. Targets 510k events. Expect ~1 wall day on shared HTCondor (or ~3 days local on 16 cores).
- [ ] **P2.4** Stage outputs to `/ceph` under the new tag.
- [ ] **P2.5** Run `preprocess.py --tag 26May13_150X_extended_v0` to produce H5 files.
- [ ] **P2.6** Re-run the strongest ablations from this session (loss_form, arch_sweep, combined_best) on the new data, verify the plateau either holds (data-limited) or moves (architecture-limited) — *this is the scientific payoff*.

### Phase 3 — Iteration

- [ ] **P3.1** Make the patched recipe a one-line CLI flag (`--variant extended`) instead of a checked-in patch, so different ablations can use different recipes without manual file editing.
- [ ] **P3.2** Add HTCondor submission path to `ntuple_produce.py`.
- [ ] **P3.3** Add per-event truth-level diagnostics (gen jet multiplicity, gen MET pT spectrum) to the catalog metadata for sample-quality monitoring.

---

## 7. What's NOT in this pipeline

- **Model training**: separate concern; pipeline output feeds it but they're independent.
- **HLS / FPGA synthesis**: covered by `scripts/synthesize.py` and the hls4ml PR seed.
- **DAS sample discovery automation**: the catalog is refreshed manually with `refresh`, not auto-watched.
- **CRAB submission**: `prun.sh`-style local + HTCondor is sufficient for our event budgets. CRAB only becomes necessary at >10M events.
- **GEN-SIM/DIGI generation**: we consume MINIAOD; we don't regenerate sims from scratch.

---

## 8. References

- FastPUPPI 15_1_X NtupleProducer: <https://github.com/p2l1pfp/FastPUPPI/tree/15_1_X/NtupleProducer>
- Specifically:
  - `python/runPerformanceNTuple.py` — main driver, helpers (`monitorPerf`, `saveCands`, `addCTL2Met`, `addDecodedCalo`, `addDecodedTk`, `addHGCalTPs`, `addStaMu`, `addTkEG`)
  - `plugins/L1PFCandTableProducer.cc` — base candidate table (pt/eta/phi/mass + StringObjectFunction-driven extras)
  - `plugins/L1PFMetTableProducer.cc` — MET table (pt/phi/para/perp per algorithm)
  - `plugins/L1HGC3DclTableProducer.cc` — HGCal cluster ID scores
  - `plugins/VertexWordTableProducer.cc` — vertex word
  - `python/runInputs140X.py` — example sample list
- Our existing data: `/ceph/cms/store/user/dprimosc/l1deepmet/25Jul8_140X_v0/`
- Our preprocessing: `scripts/preprocess.py`, `src/l1deepmet/data/preprocessing.py`, `params.yaml`
- Grid-cert blocker: see `~/.claude/projects/.../memory/project_grid_cert.md`

---

## 9. Estimated effort

| Phase | Engineering wall-time | Grid time | Blocker |
|---|---:|---:|---|
| 0 (this doc) | done | — | — |
| 1 (plumbing on existing data) | **2–3 days** | none | none |
| 2 (production run on 6 samples × ~85k events) | **0.5 day** of supervising | ~24 hr HTCondor | **grid certificate** |
| 3 (polish) | 1–2 days | none | none |

Phase 1 is unblocked **today** and can be done before the grid cert returns. Phase 2 lands the actual benefit (real new variables in the training data).
