# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development workflow (STANDING ORDER)

**Test-first.** All new code must be developed test-first:

1. **Write a unit test** describing the function/class's expected behavior, in
   `tests/unit/<area>/test_<name>.py`. Run it; it should fail (red).
2. **Write the implementation** in `src/l1deepmet/<area>/<name>.py`. Run the
   test again; it should pass (green).
3. **Commit** in small increments — one test+implementation per commit when
   possible.

Ad-hoc smoke tests via `python -c '...'` are useful for debugging but **do not
count** as the required tests. The test must live in `tests/` and run under
`pytest`.

For new infrastructure (e.g. ``src/l1deepmet/synthesis/``,
``src/l1deepmet/quantization/``) a `tests/unit/<module>/` directory should be
created alongside, with at minimum: one test per public function, one test for
each architectural variant the module supports, and one integration test that
exercises the full module end-to-end against a tiny synthetic input.

If you find yourself writing code without writing a test first, **stop and
write the test**, then continue.

## Long-running jobs (STANDING ORDER)

Anything that takes more than ~10 min wall — ntuple production loops, hyper-
parameter sweeps, training runs, multi-sample preprocessing — **MUST be
launched in a detached `screen` session**. The Claude Code harness's
background-task tracking dies on SSH disconnect or harness restart; `screen`
double-forks to PPID=1 (init/systemd) and survives anything short of host
reboot.

Pattern:

```bash
# 1. Write the job to a script (so screen has something concrete to launch
#    and the user can re-run it after a host reboot from the same artifact).
cat > /tmp/run_<jobname>.sh <<'EOF'
#!/usr/bin/env bash
set -uo pipefail                       # not -e — let later samples run past a failure
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc12      # only for jobs that need cmsRun
cd /home/users/dprimosc/CMSSW_14_2_0_pre2_L1DeepMET && eval $(scramv1 runtime -sh)
cd /home/users/dprimosc/L1DeepMET/.claude/worktrees/<worktree>
LOG=/tmp/<jobname>.log
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] BEGIN" >> $LOG
<the actual command> >> $LOG 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] END (exit=$?)" >> $LOG
EOF
chmod +x /tmp/run_<jobname>.sh

# 2. Launch detached.
screen -dmS <jobname> bash /tmp/run_<jobname>.sh

# 3. Verify it's actually running.
screen -ls
pgrep -af "<a process the job spawns>"
```

For the user (or a future Claude session) to check or reattach:

```bash
screen -ls                       # list sessions
screen -r <jobname>              # attach interactively (Ctrl-A D to detach)
tail -f /tmp/<jobname>.log       # passive tail
```

Hard rules while attached: Ctrl-A then D to detach safely. **Never** Ctrl-C,
`exit`, Ctrl-D, or Ctrl-\\ inside the screen — those kill the production.

Resumability: cmsRun production via `scripts/ntuple_produce.py` writes a
`state.json` per sample; re-running the same wrapper command after a crash
only re-runs failed/missing jobs.

UAF gotcha: TF processes spawn many internal threads at startup; running ≥2
TF training processes in parallel hits per-process pthread limits with
`EAGAIN`. For training sweeps, use `--max-parallel=1` (sequential) in
`scripts/run_sweep.py`-style drivers. For cmsRun production, 16 parallel
workers per sample is fine (each spawns a fresh CMSSW process tree).

## Reports (STANDING ORDER)

Anything cited as evidence in a `reports/<study>/report.md` — every plot,
every table, every number quoted inline — **must be regeneratable from a
script in that same `reports/<study>/scripts/` directory**, given only the
inputs that are committed (raw JSONs, aggregated CSVs) or referenced by
path (H5 files, ntuple dirs on /ceph).

Layout convention for a study:

```
reports/<study>/
  PLAN.md          # design pre-registered before measurement
  report.md        # results + interpretation; ends with a "Reproducing" section
  scripts/         # train_one.py, run_sweep.py, analyze.py — checked in
  results/raw/     # per-cell JSON + log from the sweep — checked in
  results/         # aggregated CSVs, fit JSON — derived; OK to regenerate
  plots/           # PNG/SVG — derived; OK to regenerate
```

Rules:

- **Plots**: every PNG/SVG under `plots/` must be produced by an explicit
  function call in `scripts/analyze.py` (or equivalent). One-off
  interactive notebook output that isn't checked in does NOT count as
  evidence — it can't be re-run after a refactor or with new data.
- **Tables / inline numbers**: must be computable from a CSV under
  `results/` or directly from raw JSONs there. Don't hand-copy numbers
  from a terminal session.
- **Reproducing section** at the end of `report.md` must give the exact
  commands that take inputs → plots+tables, in order.
- **Test it**: after writing the report, delete `plots/*` and re-run
  the analysis script; the plots must come back byte-similar. (We do this
  before committing.)

Goal: any future Claude session, after a refactor or after producing new
ntuples, can run a single command and get fresh plots + tables. No
human reconstruction of "what was that one Jupyter cell that made figure 3".

## Dataset card (STANDING ORDER)

Every preprocessed dataset under `outputs/preprocessed/<tag>/` MUST have
a `dataset_card.json` + `dataset_card.md` next to its H5 files. They're
produced automatically by `scripts/preprocess.py` (final step before
control plots); if you're touching the preprocessor and they don't get
emitted, that's a regression to fix, not an OK-as-long-as-the-H5-exists.

What's in the card (schema v1, see `src/l1deepmet/data/dataset_card.py`):

- **Provenance**: git SHA, preprocess args, source ntuple dir, split seed.
- **Composition**: total event count; per-split counts + fractions;
  per-sample loaded vs requested (so capped samples are visible).
- **Schema**: feature layout, event feature layout, all tensor shapes.
- **Per-candidate feature stats** on the train split: median, MAD, min,
  max, %NaN, %Inf — separating real candidates (`pt > 0`) from pad
  zeros so sparse columns aren't pulled to zero.
- **Per-event feature stats**: mean, std, range, %zero.
- **Target stats**: gen MET distribution, per-pT-bin event counts.
- **Top-level warnings**: aggregated from all per-feature and per-sample
  warnings (NaN/Inf, all-zero on real candidates, extreme magnitudes,
  sample-load shortfalls). These are what a reviewer reads first.

What the card is FOR:

- **Schema-validate training scripts**: load the card, assert the
  features layout matches what the model expects. Fail-fast on
  swapped-tag mistakes.
- **Cite numbers in reports**: don't hand-copy "we trained on 87 312
  events" from a terminal — pull it from `dataset_card.json`. Numbers
  in `reports/<study>/report.md` should trace back to a JSON file, not
  a re-derived computation that could drift.
- **Surface upstream bugs**: the card flags "all-zero on real
  candidates" and "extreme magnitude (1e38)" automatically. Two real
  preprocessor bugs (`dxyErr` not loaded; `clPuId`/`clEmId` contain
  upstream float32-limit garbage) were caught by the first card on
  the 20k tag — both pre-existed silently for months.

What the card is NOT:

- A replacement for the H5 attrs (`feature_layout`, `samples_used`,
  `sample_event_counts` etc.) — those travel inside the H5 and stay.
- A model card (different artifact, lives alongside model checkpoints).
- A "datasheet for datasets" in Gebru et al. format — that's a more
  formal, prose-heavy document. The card is the machine-readable
  bottom-up version of it.

## Evaluation metrics for architecture / recipe sweeps (STANDING ORDER)

MAE / MSE / IQR on (px, py) are useful proxies during development — but they
do not represent how the trained model is actually used by the L1 trigger.
**Any architecture sweep, loss ablation, feature ablation, or
"shall-we-ship-this" comparison must report the full L1 physics card.**

The card has three tiers: **required** for *any* reported comparison;
**strongly recommended where relevant** (almost always at least one
applies); **required for ship decisions only**.

### Tier 1 — required for every sweep

1. **Resolution × pT-bin** — IQR/2 of (pred − true) per gen-MET bin
   `[0,50), [50,100), [100,200), [200,300), [300,400), [400,∞)`. Reveals
   tail behaviour that a global IQR hides.
2. **Response × pT-bin** — ratio-of-means convention (`mean(pred)/mean(true)`
   per bin; see `reports/METRIC_FIX_addendum.md`). Catches Wiener-filter
   regression-to-mean (the failure that drove `loss_diagnosis_apr2026`).
3. **φ resolution** — IQR/2 of `phi(pred) − phi(true)` (with proper
   `±π` wrap). Mode-1 models sometimes get magnitude right and direction
   random (seed-dependent failure mode in `dense_architecture_baseline_apr2026`);
   φ catches it. Bimodal-across-seeds is itself a result.
4. **ROC + AUC** — `gen_MET > 200` vs `gen_MET < 50`. Report the full
   curve, not just the scalar — AUC of 0.97 vs 0.98 can hide that one
   model is better at the actual working point and worse elsewhere.
5. **Turn-on curves at canonical L1 thresholds** — `ε(gen_MET; L1_thr)`
   at L1 thresholds `{100, 150, 200} GeV`. *Steepness near threshold* is
   what matters for purity-vs-efficiency, not the asymptote.
6. **Rate vs threshold on MinBias_PU200** — events / s passing a given L1
   MET threshold. The L1 MET slice has a fixed kHz budget; resolution
   gains that *raise* the rate at the working point are not real wins.
7. **Working-point efficiency** — this is the **single number that
   actually decides whether the trigger is good**. From the rate curve,
   find the threshold T that gives the L1 budget (assume ~4 kHz for the
   MET slice unless the user gives a number); at that T, report the
   signal efficiency per sample. Without this, the rate curve and the
   turn-on curve are individually meaningless.
8. **PUPPI baseline on every line** — every row of the card reports
   model AND PUPPI on the same events. Without that, "ML wins" is
   meaningless — PUPPI is what ships if we don't.

These all come out of one function:

```python
from l1deepmet.metrics.physics import (
    compute_resolution_metrics,   # resolution + response per bin (1, 2)
    compute_trigger_metrics,       # scalar AUC (4)
    compute_roc_curve,             # ROC arrays for plotting (4)
    compute_turn_on,               # efficiency vs gen_MET at fixed reco thr (5)
    compute_puppi_baseline,        # PUPPI MET from raw H5 features (8)
    full_physics_card,             # one call → everything above, JSON-safe
)
```

Plotting: `src/l1deepmet/plotting.py` has `plot_roc_curve`,
`plot_turn_on_curves`, `plot_trigger_rates`, `plot_combined_rates`.

Rate eval needs MinBias_PU200 (in the current `26May22_142_extended_20k_v0`
production); signal samples (VBF, TT, SMS) give efficiency. Working-point
efficiency joins them.

### Tier 2 — strongly recommended (almost always one applies)

9. **Per-PU dependence** — resolution / response / working-point efficiency
   binned by `nL1Vtx` (we save this in the extended H5). HL-LHC oscillates
   between PU 140 and PU 220; a model that wins at average PU and breaks
   at the tails is an operational regression. The vertex producer is in
   the patched recipe so the data is there.
10. **Asymmetric tail behaviour** — fakes (reco ≫ true; eats rate) and
    misses (reco ≪ true; kills efficiency) are NOT the same problem.
    IQR/2 is symmetric and hides this. Report bias and high-side /
    low-side tail counts separately (e.g., `P(reco > true + 50 GeV)`
    and `P(reco < true − 50 GeV)`).
11. **Per-eta region** — HGCal endcap vs barrel calo behave differently.
    A weak spot in one region hides in a global metric. Compute Tier-1
    quantities for `|eta| < 1.5` and `|eta| > 1.5` separately when an
    architecture changes how it uses HGCal features (e.g., the cl* /
    HGCal-ID extended features).
12. **PUPPI-ablation** — at eval time, set `puppi_weight = 1.0` for all
    candidates. Re-run the card. If the model doesn't degrade, it isn't
    using PUPPI — it's re-deriving (worse) PU rejection from raw inputs.
    Cheap test, catches a specific failure mode.

### Tier 3 — required only for ship decisions

13. **Quantization gap** — float vs QKeras (per-tensor int) vs HGQ
    (per-bit gradient). Resolution numbers in float don't ship —
    quantized numbers do. Use `src/l1deepmet/quantization/` (currently
    stub) or QAT recipes in the loss/output_head ablation reports.
14. **Resource + latency** — HLS4ML estimate (LUTs, FFs, DSPs, BRAM) on
    the target device (VU13P, clock 320 MHz) and latency in ns. State
    the synthesis tool + version; estimates from `hls4ml` are not the
    same as place-and-routed numbers.

### The decision quantity

The thing that decides "ship this" is a Pareto on
**(working-point signal efficiency, rate on MinBias, FPGA resources)**.
A model that wins on one axis and loses on another is *not*
unambiguously better — say so explicitly when reporting it. (Same
hardware-aware principle as in the experimenter skill: state the
constraint with every number.)

### Development-iteration shortcut

Inside the inner loop of an experiment (e.g., the event-count
justification sweep), MAE is a fine fast proxy — it correlates with the
Tier-1 quantities and is much cheaper to compute per epoch. But anything
cited as evidence in a `reports/<study>/report.md` must include the
Tier-1 card. Architecture / loss / feature ablations: Tier 1 + at least
one Tier-2 metric chosen for relevance (e.g., per-PU for a feature that
should help PU rejection; PUPPI-ablation for any model that touches
`puppi_weight`).

## Project Overview

**L1DeepMET** reconstructs Level-1 Missing Transverse Energy (MET) for the CMS detector at the HL-LHC using hardware-aware deep learning. Models are ultimately deployed on FPGAs via HLS4ML. The framework processes PUPPI particle candidates (up to 128 per event) and predicts MET as (px, py).

## Setup

```bash
micromamba env create -f environment.yaml
micromamba activate l1deepmet
pip install -e .         # or pip install -e ".[dev]" for dev deps (pytest)
```

Data lives at `/ceph/cms/store/user/dprimosc/l1deepmet_data/` (symlinked to `data/`). Note: DVC cannot track data inside symlinked directories.

## Commands

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/data/test_preprocessing.py -v
pytest tests/unit/losses/test_losses.py -v
pytest tests/integration/test_preprocessing_dataflow.py -v

# Preprocess data (DVC)
dvc repro preprocess

# Preprocess directly
python scripts/preprocess.py --config params.yaml --tag <tag> --data-root <path> --output-root data/preprocessed

# Train (normfac=100 is required to normalize gen MET targets and avoid NaN)
python scripts/train.py --data-dir preprocessed/25Jul8_140X_v0 --output-dir outputs/models/run_0 \
  --epochs 100 --batch-size 256 --lr 1e-3 --normfac 100 --mode 1
```

## Architecture

### Data Flow

```
ROOT files → uproot → padded numpy arrays (N_events, 128, 9) → HDF5
→ H5DataLoader → split features → TF Dataset → model
```

**9 features per particle candidate:**
- `[0-4]`: continuous (pt, eta, phi, puppi weight, HGCal depth)
- `[5-6]`: momentum components (px, py)
- `[7]`: encoded pdgId (0=invalid, 1=charged hadron, 2=neutral hadron, 3=photon, 4=muon, 5=electron)
- `[8]`: encoded charge (0=invalid, 1=negative, 2=neutral, 3=positive)

**Targets:** `(N_events, 2)` — gen-level MET as (px, py).

### Model Inputs

All models expect a dict:
```python
{
    'continuous_inputs': (B, 128, 5),
    'momentum_inputs':   (B, 128, 2),
    'pdgid_inputs':      (B, 128),
    'charge_inputs':     (B, 128),
}
```

This split is done by `split_preprocessed_features()` in `train.py`.

### Model Types (`src/l1deepmet/models/`)

- **Dense** (`dense.py`): Implemented. Three training modes:
  - Mode 0: Direct MET regression
  - Mode 1: Per-particle weighting (weight × individual px/py, summed)
  - Mode 2: Event-level single weight × sum of px/py
- **Embedding** (`embedding.py`): Partial — categorical embedding layer exists
- **MLP-Mixer** (`mixer.py`): TODO — not implemented
- Quantized variants (`DenseQuantized`, `DenseHGQ`, `MixerQuantized`, `MixerHGQ`) use QKeras/HGQ2 for FPGA deployment

### Losses (`src/l1deepmet/losses/corrected.py`)

`CorrectedCompositeLoss` = MAE + MSE + physics-informed `BinnedDeviation` (default weights: 1.0, 1.0, 200.0). The binned term measures asymmetry of errors in pt-bins [50, 100, 200, 300, 400, ∞] GeV.

### Configuration

`params.yaml` is the single source of truth for all pipeline parameters (preprocessing options, model type, training hyperparameters). `src/l1deepmet/config.py` provides dot-notation access and CLI-merge support.

## What Is and Isn't Implemented

**Working:**
- Preprocessing pipeline (`scripts/preprocess.py`, `src/l1deepmet/data/preprocessing.py`)
- `H5DataLoader` with TF dataset creation (pass `normfac=100.0` to normalize targets)
- Dense model (mode 0/1/2), training via `scripts/train.py`
- `CorrectedCompositeLoss` (pass `normfac=100.0` to scale pt bins) and `BinnedDeviation` metric
- Unit + integration tests for preprocessing and losses

**Incomplete / TODO:**
- `src/l1deepmet/models/factory.py` — model factory not complete
- `src/l1deepmet/models/mixer.py` — MLP-Mixer body not implemented
- `src/l1deepmet/models/output_head.py` — `ParticleWeightHead`, `METWeightHead` are TODO
- `src/l1deepmet/inference.py` — new file, not yet implemented
- Training script (`scripts/train.py`) does not exist; `train.py` has framework helpers but no runnable entry point
- Quantization (`DenseQuantized`, `DenseHGQ`) — class stubs only
- DVC train/evaluate stages commented out in `dvc.yaml`
