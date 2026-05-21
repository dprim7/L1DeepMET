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
