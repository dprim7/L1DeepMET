# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
