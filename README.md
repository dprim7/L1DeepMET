# L1DeepMET

Hardware-aware deep learning framework for Level-1 Missing Transverse Energy reconstruction on FPGAs for CMS and the next generation HL-LHC.

This project is an extensive refactor and extension of https://github.com/ucsd-hep-ex/L1METML 

## Quick Start

```bash
# Setup
micromamba env create -f environment.yaml
micromamba activate l1deepmet
pip install -e .

# Run preprocessing
dvc repro preprocess
```

## Pipeline

1. **Preprocess** - Load ROOT files, clean data, create train/val/test splits
2. **Train** - Neural network training (coming soon)
3. **Evaluate** - Performance assessment (coming soon)

## Documentation

- [Data Preprocessing](docs/data-preprocessing.md) - Pipeline details
- [DVC Workflow](docs/dvc-workflow.md) - Experiment management