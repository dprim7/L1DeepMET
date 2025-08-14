# DVC Workflow Guide

Data version control and experiment management for L1DeepMET.
TODO: tracking currently doesn't work because data is saved into a simlinked
      location, which cant be tracked by dvc 

## Core Files

- `params.yaml` - All experiment parameters
- `dvc.yaml` - Pipeline definition  
- `dvc.lock` - Pipeline state (auto-generated)

## Basic Commands

```bash
# Run pipeline
dvc repro

# Check status
dvc status

# View pipeline
dvc dag

# Compare experiments
dvc params diff
dvc metrics diff 
```

## Parameter Management

Edit `params.yaml` for experiments:
```yaml
preprocess:
  samples:
    TT_PU200: 100000
    VBFHToInvisible_PU200: 48000
  max_pf_candidates: 128
  train_val_test_split: {train: 0.8, val: 0.1, test: 0.1}
```

## Experiment Workflow

```bash
# 1. Create experiment branch
git checkout -b experiment/new-params

# 2. Modify parameters
vim params.yaml

# 3. Run experiment
dvc repro

# 4. Compare results
dvc metrics diff main

# 5. Merge if successful
git checkout main
git merge experiment/new-params
```

## Data Versioning

```bash
# Track large files
dvc add data/preprocessed/

# Commit version
git add data/preprocessed.dvc
git commit -m "Add preprocessed data v1"

# Share 
dvc push  # Upload data
git push  # Share metadata
```

## Pipeline Definition

`dvc.yaml` example:
```yaml
stages:
  preprocess:
    cmd: python scripts/preprocess.py --config params.yaml --tag ${preprocess.output_tag}
    deps: [scripts/preprocess.py, src/l1deepmet/data/]
    params: [preprocess]
    outs: [data/preprocessed/]
    plots: [outputs/plots/]
```

## Troubleshooting

```bash
# Force re-run
dvc repro --force

# Check dependencies
dvc deps preprocess

# Clean cache
dvc gc --workspace
```