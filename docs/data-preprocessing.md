# Data Preprocessing Pipeline

Transforms ROOT files into HDF5 datasets for ML training.

## Pipeline Stages

```
ROOT Files → load_samples → select_events → preprocess_data → split → save_h5_files
```

### 1. Load Data (`load_samples_to_numpy`)
- Loads ROOT files with `uproot`
- Converts to numpy arrays
- Handles PUPPI candidates and MC truth

### 2. Select Events (`select_events`)
- Random sampling per sample
- Configurable event counts

### 3. Preprocess (`preprocess_data`)
- Outlier removal (pt/px/py > 500 GeV → 0)
- Data sanitization (missing values → 0)
- Feature concatenation into single array

**Output Features (9 per candidate)**: (subject to updates)
```
0: pt, 1: eta, 2: phi, 3: puppi, 4: hcal_depth
5: px, 6: py, 7: encoded_pdgId, 8: encoded_charge
```

### 4. Split Data (`combine_shuffle_split`)
- Combines samples
- Shuffles with seed=42
- 80/10/10 train/val/test split

### 5. Save HDF5 (`save_h5_files`)
- Compressed HDF5 files: `train.h5`, `val.h5`, `test.h5`
- Embedded metadata

## Usage

### With DVC (recommended)
```bash
dvc repro preprocess
```

### Direct
```bash
python scripts/preprocess.py --config params.yaml --tag 25Jul8_140X_v0
```

### Load Results
```python
import h5py
with h5py.File('data/preprocessed/train.h5', 'r') as f:
    X_train = f['features'][:]  # (n_events, max_pf, 9)
    Y_train = f['targets'][:]   # (n_events, 2)
```

### Split for Training
```python
from l1deepmet.data.split_features import split_preprocessed_features
inputs, pxpy, pdgId, charge = split_preprocessed_features(X_train)
```

## Configuration

Key parameters in `params.yaml`:
```yaml
preprocess:
  samples:
    TT_PU200: 100000
    VBFHToInvisible_PU200: 48000
  max_pf_candidates: 128
  train_val_test_split: {train: 0.8, val: 0.1, test: 0.1}
```
