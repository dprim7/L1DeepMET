# Training and Evaluation Guide

This guide explains how to train and evaluate L1DeepMET models.

## Prerequisites

Make sure you have:
1. Preprocessed data in H5 format (run preprocessing first)
2. Python environment with all dependencies installed

## Training a Model

### Using DVC (Recommended)

The simplest way to train a model is using DVC, which handles all dependencies automatically:

```bash
# Train with default parameters from params.yaml
dvc repro train

# This will:
# - Load preprocessed train.h5 and val.h5
# - Build the model specified in params.yaml
# - Train with configured parameters
# - Save best model to models/<tag>/best_model.keras
# - Save training history to models/<tag>/training_history.json
```

### Direct Script Execution

You can also run the training script directly:

```bash
python scripts/train.py --config params.yaml --tag 25Jul8_140X_v0
```

Optional arguments:
- `--data-root`: Root directory containing preprocessed data (default: `data/preprocessed`)
- `--output-dir`: Directory for model outputs (default: `models`)
- `--epochs`: Override number of training epochs
- `--batch-size`: Override batch size

### Configuration

Training parameters are defined in `params.yaml`:

```yaml
train:
  batch_size: 1024  # Batch size for training
  epochs: 100       # Number of training epochs
  mode: 1           # Training mode: 0=direct, 1=per-particle weight, 2=global weight

model:
  type: dense       # Model architecture
  units: [64, 32, 16]  # Hidden layer sizes
  activation: tanh  # Activation function
  with_bias: false  # Add bias to predictions

loss:
  mae_weight: 1.0        # Weight for MAE loss
  mse_weight: 1.0        # Weight for MSE loss
  binned_weight: 200.0   # Weight for physics-informed binned loss

optimizer:
  type: adam
  learning_rate: 0.001

callbacks:
  early_stopping:
    monitor: val_loss
    patience: 40
  reduce_lr:
    monitor: val_loss
    factor: 0.5
    patience: 10
```

### Training Outputs

Training produces the following outputs in `models/<tag>/`:

- `best_model.keras`: Best model based on validation loss
- `final_model.keras`: Model at the end of training
- `training_history.json`: Loss and metrics history
- `loss_history.csv`: CSV log of training progress
- `tensorboard_logs/`: TensorBoard logs for visualization

View training progress with TensorBoard:

```bash
tensorboard --logdir models/<tag>/tensorboard_logs/
```

## Evaluating a Model

### Using DVC

```bash
# Evaluate the trained model on test data
dvc repro evaluate
```

### Direct Script Execution

```bash
python scripts/evaluate.py \
    --config params.yaml \
    --tag 25Jul8_140X_v0 \
    --model-dir models/25Jul8_140X_v0
```

Optional arguments:
- `--data-root`: Root directory containing preprocessed data
- `--output-dir`: Directory for evaluation outputs (default: `outputs/evaluation`)
- `--batch-size`: Batch size for inference (default: 1024)

### Evaluation Outputs

Evaluation produces the following outputs in `outputs/evaluation/<tag>/`:

- `evaluation_metrics.json`: Comprehensive evaluation metrics
- `predictions.npz`: True and predicted MET values

### Evaluation Metrics

The evaluation script calculates:

- **Component metrics**: MAE and MSE for px and py
- **MET magnitude metrics**: MAE, MSE, RMSE for MET magnitude
- **Angular metrics**: MAE for phi (angle)
- **Relative metrics**: Mean relative error
- **Resolution**: Standard deviation of residuals

Example metrics output:

```json
{
  "mae_px": 5.234,
  "mae_py": 5.187,
  "mse_px": 45.123,
  "mse_py": 44.876,
  "mae_met": 6.789,
  "rmse_met": 8.234,
  "mae_phi": 0.145,
  "mean_relative_error": 0.089,
  "resolution": 7.456
}
```

## Complete Pipeline

To run the entire pipeline from preprocessing to evaluation:

```bash
# Option 1: Run all stages
dvc repro evaluate

# Option 2: Run stages individually
dvc repro preprocess
dvc repro train
dvc repro evaluate
```

## Training Different Models

To train models with different architectures or hyperparameters:

1. Create a new experiment configuration in `params.yaml`
2. Update the `output_tag` to distinguish experiments
3. Run training:

```bash
dvc repro train --force
```

## Tips

- **Monitor training**: Use TensorBoard to track training progress
- **Early stopping**: Training will stop early if validation loss doesn't improve
- **Learning rate**: Will be reduced automatically when validation loss plateaus
- **Checkpointing**: Best model is saved automatically based on validation loss
- **Memory**: Adjust batch size if you encounter OOM errors

## Troubleshooting

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size in `params.yaml`

**Issue**: Model not converging
- **Solutions**: 
  - Try different learning rate
  - Adjust loss weights
  - Change model architecture
  - Check for data quality issues

**Issue**: Training too slow
- **Solutions**:
  - Increase batch size (if GPU memory allows)
  - Reduce model size
  - Use fewer epochs with early stopping

**Issue**: Validation loss increasing
- **Solution**: Model may be overfitting; reduce model complexity or add regularization
