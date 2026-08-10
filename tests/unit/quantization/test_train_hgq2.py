"""Unit tests for ``scripts/train_hgq2.py``.

The training driver is a thin wrapper that:
  1. builds the HGQ2 model via ``build_hgq2_model``,
  2. assembles the corrected composite loss,
  3. compiles + fits on the H5 dataset,
  4. saves the trained model + per-epoch history + a result.json.

We test the helpers in isolation; full end-to-end training on the real
dataset lives under ``tests/integration/`` and runs a single tiny epoch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

pytest.importorskip("hgq")

# scripts/ isn't on the package path by default.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from train_hgq2 import build_callbacks, build_loss, build_optimizer  # noqa: E402


class TestBuildLoss:
    def test_default_loss_is_mae_only_no_xy_balance(self):
        """The default for the QAT comparison matches the loss-ablation
        winner: MAE only, no xy_balance, no binned term."""
        loss = build_loss()
        assert loss.mae_weight == 1.0
        assert loss.mse_weight == 0.0
        assert loss.binned_weight == 0.0
        assert loss.xy_balance_weight == 0.0

    def test_can_override_to_mae_mse_xy(self):
        """Allow the legacy 'broken' recipe for the side-by-side replication."""
        loss = build_loss(mae_weight=1.0, mse_weight=1.0, xy_balance_weight=10.0)
        assert loss.mae_weight == 1.0
        assert loss.mse_weight == 1.0
        assert loss.xy_balance_weight == 10.0

    def test_normfac_passed_through(self):
        loss = build_loss(normfac=100.0)
        assert loss.normfac == 100.0


class TestBuildOptimizer:
    def test_returns_adamw(self):
        import tensorflow as tf
        opt = build_optimizer(learning_rate=1e-3, clipnorm=1.0)
        assert isinstance(opt, tf.keras.optimizers.AdamW)

    def test_learning_rate_set(self):
        opt = build_optimizer(learning_rate=5e-4)
        # AdamW stores learning rate as a tracked variable.
        assert float(opt.learning_rate) == pytest.approx(5e-4)


class TestBuildCallbacks:
    def test_free_ebops_ordered_before_csvlogger(self, tmp_path):
        """Both hook on_epoch_end and Keras calls callbacks in list
        order — FreeEBOPs must write logs['ebops'] before CSVLogger
        persists the row, or the column silently never appears."""
        import tensorflow as tf
        from hgq.utils.sugar import FreeEBOPs

        callbacks = build_callbacks(tmp_path, patience=10)
        ebops_idx = [i for i, c in enumerate(callbacks)
                     if isinstance(c, FreeEBOPs)]
        csv_idx = [i for i, c in enumerate(callbacks)
                   if isinstance(c, tf.keras.callbacks.CSVLogger)]
        assert len(ebops_idx) == 1
        assert len(csv_idx) == 1
        assert ebops_idx[0] < csv_idx[0]

    def test_ebops_log_false_removes_callback(self, tmp_path):
        from hgq.utils.sugar import FreeEBOPs

        callbacks = build_callbacks(tmp_path, patience=10, ebops_log=False)
        assert not any(isinstance(c, FreeEBOPs) for c in callbacks)

    def test_standard_callbacks_present(self, tmp_path):
        import tensorflow as tf

        callbacks = build_callbacks(tmp_path, patience=7)
        by_type = {type(c).__name__: c for c in callbacks}
        assert "EarlyStopping" in by_type
        assert by_type["EarlyStopping"].patience == 7
        assert "ReduceLROnPlateau" in by_type
        assert "TerminateOnNaN" in by_type
        csv = by_type["CSVLogger"]
        assert Path(csv.filename).parent == tmp_path
        assert isinstance(csv, tf.keras.callbacks.CSVLogger)
