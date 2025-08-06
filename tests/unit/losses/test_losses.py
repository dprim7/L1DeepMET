import pytest # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from unittest.mock import Mock, patch

from l1deepmet.losses.corrected import CorrectedCompositeLoss

class TestCorrectedCompositeLoss:

    @pytest.fixture #TODO: parametrize different weights?
    def loss_fn(self): 
        return CorrectedCompositeLoss(
            mae_weight = 1.0,
            mse_weight = 1.0,
            binned_weight = 200.0,    
        )
    
    @pytest.fixture #TODO: consider moving to base class
    def sample_data(self):

        np.random.seed(42)
        batch_size = 32

        # generating random true values
        pt_true = np.random.uniform(50, 500, batch_size)
        phi_true = np.random.uniform(-np.pi, np.pi, batch_size)

        px_true = pt_true * np.cos(phi_true)
        py_true = pt_true * np.sin(phi_true)

        y_true = np.stack([px_true, py_true], axis=-1)

        # generatinf random predicted values + noise
        noise_scale = 0.1
        px_pred = px_true + np.random.normal(0, noise_scale, batch_size)
        py_pred = py_true + np.random.normal(0, noise_scale, batch_size)
        y_pred = np.stack([px_pred, py_pred], axis=-1)

        return y_true, y_pred
    
    def test_initialization(self):
        
        loss = CorrectedCompositeLoss(
            mae_weight=0.5,
            mse_weight=0.5,
            binned_weight=100.0
        )
        assert loss.mae_weight == 0.5
        assert loss.mse_weight == 0.5
        assert loss.binned_weight == 100.0

    def test_call(self, loss_fn, sample_data):
        y_true, y_pred = sample_data

        loss_value = loss_fn(y_true, y_pred)

        assert loss_value.shape == ()
        assert not tf.math.is_nan(loss_value)
        assert not tf.math.is_inf(loss_value)

    def test_zero_inputs(self, loss_fn):
        y_true = np.zeros((32, 2), dtype=np.float32)
        y_pred = np.zeros((32, 2), dtype=np.float32)

        loss_value = loss_fn(y_true, y_pred)

        assert not tf.math.is_nan(loss_value)
        assert loss_value >= 0

    def test_weight_effect(self, sample_data):
        y_true, y_pred = sample_data

        loss_equal = CorrectedCompositeLoss(
            mae_weight=1.0,
            mse_weight=1.0,
            binned_weight=1.0
        )

        loss_mse_only = CorrectedCompositeLoss(
            mae_weight=0.0,
            mse_weight=1.0,
            binned_weight=0.0
        )

        loss_mae_only = CorrectedCompositeLoss(
            mae_weight=1.0,
            mse_weight=0.0,
            binned_weight=0.0
        )

        loss_fn_high_weight = CorrectedCompositeLoss(
            mae_weight=10.0,
            mse_weight=10.0,
            binned_weight=1000.0
        )
        loss_fn_low_weight = CorrectedCompositeLoss(
            mae_weight=0.1,
            mse_weight=0.1,
            binned_weight=10.0
        )
        equal_loss_value = loss_equal(y_true, y_pred)
        mse_loss_value = loss_mse_only(y_true, y_pred)
        mae_loss_value = loss_mae_only(y_true, y_pred)
        high_loss_value = loss_fn_high_weight(y_true, y_pred)
        low_loss_value = loss_fn_low_weight(y_true, y_pred)

        assert high_loss_value > low_loss_value
        assert equal_loss_value > mse_loss_value
        assert equal_loss_value > mae_loss_value
        assert mse_loss_value < mae_loss_value

    def test_binned_deviation(self, loss_fn, sample_data):
        np.random.seed(42)

        pt_bin1 = np.random.uniform(50, 100, 20)
        phi_bin1 = np.random.uniform(-np.pi, np.pi, 20)
        px_true_bin1 = pt_bin1 * np.cos(phi_bin1)
        py_true_bin1 = pt_bin1 * np.sin(phi_bin1)

        #add positive bias to predictions
        px_pred_bin1 = px_true_bin1 * 1.1
        py_pred_bin1 = py_true_bin1 * 1.1

        y_true = tf.constant(np.stack([px_true_bin1, py_true_bin1], axis=-1), dtype=tf.float32)
        y_pred = tf.constant(np.stack([px_pred_bin1, py_pred_bin1], axis=-1), dtype=tf.float32)

        loss_binned_only = CorrectedCompositeLoss(
            mae_weight=0.0,
            mse_weight=0.0,
            binned_weight=1.0
        )

        loss_value = loss_binned_only(y_true, y_pred)

        assert loss_value > 0







