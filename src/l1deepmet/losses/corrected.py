import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from l1deepmet.metrics.binned import compute_binned_deviation

class CorrectedCompositeLoss(tf.keras.losses.Loss):
    """ Composite loss that combines MSE, MAE, as
    well as a physics-informed binned loss.
    Each loss is weighted via config.
    """
    def __init__(self,
                 mae_weight: float = 1.0,
                 mse_weight: float = 1.0,
                 binned_weight: float = 200.0,
                 normfac: float = 1.0,
                 name: str = 'corrected_composite_loss',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.binned_weight = binned_weight
        self.normfac = normfac

        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()


    def call(self, y_true, y_pred):
        mae_loss = tf.reduce_mean(self.mae(y_true, y_pred))
        mse_loss = tf.reduce_mean(self.mse(y_true, y_pred))
        pt_bins = np.array([50., 100., 200., 300., 400., np.inf]) / self.normfac
        binned_loss = compute_binned_deviation(y_true, y_pred, pt_bins=pt_bins)

        total_loss = (self.mae_weight * mae_loss +
                        self.mse_weight * mse_loss +
                        self.binned_weight * binned_loss)

        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "mae_weight": self.mae_weight,
            "mse_weight": self.mse_weight,
            "binned_weight": self.binned_weight,
            "normfac": self.normfac,
        })
        return config

    