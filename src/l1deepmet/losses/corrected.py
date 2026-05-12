import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from l1deepmet.metrics.binned import compute_binned_deviation

def compute_phi_loss(y_true, y_pred):
    """Compute angular loss: pt-weighted (1 - cos(dphi)).

    Uses the vector dot-product formulation to avoid atan2 numerical issues:
    cos(dphi) = (px_t*px_p + py_t*py_p) / (|p_t| * |p_p|)

    Args:
        y_true: (B, 2) true (px, py) in normalized units
        y_pred: (B, 2) predicted (px, py) in normalized units

    Returns:
        Scalar loss.
    """
    px_true, py_true = y_true[:, 0], y_true[:, 1]
    px_pred, py_pred = y_pred[:, 0], y_pred[:, 1]

    # Use dot product for stable cos(dphi) computation
    dot = px_true * px_pred + py_true * py_pred
    pt_true = tf.sqrt(tf.square(px_true) + tf.square(py_true) + 1e-8)
    pt_pred = tf.sqrt(tf.square(px_pred) + tf.square(py_pred) + 1e-8)

    cos_dphi = tf.clip_by_value(dot / (pt_true * pt_pred + 1e-8), -1.0, 1.0)
    cos_loss = 1.0 - cos_dphi  # in [0, 2]

    # Weight by true pT (higher pT events matter more for MET)
    weight = tf.where(pt_true > 0.01, pt_true, tf.zeros_like(pt_true))
    weighted_loss = cos_loss * weight

    total_weight = tf.reduce_sum(weight) + 1e-8
    return tf.reduce_sum(weighted_loss) / total_weight


def compute_xy_balance_loss(y_true, y_pred):
    """Penalize asymmetry between X and Y axis errors.

    Computes max(MAE_x, MAE_y) - min(MAE_x, MAE_y), encouraging the model
    to achieve equal resolution on both axes rather than sacrificing one.

    Returns:
        Scalar loss (0 = perfectly symmetric).
    """
    mae_x = tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))
    mae_y = tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))
    return tf.abs(mae_x - mae_y)


class CorrectedCompositeLoss(tf.keras.losses.Loss):
    """Composite MET loss: MAE + MSE + optional phi/xy-balance terms.

    NOTE on `binned_weight`:
        BinnedDeviation enforces unbiased mean response per pT bin. With
        non-negligible noise (the typical L1 MET regime), the resolution-
        optimal predictor has biased response (regression toward the mean —
        Wiener-filter behavior), so this term is in direct conflict with
        resolution. Empirically (see `reports/loss_diagnosis_apr2026/`),
        `binned_weight=200` degrades a scalar-weight model from PUPPI MET
        (38.30 GeV X IQR/2) down to 42.78 GeV, while `binned_weight=0` lets
        the same model improve to 34.86 GeV — beating PUPPI by ~3.5 GeV via
        a learned per-event response calibration. Default is therefore 0.0.
        Response calibration for downstream physics, if needed, should be
        applied as a post-training per-bin scale factor at inference rather
        than as a training loss term. Keep BinnedDeviation as a *metric* for
        monitoring (it is one).
    """
    def __init__(self,
                 mae_weight: float = 1.0,
                 mse_weight: float = 1.0,
                 huber_weight: float = 0.0,
                 huber_delta: float = 0.5,
                 binned_weight: float = 0.0,
                 phi_weight: float = 0.0,
                 xy_balance_weight: float = 0.0,
                 normfac: float = 1.0,
                 name: str = 'corrected_composite_loss',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.huber_delta = huber_delta
        self.binned_weight = binned_weight
        self.phi_weight = phi_weight
        self.xy_balance_weight = xy_balance_weight
        self.normfac = normfac

        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()
        # Huber: quadratic for |err| < delta, linear beyond. delta is in
        # normalized target units (so 0.5 means 50 GeV with normfac=100).
        # Keep robust to heavy tails while preserving smooth gradient near the
        # optimum. Lazily constructed when used so it always reflects current
        # huber_delta even if updated post-init.
        self._huber = tf.keras.losses.Huber(delta=huber_delta)


    def call(self, y_true, y_pred):
        mae_loss = tf.reduce_mean(self.mae(y_true, y_pred))
        mse_loss = tf.reduce_mean(self.mse(y_true, y_pred))

        total_loss = (self.mae_weight * mae_loss +
                        self.mse_weight * mse_loss)

        if self.huber_weight > 0:
            huber_loss = tf.reduce_mean(self._huber(y_true, y_pred))
            total_loss = total_loss + self.huber_weight * huber_loss

        # BinnedDeviation conflicts with resolution; only compute when explicitly enabled.
        if self.binned_weight > 0:
            pt_bins = np.array([50., 100., 200., 300., 400., np.inf]) / self.normfac
            binned_loss = compute_binned_deviation(y_true, y_pred, pt_bins=pt_bins)
            total_loss = total_loss + self.binned_weight * binned_loss

        # Optional phi-aware loss for angular reconstruction
        if self.phi_weight > 0:
            phi_loss = compute_phi_loss(y_true, y_pred)
            total_loss = total_loss + self.phi_weight * phi_loss

        # Optional XY balance loss to prevent asymmetric axis resolution
        if self.xy_balance_weight > 0:
            xy_loss = compute_xy_balance_loss(y_true, y_pred)
            total_loss = total_loss + self.xy_balance_weight * xy_loss

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "mae_weight": self.mae_weight,
            "mse_weight": self.mse_weight,
            "huber_weight": self.huber_weight,
            "huber_delta": self.huber_delta,
            "binned_weight": self.binned_weight,
            "phi_weight": self.phi_weight,
            "xy_balance_weight": self.xy_balance_weight,
            "normfac": self.normfac,
        })
        return config

    