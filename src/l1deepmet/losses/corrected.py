import tensorflow as tf # type: ignore
import numpy as np # type: ignore

class CorrectedCompositeLoss(tf.keras.losses.Loss):
    """ Composite loss that combines MSE, MAE, as 
    well as a physics-informed binned loss.
    Each loss is weighted via config.
    """
    def __init__(self,
                 mae_weight: float = 1.0,
                 mse_weight: float = 1.0,
                 binned_weight: float = 200.0,
                 name: str = 'corrected_composite_loss',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.binned_weight = binned_weight

        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()


    def call(self, y_true, y_pred):

        mae_loss = tf.reduce_mean(self.mae(y_true, y_pred))
        mse_loss = tf.reduce_mean(self.mse(y_true, y_pred))
        binned_loss = self._compute_binned_deviation(y_true, y_pred)

        total_loss = (self.mae_weight * mae_loss +
                        self.mse_weight * mse_loss +
                        self.binned_weight * binned_loss)
        
        return total_loss
    
    def _compute_binned_deviation(self, y_true, y_pred):
        
        px_true = y_true[:, 0]
        py_true = y_true[:, 1]
        px_pred = y_pred[:, 0]
        py_pred = y_pred[:, 1]

        # compute MET (with tf for graph compatibility) TODO: check if this matters
        pt_true = tf.sqrt(tf.square(px_true) + tf.square(py_true))
        pt_pred = tf.sqrt(tf.square(px_pred) + tf.square(py_pred))

        error = pt_true - pt_pred

        # filtering out zero pt events
        valid_mask = pt_true > 0
        error_filtered = tf.boolean_mask(error, valid_mask)
        pt_true_filtered = tf.boolean_mask(pt_true, valid_mask)

        pt_bins = np.array([50., 100., 200., 300., 400., np.inf]) #TODO: check if normfactor needed

        # computing deviations
        total_deviation = 0.0
        for i in range(len(pt_bins) - 1):
            if i < len(pt_bins) - 2:
                bin_mask = tf.logical_and(
                    pt_true_filtered >= pt_bins[i],
                    pt_true_filtered < pt_bins[i + 1]
                )
            else:
                #last bin: pt > 400 GeV
                bin_mask = pt_true_filtered >= pt_bins[i]
            
            bin_errors = tf.boolean_mask(error_filtered, bin_mask)

            # splitting positive and negative errors
            pos_errors = tf.boolean_mask(bin_errors, bin_errors > 0)
            neg_errors = tf.boolean_mask(bin_errors, bin_errors < 0)

            # computing asymmetry in the bin
            bin_deviation = tf.abs(
                tf.reduce_sum(pos_errors) + tf.reduce_sum(neg_errors)   
            )
            total_deviation += bin_deviation

        # normalization by total pt
        normalization = tf.reduce_sum(pt_true_filtered)
        normalized_deviation = total_deviation / (normalization + 1e-7) #norm shouldn't be zero, but being safe

        return normalized_deviation
    