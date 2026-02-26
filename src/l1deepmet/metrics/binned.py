import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore

def compute_binned_deviation(y_true, y_pred, pt_bins=None):
        
    px_true = y_true[:, 0]
    py_true = y_true[:, 1]
    px_pred = y_pred[:, 0]
    py_pred = y_pred[:, 1]

    # Safe norm: epsilon avoids NaN gradient when pt_pred -> 0 (0.5/sqrt(0) = inf)
    pt_true = tf.sqrt(tf.square(px_true) + tf.square(py_true) + 1e-8)
    pt_pred = tf.sqrt(tf.square(px_pred) + tf.square(py_pred) + 1e-8)

    error = pt_true - pt_pred

    # filtering out zero pt events
    valid_mask = pt_true > 0
    error_filtered = tf.boolean_mask(error, valid_mask)
    pt_true_filtered = tf.boolean_mask(pt_true, valid_mask)

    if pt_bins is None:
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

class BinnedDeviation(tf.keras.metrics.Metric):
    def __init__(self, pt_bins=None, name='binned_deviation', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.pt_bins = pt_bins

    def update_state(self, y_true, y_pred, sample_weight=None):
        deviation = compute_binned_deviation(y_true, y_pred, self.pt_bins)
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            sw = tf.reduce_mean(sw)
            deviation = deviation * sw
        
        self.total.assign_add(tf.cast(deviation, tf.float32))
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)
    
    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'pt_bins': self.pt_bins})
        return config

