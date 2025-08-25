import tensorflow as tf # type: ignore
import keras # type: ignore
from typing import Any, Dict, Optional  
from l1deepmet.models.base import BaseModel
from l1deepmet.models.output_head import OutputHead, DirectRegressionHead
from l1deepmet.losses.corrected import CorrectedCompositeLoss


# TODO: Implement Dense models
class Dense(BaseModel):
    def __init__(self, config: Dict[str, Any], output_head: Optional[OutputHead] = None):
        super().__init__(config=config, output_head=output_head)
    
    def build_model(self): #TODO: not fully implemented, use functional build for now
        model_cfg: Dict[str, Any] = self.config.get("model", {}) if isinstance(self.config, dict) else {}
        training_cfg: Dict[str, Any] = self.config.get("training", {}) if isinstance(self.config, dict) else {}

        self.units = list(model_cfg.get("units", [64, 32, 16]))
        self.activation_name: str = model_cfg.get("activation", "tanh")
        self.with_bias: bool = bool(model_cfg.get("with_bias", False))
        self.t_mode: int = int(training_cfg.get("mode", 0))

        self.feature_stack = []
        for i, width in enumerate(self.units):
            self.feature_stack.append(
                tf.keras.layers.Dense(
                    width,
                    activation=None,
                    kernel_initializer="lecun_uniform",
                    name=f"dense_{i}"
                )
            )
            #TODO: add option to not use batchnorm
            self.feature_stack.append(
                tf.keras.layers.BatchNormalization(momentum=0.95, name=f"bn_{i}")
            )
            self.feature_stack.append(
                tf.keras.layers.Activation(self.activation_name, name=f"act_{i}")
            )

        # Heads / utilities per mode
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D(name="pool")

        if self.t_mode == 0:
            self.out_dense = tf.keras.layers.Dense(2, activation="linear", name="output")

        elif self.t_mode == 1:
            # Optional per-candidate bias for px,py
            self.bias_dense = tf.keras.layers.Dense(2, activation="linear", name="met_bias") if self.with_bias else None
            # Per-candidate weight
            self.weight_dense = tf.keras.layers.Dense(1, activation="linear", name="met_weight")

        elif self.t_mode == 2:
            # Single global weight per event
            self.single_weight_dense = tf.keras.layers.Dense(1, activation="linear", name="puppi_weight")

    

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Expected dict inputs with keys produced by the data loader
        # - 'continuous_inputs': (B, N, 5)
        # - 'momentum_inputs':   (B, N, 2)
        # Categorical inputs may be present but are unused in this Dense model (no embedding).
        if isinstance(inputs, dict):
            x = inputs.get("continuous_inputs", None)
            pxpy = inputs.get("momentum_inputs", None)
        else:
            # Fallback: assume tuple/list like (continuous, momentum, ...)
            x = inputs[0]
            pxpy = inputs[1]

        # Feature stack over continuous candidate features only
        for layer in self.feature_stack:
            x = layer(x, training=training) if hasattr(layer, "training") else layer(x)

        if self.t_mode == 0:
            x = self.global_pool(x)
            outputs = self.out_dense(x)
            return outputs

        if self.t_mode == 1:
            # Optionally adjust pxpy with a learned bias (per candidate)
            if self.with_bias and self.bias_dense is not None:
                bias = self.bias_dense(x)
                pxpy = pxpy + bias

            # Per-candidate weight, then elementwise multiply and sum over candidates
            w = self.weight_dense(x)  # (B, N, 1)
            weighted = w * pxpy  # broadcasts to (B, N, 2)
            outputs = tf.reduce_sum(weighted, axis=1, name="output")  # (B, 2)
            return outputs

        if self.t_mode == 2:
            # Single global weight times event-level sum of px,py
            pooled = self.global_pool(x)
            w = self.single_weight_dense(pooled)  # (B, 1)
            pmet = tf.reduce_sum(pxpy, axis=1, name="pmet")  # (B, 2)
            outputs = w * pmet  # (B, 2) via broadcasting
            return outputs

        # Default fallback (should not hit): behave like mode 0
        x = self.global_pool(x)
        outputs = tf.keras.layers.Dense(2, activation="linear")(x)
        return outputs

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "DenseModel",
            "trainable": True,
            "dtype": "float32",
            "config": self.config
        }
    

class DenseQuantized(Dense):
    def __init__(self, config: Dict[str, Any]):
        # Placeholder: quantization not implemented yet
        super().__init__(config=config)

    def forward(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # TODO: Implement quantized layers once quantization scheme is defined
        raise NotImplementedError("DenseQuantized is not implemented yet")

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "DenseQuantizedModel",
            "trainable": True,
            "dtype": None,  # to be defined when quantization is implemented
            "config": self.config
        }
    

class DenseHGQ(Dense):
    def __init__(self, config: Dict[str, Any]):
        # Placeholder: HGQ-specific kernels/precision to be defined
        super().__init__(config=config)

    def forward(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # TODO: Implement HGQ-specific behavior when available
        raise NotImplementedError("DenseHGQ is not implemented yet")

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "DenseHGQModel",
            "trainable": True,
            "dtype": "mixed_HGQ",
            "config": self.config
        }
