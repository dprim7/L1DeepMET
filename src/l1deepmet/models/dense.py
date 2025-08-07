import tensorflow as tf # type: ignore
import keras # type: ignore
from typing import Any, Dict, Optional  
from l1deepmet.models.base import BaseModel
from l1deepmet.models.output_head import OutputHead, DirectRegressionHead
from l1deepmet.losses.corrected import CorrectedCompositeLoss


#TODO: Implement Dense models
class Dense(BaseModel):
    def __init__(self):
        pass
    
    def build_model(self):
        pass

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        pass

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "DenseModel",
            "trainable": True,
            "dtype": "float32",
            "config": self.config
        }
    

class DenseQuantized(Dense):
    def __init__(self):
        pass

    def build_model(self):
        pass

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        pass

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "DenseQuantizedModel",
            "trainable": True,
            "dtype": None, #fill with fitting value pattern
            "config": self.config
        }
    

class DenseHGQ(Dense):
    def __init__(self):
        pass

    def build_model(self):
        pass

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        pass

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "DenseHGQModel",
            "trainable": True,
            "dtype": "mixed_HGQ",
            "config": self.config
        }
