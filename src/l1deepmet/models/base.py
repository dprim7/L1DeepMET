from abc import ABC, abstractmethod
import tensorflow as tf # type: ignore
from typing import Any, Dict, Optional
from l1deepmet.models.output_head import OutputHead, DirectRegressionHead, ParticleWeightHead, METWeightHead

class BaseModel(ABC, tf.keras.Model):

    def __init__(self, config: Dict[str, Any], output_head: OutputHead = None):
        super().__init__()
        self.config = config
        self.build_model()

        if output_head is None:
            self.outputhead = DirectRegressionHead()
        else:
            self.output_head = output_head

    @abstractmethod
    def build_model(self):
        pass
    
    @abstractmethod
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

     