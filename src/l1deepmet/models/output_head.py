from abc import ABC, abstractmethod
import tensorflow as tf  # type: ignore
from typing import Any, Dict, Optional

# TODO: implement px,py multiplication where needed
# TODO: fix input dimensions in output head
class OutputHead(ABC):
    """
    Abstract base class for output head strategies.
    MET can be regressed directly or by estimating per particle or per event weight(s).
    """

    @abstractmethod
    def build(self, input_dim: int) -> tf.keras.layers.Layer:
        pass

    @property 
    @abstractmethod
    def name(self) -> str:
        pass

class DirectRegressionHead(OutputHead):

    def __init__(self):
        self.output_dim = 2

    def build(self, input_dim: int) -> tf.keras.layers.Layer:
        return tf.keras.layers.Dense(
            self.output_dim,
            name='direct_regression_head',
        ) #TODO: check activation/initialization needs

    @property
    def name(self) -> str:
        return 'direct_regression'
    
#TODO: implement
class ParticleWeightHead(OutputHead):
    pass

#TODO: implement
class METWeightHead(OutputHead):
    pass