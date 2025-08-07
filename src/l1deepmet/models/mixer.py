import tensorflow as tf # type: ignore
import keras # type: ignore
from typing import Any, Dict, Optional  
from hgq.config import QuantizerConfigScope, LayerConfigScope, QuantizerConfig
from hgq.layers import QEinsumDense, QEinsumDenseBatchnorm, QAdd

from l1deepmet.models.base import BaseModel
from l1deepmet.models.output_head import OutputHead, DirectRegressionHead
from l1deepmet.losses.corrected import CorrectedCompositeLoss

class Mixer(BaseModel): #TODO implement
    def __init__(self):
        super().__init__()
    
    def build_model(self):
        pass

    def forward(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        pass

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "MixerModel",
            "dtype": "float32",
            "config": self.config
        }

class MixerQuantized(Mixer): #TODO: implement
    def __init__(self):
        super().__init__()

    def build_model(self):
        pass

    def forward(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        pass

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "MixerQuantizedModel",
            "dtype": None,  # fill with fitting value pattern
            "config": self.config
        }
    
class MixerHGQ(Mixer):
    def __init__(self, config: Dict[str, Any], output_head: OutputHead = None):
        super().__init__(config, output_head)

    def build_model(self):
        n_puppi_cands = self.config.get("n_puppi_cands", 128)
        n_features = self.config.get("n_features", 8)

        self.iq_conf = self.config.get("iq_conf", QuantizerConfig(bits=6, int_bits=2))
        self.iq_default = self.config.get("iq_default", QuantizerConfig(bits=8, int_bits=3))
        
        # Token mixing layers
        self.token_mix1 = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (self.N, 16), bias_axes='C', 
            activation='relu', iq_conf=self.iq_conf, name='token_mix1'
        )
        self.token_mix2 = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (self.N, self.n), bias_axes='C', 
            activation='relu', name='token_mix2'
        )
        
        # Cross-token mixing
        self.cross_token = QEinsumDenseBatchnorm(
            'bnc,nN->bNc', (self.N, self.n), bias_axes='N', 
            name='cross_token'
        )
        
        # Residual add
        self.residual_add = QAdd(iq_confs=(self.iq_conf, self.iq_default))
        
        # Post-residual token mixing
        self.post_res1 = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (self.N, 16), bias_axes='C', 
            activation='relu', name='post_res1'
        )
        self.post_res2 = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (self.N, 16), bias_axes='C', 
            activation='relu', name='post_res2'
        )
        
        # Token aggregation
        self.token_agg = QEinsumDense('bnc,n->bc', 16, name='token_agg')
        
        # Final dense layers
        self.dense1 = QEinsumDenseBatchnorm(
            'bc,cC->bC', 16, bias_axes='C', 
            activation='relu', name='dense1'
        )
        self.dense2 = QEinsumDenseBatchnorm(
            'bc,cC->bC', 16, bias_axes='C', 
            activation='relu', name='dense2'
        )
        self.dense3 = QEinsumDenseBatchnorm(
            'bc,cC->bC', 16, bias_axes='C', 
            activation='relu', name='dense3'
        )

    def forward(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x1 = self.token_mix1(inputs, training=training)
        x1 = self.token_mix2(x1, training=training)
        
        # Cross-token interaction
        x2 = self.cross_token(x1, training=training)
        
        # Residual connection
        x = self.residual_add([inputs, x2])
        
        # More token mixing
        x = self.post_res1(x, training=training)
        x = self.post_res2(x, training=training)
        
        # Aggregate tokens
        x = self.token_agg(x)
        
        # Final dense processing
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        

        return x

