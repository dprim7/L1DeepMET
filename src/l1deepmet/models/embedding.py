import tensorflow as tf
from typing import Dict, List, Any, Optional


class CategoricalEmbedding(tf.keras.layers.Layer):
    """
    Embedding layer that separates continuous and categorical features.
    TODO: check match with legacy repo
    """
    
    def __init__(self, 
                 embedding_dims: List[int] = [4, 13, 16],
                 output_dim: int = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.output_dim = output_dim
        
    def build(self, input_shape):
        self.embeddings = []
        for i, input_dim in enumerate(self.embedding_dims):
            self.embeddings.append(
                tf.keras.layers.Embedding(
                    input_dim=input_dim,
                    output_dim=self.output_dim,
                    embeddings_initializer=tf.keras.initializers.RandomNormal(
                        mean=0, stddev=0.4/self.output_dim
                    ),
                    name=f'embedding_{i}'
                )
            )
        
    def forward(self, inputs):

        continuous = inputs[0] #TODO: this is wrong, should be a list of inputs
        pxpy = inputs[1] # TODO: same as above
        categorical = inputs[2:]
        
        # Embed categorical features
        embedded = [emb(cat) for emb, cat in zip(self.embeddings, categorical)]
        
        # Concatenate everything
        return tf.concat([continuous, pxpy] + embedded, axis=-1)


class GraphEmbedding(tf.keras.layers.Layer):
    """TODO: Implement"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        raise NotImplementedError("GraphEmbedding not implemented yet")
