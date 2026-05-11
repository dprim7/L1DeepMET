"""Quantization-aware training (QAT) for L1DeepMET models.

Two paths supported:

  - ``hgq2`` — Heterogeneous Gradient-based Quantization v2. Bit-widths are
    *learned* per-parameter / per-channel via gradient descent, jointly with
    the weights. Recommended default for FPGA deployment. Trained models
    convert to hls4ml directly without an export rewrite — every layer is
    already a quantized primitive that hls4ml understands.

  - ``qkeras`` — manual per-layer bit-width selection. Useful when you have
    prior knowledge of the precision budget. Not implemented yet.

The HGQ2 architecture is functionally equivalent to the full-precision
``arch_search.build_model`` body (Dense / BN / ReLU stack, scalar weight
head, multiply-and-sum reduction), but uses HGQ2 quantized layers and a
one-hot replacement for the Embedding lookup (HGQ2 lacks a QEmbedding).

Entry point: :func:`hgq2_model.build_hgq2_model`.
"""

from .data import (
    encode_split_for_hgq2,
    make_hgq2_tf_dataset_from_features,
    make_hgq2_tf_dataset_from_h5,
)
from .hgq2_model import build_hgq2_model, one_hot_encode_features

__all__ = [
    "build_hgq2_model",
    "one_hot_encode_features",
    "encode_split_for_hgq2",
    "make_hgq2_tf_dataset_from_features",
    "make_hgq2_tf_dataset_from_h5",
]
