"""hls4ml synthesis pipeline for L1DeepMET.

End-to-end: take a trained Keras model, build an hls4ml fixed-point project,
optionally run Vivado HLS synthesis, and verify the bit-accurate predictions
against the original Keras model on a held-out test set.

Public surface:
    config:   build_hls_config()        — sensible defaults for L1DeepMET
    convert:  convert_to_hls()          — keras model + config → HLS project
    validate: compare_keras_vs_hls()    — bit-accurate numerical comparison
"""

from .config import build_hls_config, DEFAULT_PART, DEFAULT_PRECISION
from .convert import convert_to_hls, ConversionResult
from .export import export_for_hls
from .validate import compare_keras_vs_hls, ValidationResult

__all__ = [
    "build_hls_config",
    "DEFAULT_PART",
    "DEFAULT_PRECISION",
    "convert_to_hls",
    "ConversionResult",
    "export_for_hls",
    "compare_keras_vs_hls",
    "ValidationResult",
]
