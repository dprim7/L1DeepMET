"""Synthesis-free FPGA resource estimation for L1DeepMET models.

Submodules, split by dependency weight:

- ``analytic`` — technology-independent per-layer params/MACs table.
  Pure Keras; works on any model (full-precision or HGQ2).
- ``ebops``   — HGQ2 EBOPs (bit-operation) collection and recompute.
  Requires ``hgq``; imported lazily inside functions.
- ``tracing`` — da4ml LUT-cost / latency tracing of HGQ2 models.
  Requires ``da4ml`` with a compatible hgq2; imported lazily.
- ``report``  — assembly of all estimators into one artifact.

Importing this package must never fail just because ``hgq`` or ``da4ml``
is absent or incompatible — heavy/optional imports live inside the
functions that need them.
"""
from l1deepmet.estimation.analytic import AnalyticReport, LayerCost, analyze_model
from l1deepmet.estimation.ebops import (
    EbopsReport,
    EbopsRow,
    collect_ebops,
    recompute_ebops,
    synthetic_calibration_batch,
)

__all__ = [
    "AnalyticReport",
    "LayerCost",
    "analyze_model",
    "EbopsReport",
    "EbopsRow",
    "collect_ebops",
    "recompute_ebops",
    "synthetic_calibration_batch",
]
