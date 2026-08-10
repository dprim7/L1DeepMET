"""Loading saved L1DeepMET models of either family (full-precision or
HGQ2-QAT) without the caller knowing which subset of custom objects the
checkpoint references.

Promoted from ``scripts/synthesize.py::load_custom_objects`` (which now
delegates here) so estimation and synthesis code in ``src/`` never
imports from ``scripts/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import keras

from l1deepmet.losses.corrected import CorrectedCompositeLoss
from l1deepmet.models.custom_layers import (
    BoundedWeight,
    CastToInt,
    PaddingMask,
    ShiftByConstant,
    SumOverParticles,
    ZeroReduce,
)


def get_custom_objects() -> dict:
    """All the custom Keras layers and losses our models may use.

    Centralized so any driver can load any saved model from any ablation
    (full-precision or HGQ2-QAT) without knowing which subset of layers
    was used. The HGQ2 section is skipped when hgq isn't installed.
    """
    co: dict = {
        # Full-precision custom layers (l1deepmet.models.custom_layers).
        "CastToInt": CastToInt,
        "ZeroReduce": ZeroReduce,
        "ShiftByConstant": ShiftByConstant,
        "PaddingMask": PaddingMask,
        "SumOverParticles": SumOverParticles,
        "BoundedWeight": BoundedWeight,
        "CorrectedCompositeLoss": CorrectedCompositeLoss,
    }

    # HGQ2 quantized layers. Add every public Q* class (and the
    # quantizer-config / activation primitives saved models can reference).
    try:
        import hgq.layers as _hgq_layers
        for _name in dir(_hgq_layers):
            if _name.startswith("Q") and _name[1:2].isupper():
                co[_name] = getattr(_hgq_layers, _name)
        # The non-Q-prefixed Activation in hgq.layers.activation we use.
        from hgq.layers.activation import Activation as _QActivation
        co.setdefault("Activation", _QActivation)
        # HGQ2 quantizer config types referenced by the saved model JSON.
        import hgq.quantizer.config as _qcfg
        for _name in dir(_qcfg):
            cls = getattr(_qcfg, _name, None)
            if isinstance(cls, type) and not _name.startswith("_"):
                co.setdefault(_name, cls)
        import hgq.constraints as _hc
        for _name in dir(_hc):
            cls = getattr(_hc, _name, None)
            if isinstance(cls, type) and not _name.startswith("_"):
                co.setdefault(_name, cls)
        import hgq.regularizers as _hr
        for _name in dir(_hr):
            cls = getattr(_hr, _name, None)
            if isinstance(cls, type) and not _name.startswith("_"):
                co.setdefault(_name, cls)
    except ImportError:
        pass

    return co


def load_any_model(path: Union[str, Path], *, compile: bool = False) -> keras.Model:
    """Load a saved ``.keras`` model with the full custom-objects dict.

    Works for both full-precision and HGQ2 checkpoints. ``compile=False``
    (the default) skips optimizer/loss restoration — estimation and
    synthesis only need the forward graph.
    """
    return keras.models.load_model(
        str(path), custom_objects=get_custom_objects(), compile=compile
    )


def is_hgq2_model(model: keras.Model) -> bool:
    """True when the model contains at least one HGQ2 quantized layer.

    Returns False (rather than raising) when hgq isn't installed — a
    checkpoint that needs hgq layers would already have failed to load.
    """
    try:
        from hgq.layers.core.base import QLayerBase
    except ImportError:
        return False
    return any(isinstance(layer, QLayerBase) for layer in model._flatten_layers())
