"""Technology-independent per-layer cost table for any Keras model.

For each layer this reports parameters, MACs (multiply-accumulates in
matmul-like ops) and elementwise ops, using the *symbolic tensor shapes*
— which is what makes the numbers honest for per-particle architectures:
a body ``Dense(width)`` on ``(B, 128, C)`` shares its kernel across the
128 particle slots and therefore does ``128 * C * width`` MACs, not
``C * width``.

Counting rules (per example, batch axis excluded):

==========================  =================================================
Layer                       Rule
==========================  =================================================
Dense (incl. QDense)        macs = prod(in[1:-1]) * d_in * units;
                            elementwise = bias adds when use_bias
Embedding                   0 macs (lookup table)
BatchNormalization          0 macs; 2 ops/element; foldable at inference
LayerNormalization          0 macs; ~4 ops/element (approximate)
Activation                  1 op/element; tanh/sigmoid/... -> LUT on FPGA
MultiHeadAttention          QKV+output projections + 2 attention matmuls,
                            from num_heads/key_dim/value_dim in the config;
                            softmax counted as h*N^2 elementwise
Multiply / Add              1 op/output element
Concatenate / Reshape       0 (wiring only)
GlobalAveragePooling1D      (N-1)*F adds + F divides
SumOverParticles            (N-1)*F adds
BoundedWeight               1 op/element; tanh -> LUT on FPGA
ShiftByConstant             1 op/element
CastToInt / PaddingMask /   0 (wiring, casts, constants)
ZeroReduce / Dropout
unknown class               params only, macs=None, listed in ``unhandled``
==========================  =================================================

Dispatch walks ``type(layer).__mro__`` against the handler table, so HGQ2
layers that subclass their Keras counterparts (QDense -> Dense,
QBatchNormalization -> BatchNormalization, ...) are costed for free. Note
that for HGQ2 layers ``params`` includes the quantizer bookkeeping
variables Keras counts as weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

import keras
from keras import layers as KL

from l1deepmet.models.custom_layers import (
    BoundedWeight,
    CastToInt,
    PaddingMask,
    ShiftByConstant,
    SumOverParticles,
    ZeroReduce,
)


@dataclass(frozen=True)
class LayerCost:
    """Cost row for one layer. ``macs is None`` means unhandled class."""
    name: str
    class_name: str
    params: int
    macs: Optional[int]
    elementwise_ops: int
    output_shape: Tuple
    notes: str = ""


@dataclass
class AnalyticReport:
    layers: List[LayerCost] = field(default_factory=list)
    total_params: int = 0
    total_macs: int = 0
    unhandled: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layers": [asdict(row) for row in self.layers],
            "total_params": int(self.total_params),
            "total_macs": int(self.total_macs),
            "unhandled": list(self.unhandled),
        }


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _shape_of(tensor) -> Tuple:
    return tuple(None if d is None else int(d) for d in tensor.shape)


def _input_shapes(layer) -> List[Tuple]:
    try:
        inp = layer.input
    except (AttributeError, ValueError, RuntimeError):
        return []
    if isinstance(inp, (list, tuple)):
        return [_shape_of(t) for t in inp]
    return [_shape_of(inp)]


def _positions(shape: Tuple) -> int:
    """Product of the non-batch, non-channel dims (the weight-sharing axes)."""
    n = 1
    for d in shape[1:-1]:
        n *= d or 1
    return n


def _elements(shape: Tuple) -> int:
    """Per-example output elements."""
    n = 1
    for d in shape[1:]:
        n *= d or 1
    return n


# ---------------------------------------------------------------------------
# Per-class handlers: (layer, input_shapes, output_shape) -> (macs, ew, notes)
# ---------------------------------------------------------------------------

def _h_dense(layer, ins, out):
    d_in = ins[0][-1] or 1
    pos = _positions(ins[0])
    macs = pos * d_in * layer.units
    ew = pos * layer.units if layer.use_bias else 0
    act = layer.get_config().get("activation")
    notes = "" if act in (None, "linear") else f"fused activation: {act}"
    return macs, ew, notes


def _h_embedding(layer, ins, out):
    return 0, 0, "lookup table, no arithmetic"


def _h_batchnorm(layer, ins, out):
    return 0, 2 * _elements(out), "per-element affine; foldable at inference"


def _h_layernorm(layer, ins, out):
    return 0, 4 * _elements(out), "approximate: mean/var reduction + normalize + affine"


def _h_activation(layer, ins, out):
    act = layer.get_config().get("activation", "?")
    if callable(act):
        act = getattr(act, "__name__", str(act))
    lut = act not in ("linear", "relu")
    notes = f"{act}" + (" -> LUT on FPGA" if lut else "")
    return 0, _elements(out), notes


def _h_mha(layer, ins, out):
    cfg = layer.get_config()
    h = cfg["num_heads"]
    k = cfg["key_dim"]
    v = cfg.get("value_dim") or k
    n = out[1] or 1
    d = out[-1] or 1
    macs = (
        n * d * h * k      # Q projection
        + n * d * h * k    # K projection
        + n * d * h * v    # V projection
        + h * n * n * k    # QK^T
        + h * n * n * v    # attention . V
        + n * h * v * d    # output projection
    )
    ew = h * n * n
    return macs, ew, ("assumes self-attention (q/k/v dims from config); "
                      "softmax counted as elementwise")


def _h_elementwise(layer, ins, out):
    return 0, _elements(out), ""


def _h_wiring(layer, ins, out):
    return 0, 0, "wiring only"


def _h_cast(layer, ins, out):
    return 0, 0, "type cast"


def _h_zero_reduce(layer, ins, out):
    return 0, 0, "emits constant zeros (dummy path)"


def _h_padding_mask(layer, ins, out):
    return 0, 0, "attention-mask wiring"


def _h_dropout(layer, ins, out):
    return 0, 0, "inference no-op"


def _h_gap(layer, ins, out):
    n = (ins[0][1] or 1) if ins else 1
    f = _elements(out)
    return 0, n * f, "(N-1) adds + 1 divide per feature"


def _h_sum_particles(layer, ins, out):
    n = (ins[0][1] or 1) if ins else 1
    f = _elements(out)
    return 0, (n - 1) * f, "(N-1) adds per feature"


def _h_shift(layer, ins, out):
    return 0, _elements(out), "constant shift"


def _h_bounded_weight(layer, ins, out):
    return 0, _elements(out), "tanh-bounded affine weight -> tanh LUT on FPGA"


_HANDLERS: dict = {
    KL.Dense: _h_dense,
    KL.Embedding: _h_embedding,
    KL.BatchNormalization: _h_batchnorm,
    KL.LayerNormalization: _h_layernorm,
    KL.Activation: _h_activation,
    KL.MultiHeadAttention: _h_mha,
    KL.Multiply: _h_elementwise,
    KL.Add: _h_elementwise,
    KL.Concatenate: _h_wiring,
    KL.Reshape: _h_wiring,
    KL.Dropout: _h_dropout,
    KL.GlobalAveragePooling1D: _h_gap,
    SumOverParticles: _h_sum_particles,
    CastToInt: _h_cast,
    ZeroReduce: _h_zero_reduce,
    ShiftByConstant: _h_shift,
    BoundedWeight: _h_bounded_weight,
    PaddingMask: _h_padding_mask,
}


def _resolve_handler(layer) -> Optional[Callable]:
    for klass in type(layer).__mro__:
        if klass in _HANDLERS:
            return _HANDLERS[klass]
    return None


def _quantized_note(layer) -> str:
    try:
        from hgq.layers.core.base import QLayerBase
    except ImportError:
        return ""
    if isinstance(layer, QLayerBase):
        return "params include HGQ2 quantizer bookkeeping"
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_model(model: keras.Model) -> AnalyticReport:
    """Build the per-layer cost table for ``model``.

    Input layers are skipped (no compute). Layers whose class (walking the
    MRO) has no handler get a params-only row with ``macs=None`` and are
    listed in ``report.unhandled`` — totals stay finite either way.
    """
    report = AnalyticReport()
    for layer in model.layers:
        if isinstance(layer, KL.InputLayer):
            continue
        out_shape = _shape_of(layer.output)
        params = int(layer.count_params())
        handler = _resolve_handler(layer)
        if handler is None:
            report.layers.append(LayerCost(
                name=layer.name,
                class_name=type(layer).__name__,
                params=params,
                macs=None,
                elementwise_ops=0,
                output_shape=out_shape,
                notes="unhandled layer class",
            ))
            report.unhandled.append(layer.name)
            continue
        macs, ew, notes = handler(layer, _input_shapes(layer), out_shape)
        q_note = _quantized_note(layer)
        if q_note:
            notes = f"{notes}; {q_note}" if notes else q_note
        report.layers.append(LayerCost(
            name=layer.name,
            class_name=type(layer).__name__,
            params=params,
            macs=int(macs),
            elementwise_ops=int(ew),
            output_shape=out_shape,
            notes=notes,
        ))

    report.total_params = sum(r.params for r in report.layers)
    report.total_macs = sum(r.macs for r in report.layers if r.macs is not None)
    return report
