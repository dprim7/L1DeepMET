"""da4ml LUT-cost and latency tracing for HGQ2 models — no HLS run.

da4ml replays the Keras graph symbolically as fixed-point variables and
prices every operation in LUT-equivalent cost for a fully-parallel
distributed-arithmetic implementation; ``comb_trace`` then gives total
cost and (min, max) latency in adder stages. This is a *relative*
hardware-cost axis, not a Vivado report.

Two compatibility notes handled here:

- ``da4ml.converter.hgq2`` imports hgq2 classes that only exist from
  hgq2 0.1.9 — all converter imports are deferred into functions and an
  incompatible pairing raises :class:`TracingUnavailableError` at call
  time, keeping ``l1deepmet.estimation`` importable regardless.
- da4ml's handler registry is exact-class keyed and has no entry for
  plain ``keras.layers.Activation`` — which is what
  ``build_hgq2_model``'s relu/tanh layers are (hgq's "QActivation" is an
  alias of the Keras class). :func:`register_extra_handlers` adds one:
  linear passes through, relu maps to da4ml's native op, and any other
  function goes through ``FixedVariableArray.apply`` so the downstream
  Q-layer's input quantizer materializes the LUT cost (the same
  mechanism da4ml uses for ``QUnaryFunctionLUT``).
- da4ml 0.5.2's ``ReplayMerge`` also mishandles plain
  ``keras.layers.Concatenate``: it broadcasts all inputs against each
  other before dispatching, which fails for the normal concat case of
  unequal channel widths (e.g. our (N,5) continuous ++ (N,4) embedding).
  :func:`register_extra_handlers` overrides the registry entry with a
  straight ``np.concatenate``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


class TracingUnavailableError(RuntimeError):
    """The da4ml hgq2 converter cannot be imported in this environment."""


_HANDLERS_REGISTERED = False


def _require_converter():
    try:
        from da4ml.converter.hgq2 import trace_model
        return trace_model
    except ImportError as exc:
        raise TracingUnavailableError(
            "da4ml's hgq2 converter failed to import — it requires "
            "HGQ2>=0.1.9 (see environment.yaml). Underlying error: "
            f"{exc}"
        ) from exc


def register_extra_handlers() -> None:
    """Register the plain-``Activation`` replay handler with da4ml.

    Idempotent; called automatically by :func:`check_traceable` and
    :func:`trace_costs`.
    """
    global _HANDLERS_REGISTERED
    if _HANDLERS_REGISTERED:
        return
    _require_converter()

    import keras
    import numpy as np
    from da4ml.converter.hgq2.layers._base import ReplayOperationBase
    from da4ml.trace.ops import relu as da_relu

    class ReplayPlainActivation(ReplayOperationBase):  # noqa: F841
        """Replay ``keras.layers.Activation`` (incl. hgq's alias of it)."""

        handles = (keras.layers.Activation,)

        def call(self, x):
            act = self.op.activation
            if act is keras.activations.linear:
                return x
            if act is keras.activations.relu:
                return da_relu(x)

            def as_numpy_fn(values: np.ndarray) -> np.ndarray:
                tensor = keras.ops.convert_to_tensor(values[None])
                return np.asarray(keras.ops.convert_to_numpy(act(tensor)))[0]

            # Exact-valued deferred array; the next quantizer prices the LUT.
            return x.apply(as_numpy_fn)

    class ReplayConcatenate(ReplayOperationBase):  # noqa: F841
        """Override da4ml's ReplayMerge for plain ``Concatenate``.

        ReplayMerge broadcasts all inputs against each other before
        dispatching, which fails for concatenation of unequal channel
        widths. Registering after da4ml's import wins the exact-class
        registry slot.
        """

        handles = (keras.layers.Concatenate,)

        def call(self, inputs):
            axis = self.op.axis
            # Traced arrays have the batch axis stripped.
            if axis > 0:
                axis -= 1
            return np.concatenate(inputs, axis=axis)

    _HANDLERS_REGISTERED = True


def check_traceable(model) -> List[str]:
    """Names of layers whose class has no da4ml replay handler.

    Mirrors the parser's exact-class registry lookup (no MRO walk!) and
    its InputLayer skip. Empty list means :func:`trace_costs` will not
    fail on layer coverage.
    """
    register_extra_handlers()

    import keras
    from da4ml.converter.hgq2.layers._base import _registry

    unsupported = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        if type(layer) not in _registry:
            unsupported.append(layer.name)
    return unsupported


@dataclass
class TraceReport:
    cost: float
    latency_min: float
    latency_max: float
    hwconfig: Tuple[int, int, float]
    n_inputs: int
    n_outputs: int
    per_layer: Optional[List[Tuple[str, float]]] = None

    def to_dict(self) -> dict:
        return {
            "cost": float(self.cost),
            "latency_min": float(self.latency_min),
            "latency_max": float(self.latency_max),
            "hwconfig": list(self.hwconfig),
            "n_inputs": int(self.n_inputs),
            "n_outputs": int(self.n_outputs),
            "per_layer": ([[name, float(cost)] for name, cost in self.per_layer]
                          if self.per_layer is not None else None),
        }


def trace_costs(model, *, adder_size: int = 1, carry_size: int = -1,
                latency_cutoff: float = -1.0,
                inputs_kif: Optional[Tuple[int, int, int]] = None,
                per_layer: bool = False, verbose: bool = False) -> TraceReport:
    """Trace ``model`` and return LUT-equivalent cost and latency.

    Args:
        adder_size / carry_size / latency_cutoff: da4ml ``HWConfig``
            fields; the defaults ``(1, -1, -1)`` mean fully combinational
            with unbounded carry (da4ml's own defaults).
        inputs_kif: optional (keep_negative, integer_bits, fraction_bits)
            quantization applied to the model inputs. ``None`` leaves
            inputs unquantized, which is fine when every input feeds a
            Q-layer first (its input quantizer bounds the range). Models
            that do arithmetic on raw inputs before any quantizer — e.g.
            ``build_hgq2_model(use_embeddings=False)``'s zero-kernel
            dummy Adds — REQUIRE this, or the trace fails on the
            unbounded sentinel ranges.
        per_layer: also compute, per layer, the cumulative cost of
            producing that layer's output tensor from the model inputs
            (opt-in: a second trace; ``None`` on any mapping failure).
        verbose: forward da4ml's per-op progress printing.

    Raises:
        TracingUnavailableError: the converter cannot be imported.
        ValueError: the model contains layers with no replay handler
            (all offending names listed) — e.g. full-precision models
            with the l1deepmet custom layers.
    """
    register_extra_handlers()

    from da4ml.trace import HWConfig, comb_trace

    trace_model = _require_converter()

    unsupported = check_traceable(model)
    if unsupported:
        raise ValueError(
            "model contains layers da4ml cannot trace: "
            f"{unsupported}. da4ml tracing supports HGQ2 models only; "
            "for full-precision models use the analytic table instead."
        )

    hwconf = HWConfig(adder_size, carry_size, latency_cutoff)
    try:
        flat_inputs, flat_outputs = trace_model(
            model, hwconf=hwconf, inputs_kif=inputs_kif, verbose=verbose
        )
    except ValueError as exc:
        if "math domain error" in str(exc) and inputs_kif is None:
            raise ValueError(
                "tracing failed on unbounded input ranges — the model does "
                "arithmetic on raw inputs before any Q-layer input "
                "quantizer bounds them. Pass inputs_kif=(keep_negative, "
                "integer_bits, fraction_bits) to quantize the model inputs."
            ) from exc
        raise
    solution = comb_trace(flat_inputs, flat_outputs)
    latency_min, latency_max = solution.latency

    per_layer_rows = None
    if per_layer:
        per_layer_rows = _per_layer_costs(model, hwconf, inputs_kif)

    return TraceReport(
        cost=float(solution.cost),
        latency_min=float(latency_min),
        latency_max=float(latency_max),
        hwconfig=(adder_size, carry_size, latency_cutoff),
        n_inputs=int(flat_inputs.shape[0]),
        n_outputs=int(flat_outputs.shape[0]),
        per_layer=per_layer_rows,
    )


def _per_layer_costs(model, hwconf, inputs_kif) -> Optional[List[Tuple[str, float]]]:
    """Cumulative cost of each layer's output tensor, via a dump trace.

    Uses da4ml's private ``_apply_nn``/``_flatten_arr`` instead of the
    public ``trace_model(dump=True)``: the latter flattens every dumped
    tensor eagerly and crashes on deferred (Retarded) arrays — exactly
    what the tanh-LUT mechanism produces. Here deferred tensors are
    simply skipped per-tensor (they have no standalone cost; the next
    quantizer's output carries it). Keys the dump by
    ``layer.output.name`` — valid because the dump comes from the same
    model object. Returns None (totals unaffected) if the mapping fails.
    """
    import keras
    from da4ml.converter.hgq2.parser import _apply_nn, _flatten_arr
    from da4ml.trace import FixedVariableArrayInput, comb_trace

    _require_converter()

    try:
        inputs = tuple(
            FixedVariableArrayInput(tensor.shape[1:], hwconf=hwconf)
            for tensor in model.inputs
        )
        if inputs_kif is not None:
            inputs = tuple(inp.quantize(*inputs_kif) for inp in inputs)
        tensor_map = _apply_nn(model, inputs, dump=True)
        flat_inputs = _flatten_arr(inputs)

        rows: List[Tuple[str, float]] = []
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            tensor_name = getattr(layer.output, "name", None)
            if tensor_name not in tensor_map:
                continue
            try:
                solution = comb_trace(flat_inputs,
                                      _flatten_arr(tensor_map[tensor_name]))
            except Exception:  # noqa: BLE001 — deferred/unpriceable tensor
                continue
            rows.append((layer.name, float(solution.cost)))
        return rows or None
    except Exception:  # noqa: BLE001 — opt-in feature degrades to None
        return None
