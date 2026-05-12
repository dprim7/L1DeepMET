"""Default hls4ml config for L1DeepMET models.

Mirrors L1METML's convert_full_model.py choices (CMS L1 Phase-2 standard
target board, 5 ns clock period, parallel I/O, latency-optimized strategy),
plus tighter precisions for the categorical (pdgId / charge) input branches.

The returned config is a plain dict that callers can further customize before
handing to ``hls4ml.converters.convert_from_keras_model``.
"""
from __future__ import annotations

from typing import Mapping, Optional

import tensorflow as tf  # type: ignore

# ─── CMS L1 Phase-2 deployment defaults ──────────────────────────────────────
# AMD/Xilinx Virtex UltraScale+ VU13P on a Serenity ATCA blade — the standard
# CMS L1 Phase-2 testbed used by both L1METML and the recent CMS-L1T-Jet-Tagging
# DeepSets paper (arXiv:2509.24371). 5 ns clock = 200 MHz.
DEFAULT_PART = "xcvu13p-flga2577-2-e"
DEFAULT_CLOCK_PERIOD_NS = 5.0
DEFAULT_IO_TYPE = "io_parallel"           # 'io_stream' would pipeline candidates
DEFAULT_STRATEGY = "Latency"              # 'Resource' is smaller but slower

# Full-precision starting point (Q2a per project plan). Plenty of bits so that
# fixed-point quantization is not the dominant error source. Once we have a
# working baseline we'll sweep this down to find the resolution/LUT knee.
DEFAULT_PRECISION = "ap_fixed<32,16>"     # 32 bits total, 16 integer

# Tight precisions for known-bounded categorical inputs in the L1DeepMET model.
# pdgId is encoded 0–5 (6 classes), charge is encoded 0–3 (4 classes) — 4-bit
# unsigned ints suffice.
CATEGORICAL_INPUT_NAMES = ("pdgid_inputs", "charge_inputs")
CATEGORICAL_PRECISION = "ap_uint<4>"


def build_hls_config(
    model: tf.keras.Model,
    *,
    default_precision: str = DEFAULT_PRECISION,
    default_reuse_factor: int = 1,
    strategy: str = DEFAULT_STRATEGY,
    trace_layers: bool = True,
    granularity: str = "name",
    layer_precisions: Optional[Mapping[str, str]] = None,
) -> dict:
    """Build an hls4ml config dict tuned for L1DeepMET models.

    Args:
        model:                 trained Keras model.
        default_precision:     fallback ``ap_fixed`` / ``ap_int`` string for any
                               layer not explicitly overridden.
        default_reuse_factor:  hls4ml ``reuse_factor`` (1 = full parallel, more LUTs;
                               larger = more time-multiplexing, fewer LUTs).
        strategy:              ``"Latency"`` (default, fastest, more LUTs) or
                               ``"Resource"`` (slower, fewer LUTs).
        trace_layers:          enable per-layer tensor tracing for the bit-accurate
                               profiling pass. Costs nothing at synthesis time.
        granularity:           ``"name"`` for per-layer config (recommended) or
                               ``"model"`` for one config for everything.
        layer_precisions:      optional dict ``{layer_name: precision_str}`` to
                               override specific layers (e.g. tighter precision
                               for integer inputs, wider for accumulators).

    Returns:
        Plain Python dict suitable for ``hls4ml.converters.convert_from_keras_model``.
    """
    import hls4ml  # local import — hls4ml warns at import time

    config = hls4ml.utils.config_from_keras_model(
        model,
        granularity=granularity,
        default_reuse_factor=default_reuse_factor,
        default_precision=default_precision,
    )
    config["Model"]["Strategy"] = strategy

    if granularity == "name":
        # Enable tracing globally so the validate step can plot per-layer
        # numerics. Free at synthesis time, cheap at conversion time.
        for layer_name in config["LayerName"]:
            config["LayerName"][layer_name]["Trace"] = trace_layers

        # Categorical inputs: replace the float default with ap_uint<4>.
        # Only applies if a layer with that name exists.
        for name in CATEGORICAL_INPUT_NAMES:
            if name in config["LayerName"]:
                config["LayerName"][name].setdefault("Precision", {})
                config["LayerName"][name]["Precision"]["result"] = CATEGORICAL_PRECISION

        # User-provided per-layer overrides last so they win.
        if layer_precisions:
            for layer_name, precision in layer_precisions.items():
                if layer_name not in config["LayerName"]:
                    raise KeyError(
                        f"layer_precisions: layer {layer_name!r} not in model. "
                        f"Available: {list(config['LayerName'])}"
                    )
                config["LayerName"][layer_name].setdefault("Precision", {})
                config["LayerName"][layer_name]["Precision"]["result"] = precision

    return config


def pretty_print(config: dict, indent: int = 0) -> None:
    """Pretty-print a nested config dict to stdout (no logging deps)."""
    align = 22
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + str(key))
            pretty_print(value, indent + 1)
        else:
            print(f"{'  ' * indent}{key}{':':>{align - len(key) - 2 * indent}} {value}")
