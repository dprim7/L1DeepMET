"""Convert a trained Keras model to an hls4ml HLS C++ project.

This is the thin wrapper around ``hls4ml.converters.convert_from_keras_model``
that knows about L1DeepMET's defaults (VU13P, 5 ns, parallel I/O) and our
custom layers. The returned :class:`ConversionResult` is a small dataclass
bundling the hls4ml model, the resolved config, and metadata used by
downstream validation / reporting.

Conversion is the cheap step (seconds). Synthesis (``hls_model.build(synth=True)``)
takes ~30 minutes per model and is opt-in via the CLI.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import tensorflow as tf  # type: ignore

from .config import (
    DEFAULT_CLOCK_PERIOD_NS,
    DEFAULT_IO_TYPE,
    DEFAULT_PART,
    DEFAULT_PRECISION,
    DEFAULT_STRATEGY,
    build_hls_config,
    pretty_print,
)

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Bundle returned by :func:`convert_to_hls`.

    Attributes:
        hls_model:    compiled hls4ml model; call ``.predict(X)`` for bit-accurate
                      simulation results.
        config:       the resolved hls4ml config dict (post-customization).
        output_dir:   absolute path to the generated HLS C++ project.
        keras_model:  the input Keras model (kept here for convenience so the
                      validate step doesn't need to pass it around separately).
        meta:         provenance: part, precision, clock, etc.
    """

    hls_model: Any
    config: dict
    output_dir: Path
    keras_model: tf.keras.Model
    meta: dict = field(default_factory=dict)


def convert_to_hls(
    model: tf.keras.Model,
    output_dir: str | os.PathLike,
    *,
    project_name: str = "l1deepmet",
    part: str = DEFAULT_PART,
    clock_period_ns: float = DEFAULT_CLOCK_PERIOD_NS,
    io_type: str = DEFAULT_IO_TYPE,
    strategy: str = DEFAULT_STRATEGY,
    default_precision: str = DEFAULT_PRECISION,
    default_reuse_factor: int = 1,
    layer_precisions: Optional[Mapping[str, str]] = None,
    print_config: bool = False,
    save_keras_copy: bool = True,
) -> ConversionResult:
    """Run hls4ml conversion end-to-end.

    Args:
        model:                 trained Keras model.
        output_dir:            where to write the HLS C++ project.
        project_name:          HLS project name (used as a C++ symbol prefix).
        part:                  Xilinx part number. Default = CMS L1 Phase-2 VU13P.
        clock_period_ns:       target clock period. Default 5 ns (200 MHz).
        io_type:               ``'io_parallel'`` or ``'io_stream'``.
        strategy:              ``'Latency'`` (default) or ``'Resource'``.
        default_precision:     full-model default ``ap_fixed`` precision.
        default_reuse_factor:  reuse factor (1 = full parallel).
        layer_precisions:      ``{layer_name: precision_str}`` overrides.
        print_config:          dump the resolved config to stdout before convert.
        save_keras_copy:       also save the input Keras model to ``output_dir``
                               (mirrors L1METML practice — useful for re-runs).

    Returns:
        :class:`ConversionResult`.
    """
    import hls4ml  # local — hls4ml prints noisy warnings at import

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_keras_copy:
        try:
            model.save(output_dir / "keras_model.keras")
        except Exception as e:
            logger.warning("Could not save Keras copy to %s: %s", output_dir, e)

    logger.info("Building hls4ml config…")
    config = build_hls_config(
        model,
        default_precision=default_precision,
        default_reuse_factor=default_reuse_factor,
        strategy=strategy,
        layer_precisions=layer_precisions,
    )
    if print_config:
        print("-" * 72)
        pretty_print(config)
        print("-" * 72)

    logger.info(
        "Converting to HLS: part=%s clock=%.1fns io=%s strategy=%s precision=%s rf=%d",
        part, clock_period_ns, io_type, strategy,
        default_precision, default_reuse_factor,
    )
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(output_dir),
        part=part,
        clock_period=clock_period_ns,
        project_name=project_name,
    )

    logger.info("Compiling bit-accurate simulation library…")
    hls_model.compile()

    # Render the model graph to PNG (useful for debugging conversion mismatches).
    try:
        hls4ml.utils.plot_model(
            hls_model, show_shapes=True, show_precision=True,
            to_file=str(output_dir / "model_hls4ml.png"),
        )
    except Exception as e:
        logger.warning("Could not render model graph: %s", e)

    meta = {
        "part": part,
        "clock_period_ns": clock_period_ns,
        "io_type": io_type,
        "strategy": strategy,
        "default_precision": default_precision,
        "default_reuse_factor": default_reuse_factor,
        "project_name": project_name,
        "n_keras_params": int(model.count_params()),
    }
    logger.info("Conversion complete: %s", output_dir)
    return ConversionResult(
        hls_model=hls_model,
        config=config,
        output_dir=output_dir,
        keras_model=model,
        meta=meta,
    )
