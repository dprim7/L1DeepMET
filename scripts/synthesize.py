#!/usr/bin/env python3
"""End-to-end hls4ml conversion + validation for a trained L1DeepMET model.

By default runs:
  1. Load the saved Keras model.
  2. Build the hls4ml config (CMS L1 VU13P / 5 ns / parallel / Latency / ap_fixed<32,16>).
  3. Convert to HLS C++ project under ``output_dir``.
  4. Compile bit-accurate simulation lib (.so).
  5. Validate against ``test.h5``: predictions, response, physics card,
     diagnostic plots, JSON summary.

Vivado HLS C-synth + Vivado RTL synth are NOT run (Q3a). Pass ``--synth`` later
when we want real LUT/DSP numbers; not implemented yet.

Example:
    python scripts/synthesize.py \
        --model /path/to/best_model.keras \
        --output-dir hls_output_bounded_no_bias \
        --n-validation-events 1000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Same TF threading hygiene as the rest of the project — shared system, please
# don't spawn 256 threads.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")

# Allow ``scripts/`` to find the local custom layer module ``arch_search``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import tensorflow as tf  # type: ignore  # noqa: E402

from l1deepmet.synthesis import (  # noqa: E402
    DEFAULT_PART, DEFAULT_PRECISION, compare_keras_vs_hls, convert_to_hls,
    export_for_hls,
)


def load_custom_objects() -> dict:
    """All the custom Keras layers and losses our models may use.

    Centralized here so the CLI can load any saved model from any ablation
    (full-precision or HGQ2-QAT) without the caller having to know which
    subset of layers was used.
    """
    from arch_search import (
        BoundedWeight, CastToInt, ShiftByConstant, SumOverParticles, ZeroReduce,
    )
    from l1deepmet.losses.corrected import CorrectedCompositeLoss

    co: dict = {
        # Full-precision custom layers (scripts/arch_search.py).
        "CastToInt": CastToInt,
        "ZeroReduce": ZeroReduce,
        "ShiftByConstant": ShiftByConstant,
        "SumOverParticles": SumOverParticles,
        "BoundedWeight": BoundedWeight,
        "CorrectedCompositeLoss": CorrectedCompositeLoss,
    }

    # HGQ2 quantized layers. Add every public Q* class (and the
    # quantizer-config / activation primitives saved models can reference).
    # Optional: skip if HGQ2 isn't installed in this env.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, type=Path,
                   help="Path to a saved Keras model (best_model.keras).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write the HLS C++ project + plots + validation.json.")
    p.add_argument("--test-h5",
                   default="/home/users/dprimosc/L1DeepMET/preprocessed/25Jul8_140X_v0/test.h5",
                   type=Path, help="Preprocessed test split for validation.")

    # hls4ml knobs
    p.add_argument("--precision", default=DEFAULT_PRECISION,
                   help=f"Default ap_fixed precision string. Default: {DEFAULT_PRECISION}")
    p.add_argument("--reuse-factor", type=int, default=1,
                   help="hls4ml reuse_factor (1 = full parallel).")
    p.add_argument("--strategy", default="Latency", choices=("Latency", "Resource"))
    p.add_argument("--io-type", default="io_parallel", choices=("io_parallel", "io_stream"))
    p.add_argument("--part", default=DEFAULT_PART, help="Xilinx part. Default = VU13P.")
    p.add_argument("--clock-period-ns", type=float, default=5.0)
    p.add_argument("--project-name", default="l1deepmet")
    p.add_argument("--print-config", action="store_true",
                   help="Dump the resolved hls4ml config to stdout before conversion.")

    # Validation
    p.add_argument("--normfac", type=float, default=100.0,
                   help="Training-time target normalization factor (model output × normfac = GeV).")
    p.add_argument("--n-validation-events", type=int, default=1000,
                   help="Subsample size for bit-accurate validation. hls_model.predict "
                        "is the bottleneck; 1000 events ≈ a few minutes.")
    p.add_argument("--skip-validation", action="store_true",
                   help="Convert + compile only; don't run any predictions.")
    p.add_argument("--trace-layers", action="store_true",
                   help="Also save per-layer Keras-vs-hls4ml scatter plots.")

    # Future: Vivado HLS synth (Q3b)
    p.add_argument("--synth", action="store_true",
                   help="(Not yet implemented) Trigger Vivado HLS C-synthesis after conversion.")

    p.add_argument("--no-export-pass", action="store_true",
                   help="Skip the export_for_hls custom-layer rewrite. By default we "
                        "rebuild the trained model with hls4ml-friendly primitives "
                        "(no CastToInt, no BoundedWeight, no SumOverParticles).")

    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    log = logging.getLogger("synthesize")

    if args.synth:
        log.error("--synth is not implemented yet (Q3a path is convert-only). Aborting.")
        return 2

    if not args.model.is_file():
        log.error("Model file not found: %s", args.model)
        return 2

    log.info("Loading Keras model: %s", args.model)
    custom = load_custom_objects()
    model = tf.keras.models.load_model(args.model, custom_objects=custom, compile=False)
    log.info("Loaded source model: %d params, %d layers",
             model.count_params(), len(model.layers))

    if not args.no_export_pass:
        log.info("Running export_for_hls pass (replace custom layers with hls4ml-friendly primitives)…")
        model = export_for_hls(model)
        log.info("Export-friendly model: %d params, %d layers",
                 model.count_params(), len(model.layers))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = convert_to_hls(
        model,
        output_dir=args.output_dir,
        project_name=args.project_name,
        part=args.part,
        clock_period_ns=args.clock_period_ns,
        io_type=args.io_type,
        strategy=args.strategy,
        default_precision=args.precision,
        default_reuse_factor=args.reuse_factor,
        print_config=args.print_config,
    )

    # Save the resolved meta — useful for tracking precision sweeps later.
    with open(args.output_dir / "synthesize_meta.json", "w") as f:
        json.dump(
            {
                "model_path": str(args.model.resolve()),
                **result.meta,
                "skip_validation": args.skip_validation,
            },
            f, indent=2,
        )

    if args.skip_validation:
        log.info("Conversion done (validation skipped). Project: %s", result.output_dir)
        return 0

    log.info("Running bit-accurate validation (n=%d events)…", args.n_validation_events)
    val = compare_keras_vs_hls(
        keras_model=model,
        hls_model=result.hls_model,
        test_h5_path=args.test_h5,
        output_dir=args.output_dir,
        normfac=args.normfac,
        n_events=args.n_validation_events,
        make_plots=True,
        trace_layers=args.trace_layers,
    )

    print()
    print("=" * 70)
    print(f"Bit-accurate validation summary ({val.n_events} events)")
    print("=" * 70)
    print(f"{'metric':<28} {'keras':>12} {'hls4ml':>12} {'Δ (hls−k)':>12}")
    print("-" * 70)
    for key in ("met_pt_resolution", "met_x_resolution", "met_y_resolution",
                "phi_resolution", "auc", "mean_response"):
        if key in val.keras_card and key in val.hls_card:
            k = val.keras_card[key]
            h = val.hls_card[key]
            d = val.delta_card.get(key, 0.0)
            print(f"{key:<28} {k:>12.4f} {h:>12.4f} {d:>+12.4f}")
    print()
    print(f"Plots:    {args.output_dir}/profiling_MET*.png, response_MET.png")
    print(f"Summary:  {args.output_dir}/validation.json")
    print(f"HLS code: {args.output_dir}/firmware/")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
