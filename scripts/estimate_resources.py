#!/usr/bin/env python
"""Estimate FPGA resources for a saved model — no synthesis run.

Loads a ``.keras`` checkpoint (full-precision or HGQ2, auto-detected),
runs every applicable synthesis-free estimator and writes
``estimate.json`` + ``estimate.txt`` to the output directory:

- analytic per-layer params/MACs table (any model);
- HGQ2 only: saved EBOPs, calibration-recomputed EBOPs, and da4ml
  LUT-equivalent cost + latency tracing.

Full-precision models get the analytic table only — there is no honest
synthesis-free LUT/DSP number for float models, and the report says so
instead of inventing one.

Usage:
    python scripts/estimate_resources.py \
        --model reports/qat_validation_may2026/hgq2/best_model.keras \
        --output-dir outputs/estimation/hgq2_baseline

    # real-data calibration instead of the synthetic default:
    python scripts/estimate_resources.py --model ... --output-dir ... \
        --calibration-h5 data/preprocessed/<tag>/test.h5 --n-calibration 512

Exit codes: 0 ok, 2 bad input (model missing/unloadable).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
# Shared-node politeness: numpy/OpenBLAS otherwise spawns nproc threads,
# which trips per-user pthread limits on busy hosts (same rationale as
# the TF caps in arch_search.py).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

log = logging.getLogger("estimate_resources")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, type=Path,
                   help="Path to a saved Keras model (best_model.keras).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write estimate.json + estimate.txt.")
    p.add_argument("--calibration-h5", type=Path, default=None,
                   help="Preprocessed .h5 file (features dataset) for the "
                        "EBOPs recompute. Default: synthetic random batch.")
    p.add_argument("--n-calibration", type=int, default=512,
                   help="Number of calibration events (default 512).")
    p.add_argument("--no-trace", action="store_true",
                   help="Skip the da4ml LUT-cost/latency trace.")
    p.add_argument("--no-ebops-recompute", action="store_true",
                   help="Only report the EBOPs stored in the checkpoint.")
    p.add_argument("--per-layer-trace", action="store_true",
                   help="Also compute cumulative per-layer trace costs "
                        "(second trace, slower).")
    p.add_argument("--adder-size", type=int, default=1,
                   help="da4ml HWConfig adder size (default 1).")
    p.add_argument("--carry-size", type=int, default=-1,
                   help="da4ml HWConfig carry size (default -1: unbounded).")
    p.add_argument("--latency-cutoff", type=float, default=-1.0,
                   help="da4ml HWConfig latency cutoff (default -1: none).")
    p.add_argument("--inputs-kif", type=int, nargs=3, default=None,
                   metavar=("K", "I", "F"),
                   help="Quantize model inputs to (keep_negative, "
                        "integer_bits, fraction_bits) before tracing. "
                        "Required for models that do arithmetic on raw "
                        "inputs before any Q-layer.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _load_calibration(model, args) -> list:
    """Calibration inputs in ``model.inputs`` order."""
    import numpy as np

    from l1deepmet.estimation import synthetic_calibration_batch

    if args.calibration_h5 is None:
        log.info("using synthetic calibration batch (n=%d)", args.n_calibration)
        return synthetic_calibration_batch(model, n=args.n_calibration)

    import h5py

    from l1deepmet.quantization import encode_split_for_hgq2

    with h5py.File(str(args.calibration_h5), "r") as f:
        features = f["features"][:args.n_calibration]
    inputs_dict, _ = encode_split_for_hgq2(
        features, np.zeros((len(features), 2), dtype=np.float32)
    )
    log.info("loaded %d calibration events from %s",
             len(features), args.calibration_h5)
    return [np.asarray(inputs_dict[tensor.name]) for tensor in model.inputs]


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from l1deepmet.estimation import estimate_resources, format_report, write_report
    from l1deepmet.models.serialization import is_hgq2_model, load_any_model

    if not args.model.is_file():
        log.error("model file not found: %s", args.model)
        return 2
    try:
        model = load_any_model(args.model)
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        log.error("could not load %s: %s: %s",
                  args.model, type(exc).__name__, exc)
        return 2

    calibration = None
    if is_hgq2_model(model) and not args.no_ebops_recompute:
        calibration = _load_calibration(model, args)

    trace_kwargs = {
        "adder_size": args.adder_size,
        "carry_size": args.carry_size,
        "latency_cutoff": args.latency_cutoff,
        "per_layer": args.per_layer_trace,
        "verbose": args.verbose,
    }
    if args.inputs_kif is not None:
        trace_kwargs["inputs_kif"] = tuple(args.inputs_kif)

    estimate = estimate_resources(
        model,
        model_path=str(args.model),
        calibration=calibration,
        run_trace=not args.no_trace,
        trace_kwargs=trace_kwargs,
    )

    print(format_report(estimate))
    paths = write_report(estimate, args.output_dir)
    log.info("wrote %s and %s", paths["json"], paths["txt"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
