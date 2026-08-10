"""Assembly of all synthesis-free estimators into one artifact.

``estimate_resources`` runs whatever applies to the model family and
never lets an optional section raise — failures and deliberate skips are
recorded as reason strings in ``ResourceEstimate.errors`` instead of
fabricated numbers. Full-precision models get the analytic table only:
EBOPs and da4ml LUT cost are only meaningful for HGQ2-quantized models.

Section order matters and is fixed: saved EBOPs are collected *before*
the calibration recompute (which overwrites them in-model), and the
da4ml trace runs *after* the recompute so it sees calibration-refreshed
quantizer ranges.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

from l1deepmet.estimation.analytic import AnalyticReport, analyze_model
from l1deepmet.estimation.ebops import (
    EbopsReport,
    collect_ebops,
    recompute_ebops,
)
from l1deepmet.estimation.tracing import TraceReport, trace_costs
from l1deepmet.models.serialization import is_hgq2_model

_NOT_HGQ2_EBOPS = ("skipped: not an HGQ2 model — EBOPs only exist for "
                   "quantized layers")
_NOT_HGQ2_TRACE = ("skipped: not an HGQ2 model — da4ml LUT-cost tracing "
                   "needs quantized layers, and no honest synthesis-free "
                   "LUT proxy exists for float models (use the analytic "
                   "MACs table)")


@dataclass
class ResourceEstimate:
    model_path: str
    model_family: str  # "hgq2" | "fp32"
    analytic: AnalyticReport
    ebops: Optional[EbopsReport] = None
    ebops_recomputed: Optional[EbopsReport] = None
    trace: Optional[TraceReport] = None
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_family": self.model_family,
            "analytic": self.analytic.to_dict(),
            "ebops": self.ebops.to_dict() if self.ebops else None,
            "ebops_recomputed": (self.ebops_recomputed.to_dict()
                                 if self.ebops_recomputed else None),
            "trace": self.trace.to_dict() if self.trace else None,
            "errors": dict(self.errors),
        }


def estimate_resources(model, *, model_path: str = "", calibration=None,
                       run_ebops: bool = True, run_trace: bool = True,
                       trace_kwargs: Optional[dict] = None) -> ResourceEstimate:
    """Run every applicable estimator on ``model``.

    Args:
        model: a loaded Keras model (either family).
        model_path: recorded in the artifact for provenance.
        calibration: inputs for the EBOPs recompute (list of arrays in
            ``model.inputs`` order, e.g. from
            ``synthetic_calibration_batch``). ``None`` skips the
            recompute with a reason. Note the recompute mutates the
            model's EBOPs weights and WRAP quantizer ranges.
        run_ebops / run_trace: disable sections wholesale.
        trace_kwargs: forwarded to :func:`trace_costs`.
    """
    family = "hgq2" if is_hgq2_model(model) else "fp32"
    estimate = ResourceEstimate(
        model_path=str(model_path),
        model_family=family,
        analytic=analyze_model(model),
    )

    if family == "fp32":
        estimate.errors["ebops"] = _NOT_HGQ2_EBOPS
        estimate.errors["ebops_recomputed"] = _NOT_HGQ2_EBOPS
        estimate.errors["trace"] = _NOT_HGQ2_TRACE
        return estimate

    if run_ebops:
        try:
            estimate.ebops = collect_ebops(model)
        except Exception as exc:  # noqa: BLE001 — section must not raise
            estimate.errors["ebops"] = f"failed: {type(exc).__name__}: {exc}"
        if calibration is not None:
            try:
                estimate.ebops_recomputed = recompute_ebops(model, calibration)
            except Exception as exc:  # noqa: BLE001
                estimate.errors["ebops_recomputed"] = \
                    f"failed: {type(exc).__name__}: {exc}"
        else:
            estimate.errors["ebops_recomputed"] = (
                "skipped: no calibration data provided"
            )
    else:
        estimate.errors["ebops"] = "skipped: run_ebops=False"
        estimate.errors["ebops_recomputed"] = "skipped: run_ebops=False"

    if run_trace:
        try:
            estimate.trace = trace_costs(model, **(trace_kwargs or {}))
        except Exception as exc:  # noqa: BLE001
            estimate.errors["trace"] = f"failed: {type(exc).__name__}: {exc}"
    else:
        estimate.errors["trace"] = "skipped: run_trace=False"

    return estimate


def format_report(estimate: ResourceEstimate) -> str:
    """Human-readable fixed-width rendering of the estimate."""
    lines = []
    lines.append(f"Resource estimate — {estimate.model_path or '(in-memory model)'}")
    lines.append(f"model family: {estimate.model_family}")

    analytic = estimate.analytic
    lines.append("")
    lines.append(f"Analytic table (technology-independent): "
                 f"{analytic.total_params:,} params, "
                 f"{analytic.total_macs:,} MACs")
    lines.append(f"  {'layer':<24}{'class':<28}{'params':>10}{'macs':>12}"
                 f"{'elemwise':>10}  {'output':<15} notes")
    for row in analytic.layers:
        macs = f"{row.macs:,}" if row.macs is not None else "?"
        shape = str(tuple(row.output_shape))
        lines.append(f"  {row.name:<24}{row.class_name:<28}{row.params:>10,}"
                     f"{macs:>12}{row.elementwise_ops:>10,}  {shape:<15} "
                     f"{row.notes}")
    if analytic.unhandled:
        lines.append(f"  unhandled layers: {', '.join(analytic.unhandled)}")

    for label, section in (("EBOPs (saved)", estimate.ebops),
                           ("EBOPs (recomputed)", estimate.ebops_recomputed)):
        if section is None:
            continue
        lines.append("")
        suffix = ("  [all-zero: quantizers untrained or collapsed]"
                  if section.all_zero else "")
        lines.append(f"{label}: total {section.total:,} over "
                     f"{section.n_q_layers} Q-layers{suffix}")
        for row in section.per_layer:
            beta = f"{row.beta:g}" if row.beta is not None else "-"
            lines.append(f"  {row.layer:<24}{row.class_name:<28}"
                         f"{row.ebops:>10,}  beta={beta}")

    if estimate.trace is not None:
        trace = estimate.trace
        lines.append("")
        lines.append(f"da4ml trace (LUT-equivalent, fully-parallel DA): "
                     f"cost {trace.cost:,.1f}, latency "
                     f"{trace.latency_min:g}-{trace.latency_max:g} stages, "
                     f"hwconfig {trace.hwconfig}")
        if trace.per_layer:
            lines.append("  cumulative cost by layer output:")
            for name, cost in trace.per_layer:
                lines.append(f"    {name:<24}{cost:>12,.1f}")

    if estimate.errors:
        lines.append("")
        lines.append("Skipped / failed sections:")
        for key, reason in estimate.errors.items():
            lines.append(f"  {key}: {reason}")

    return "\n".join(lines) + "\n"


def write_report(estimate: ResourceEstimate,
                 output_dir: Union[str, Path]) -> Dict[str, Path]:
    """Write ``estimate.json`` and ``estimate.txt`` under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "estimate.json"
    txt_path = output_dir / "estimate.txt"
    json_path.write_text(json.dumps(estimate.to_dict(), indent=2) + "\n")
    txt_path.write_text(format_report(estimate))
    return {"json": json_path, "txt": txt_path}
