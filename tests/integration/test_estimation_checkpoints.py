"""Integration: run the estimation stack on the real May-2026 checkpoints.

These pin the numbers recorded when the module was built, anchoring both
the checkpoint-loading semantics (across e.g. hgq2/keras upgrades) and
the estimator implementations:

- HGQ2 checkpoint: saved EBOPs total == 10036 (degenerate by design —
  the known MonoL1 bit-width collapse — but stable), da4ml trace runs.
- fp32 checkpoint: 9,953 params, body dense_0 MACs == 128*11*64.

Marked slow: full-size traces take minutes.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

ROOT = Path(__file__).resolve().parents[2]
HGQ2_CKPT = ROOT / "reports/qat_validation_may2026/hgq2/best_model.keras"
FP32_CKPT = (ROOT / "reports/qat_validation_may2026/fp32/"
             "bounded_mae_only_xy0_seed42/best_model.keras")

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not HGQ2_CKPT.is_file(), reason="HGQ2 checkpoint missing")
class TestHgq2Checkpoint:
    @pytest.fixture(scope="class")
    def model(self):
        pytest.importorskip("hgq")
        from l1deepmet.models.serialization import load_any_model

        return load_any_model(HGQ2_CKPT)

    def test_saved_ebops_total_pinned(self, model):
        from l1deepmet.estimation import collect_ebops

        report = collect_ebops(model)
        assert report.total == 10036
        assert report.n_q_layers == 13
        assert report.source == "saved"
        by_layer = {row.layer: row.ebops for row in report.per_layer}
        assert by_layer["qweight_pxpy"] == 6921
        assert by_layer["qavg_over_particles"] == 1833

    def test_analytic_covers_all_layers(self, model):
        from l1deepmet.estimation import analyze_model

        assert analyze_model(model).unhandled == []

    def test_trace_cost_positive(self, model):
        pytest.importorskip("da4ml")
        pytest.importorskip("da4ml.converter.hgq2")
        from l1deepmet.estimation import trace_costs

        report = trace_costs(model)
        assert report.cost > 0
        assert report.latency_max >= report.latency_min > 0
        assert report.n_outputs == 2

    def test_recompute_with_synthetic_calibration(self, model):
        from l1deepmet.estimation import (
            recompute_ebops,
            synthetic_calibration_batch,
        )

        batch = synthetic_calibration_batch(model, n=64, seed=42)
        report = recompute_ebops(model, batch)
        assert report.source == "recomputed"
        assert report.n_q_layers == 13


@pytest.mark.skipif(not FP32_CKPT.is_file(), reason="fp32 checkpoint missing")
class TestFp32Checkpoint:
    @pytest.fixture(scope="class")
    def estimate(self):
        from l1deepmet.estimation import estimate_resources
        from l1deepmet.models.serialization import load_any_model

        model = load_any_model(FP32_CKPT)
        return estimate_resources(model, model_path=str(FP32_CKPT))

    def test_family_and_pinned_totals(self, estimate):
        assert estimate.model_family == "fp32"
        assert estimate.analytic.total_params == 9953
        assert estimate.analytic.unhandled == []

    def test_body_dense_macs_rank3(self, estimate):
        by_name = {row.name: row for row in estimate.analytic.layers}
        assert by_name["dense_0"].macs == 128 * 11 * 64

    def test_no_fabricated_hardware_numbers(self, estimate):
        assert estimate.trace is None
        assert estimate.ebops is None
        assert "hgq2" in estimate.errors["trace"].lower()
