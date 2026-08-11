"""Tests for ``l1deepmet.estimation.report`` — assembly of the analytic,
EBOPs and da4ml-trace sections into one ResourceEstimate artifact.

Honesty rule pinned here: full-precision models get the analytic table
only; the EBOPs and trace sections are recorded as skipped with a reason
in ``errors`` instead of fabricating numbers. Optional sections never
raise out of ``estimate_resources``.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

import keras

from l1deepmet.estimation.report import (
    ResourceEstimate,
    estimate_resources,
    format_report,
    write_report,
)


@pytest.fixture()
def tiny_fp32_model():
    from l1deepmet.models.custom_layers import BoundedWeight, SumOverParticles

    pxpy = keras.Input(shape=(16, 2), name="momentum_inputs")
    raw = keras.layers.Dense(1, name="met_weight")(pxpy)
    w = BoundedWeight(name="bounded")(raw)
    weighted = keras.layers.Multiply(name="weighted")([w, pxpy])
    out = SumOverParticles(name="met")(weighted)
    return keras.Model(pxpy, out)


class TestFp32Estimate:
    def test_analytic_only_with_reasons(self, tiny_fp32_model):
        est = estimate_resources(tiny_fp32_model, model_path="tiny.keras")
        assert isinstance(est, ResourceEstimate)
        assert est.model_family == "fp32"
        assert est.analytic.total_params == tiny_fp32_model.count_params()
        assert est.ebops is None
        assert est.ebops_recomputed is None
        assert est.trace is None
        assert "hgq2" in est.errors["trace"].lower()
        assert "hgq2" in est.errors["ebops"].lower()

    def test_never_fabricates_lut_numbers(self, tiny_fp32_model):
        payload = estimate_resources(tiny_fp32_model).to_dict()
        assert payload["trace"] is None
        assert payload["ebops"] is None


class TestHgq2Estimate:
    @pytest.fixture(scope="class")
    def estimate(self):
        pytest.importorskip("hgq")
        from l1deepmet.estimation.ebops import synthetic_calibration_batch
        from l1deepmet.quantization import build_hgq2_model

        model = build_hgq2_model(width=8, depth=1, use_embeddings=True,
                                 bounded_weight=True, n_particles=16)
        calibration = synthetic_calibration_batch(model, n=16, seed=42)
        return estimate_resources(model, model_path="tiny_hgq2.keras",
                                  calibration=calibration, run_trace=False)

    def test_family_and_sections(self, estimate):
        assert estimate.model_family == "hgq2"
        assert estimate.analytic.unhandled == []
        assert estimate.ebops is not None
        assert estimate.ebops.source == "saved"
        assert estimate.ebops_recomputed is not None
        assert estimate.ebops_recomputed.source == "recomputed"
        assert estimate.ebops_recomputed.total > 0

    def test_trace_skip_reason_recorded(self, estimate):
        assert estimate.trace is None
        assert "trace" in estimate.errors

    def test_no_calibration_records_reason(self):
        pytest.importorskip("hgq")
        from l1deepmet.quantization import build_hgq2_model

        model = build_hgq2_model(width=8, depth=1, n_particles=16)
        est = estimate_resources(model, run_trace=False)
        assert est.ebops_recomputed is None
        assert "calibration" in est.errors["ebops_recomputed"].lower()


class TestFormatAndWrite:
    def test_format_report_mentions_layers_and_totals(self, tiny_fp32_model):
        est = estimate_resources(tiny_fp32_model, model_path="tiny.keras")
        text = format_report(est)
        assert "met_weight" in text
        assert str(est.analytic.total_params) in text
        assert "fp32" in text

    def test_write_report_creates_artifacts(self, tiny_fp32_model, tmp_path):
        est = estimate_resources(tiny_fp32_model, model_path="tiny.keras")
        paths = write_report(est, tmp_path / "out")
        assert paths["json"].is_file()
        assert paths["txt"].is_file()

        data = json.loads(paths["json"].read_text())
        assert set(data) >= {"model_path", "model_family", "analytic",
                             "ebops", "ebops_recomputed", "trace", "errors"}
        assert data["model_family"] == "fp32"
        assert data["analytic"]["total_macs"] == est.analytic.total_macs

    def test_to_dict_json_round_trip(self, tiny_fp32_model):
        est = estimate_resources(tiny_fp32_model)
        round_tripped = json.loads(json.dumps(est.to_dict()))
        assert round_tripped["analytic"]["total_params"] == \
            est.analytic.total_params
