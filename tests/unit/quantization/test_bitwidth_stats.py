"""Tests for the bit-width / sparsity extractor used by the QAT report.

``scripts/experiments/qat_validation.py::extract_bitwidth_stats`` walks a
trained HGQ2 model and summarizes the learned bit-widths per layer:
total parameters, zero-bit (pruned) count, average kernel bit-width,
total kernel-bit budget. The report uses these for the headline table
and for the per-layer breakdown.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

pytest.importorskip("hgq")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

from qat_validation import extract_bitwidth_stats  # noqa: E402


@pytest.fixture(scope="module")
def saved_hgq2_model(tmp_path_factory):
    """Build and save a tiny HGQ2 model so the extractor has something to walk."""
    from l1deepmet.quantization import build_hgq2_model
    m = build_hgq2_model(width=8, depth=1, use_embeddings=True,
                         bounded_weight=False)
    tmp = tmp_path_factory.mktemp("bw")
    path = tmp / "model.keras"
    m.save(path)
    return path


def test_returns_expected_keys(saved_hgq2_model):
    stats = extract_bitwidth_stats(saved_hgq2_model)
    for k in ("per_layer", "total_kernel_bits", "total_kernel_zeros",
              "total_kernel_params", "sparsity_pct", "avg_kernel_bitwidth"):
        assert k in stats, f"missing key {k!r}"


def test_per_layer_rows_have_expected_shape(saved_hgq2_model):
    stats = extract_bitwidth_stats(saved_hgq2_model)
    assert len(stats["per_layer"]) > 0
    sample = stats["per_layer"][0]
    for k in ("layer", "kind", "shape", "min", "max", "mean",
              "n_zero", "n_total", "sum_bits"):
        assert k in sample, f"per-layer row missing key {k!r}"


def test_includes_kernel_quantizer_rows(saved_hgq2_model):
    """Every HGQ2 layer with a trainable kernel has a 'kq' row."""
    stats = extract_bitwidth_stats(saved_hgq2_model)
    kq_layers = [r["layer"] for r in stats["per_layer"] if r["kind"] == "kq"]
    # We built width=8 depth=1 with embeddings + bounded_weight=False:
    # expected QDense layers are emb_pdgid, emb_charge, qdense_0, qmet_weight,
    # qweight_shift, output (the fixed-N·I final scale).
    for expected_layer in ("qdense_0", "qmet_weight"):
        assert expected_layer in kq_layers, \
            f"expected {expected_layer!r} in kernel-quantizer layers, got {kq_layers}"


def test_kernel_totals_consistent_with_per_layer(saved_hgq2_model):
    stats = extract_bitwidth_stats(saved_hgq2_model)
    expected_total = sum(r["n_total"] for r in stats["per_layer"] if r["kind"] == "kq")
    expected_zeros = sum(r["n_zero"] for r in stats["per_layer"] if r["kind"] == "kq")
    expected_bits = sum(int(r["sum_bits"]) for r in stats["per_layer"] if r["kind"] == "kq")
    assert stats["total_kernel_params"] == expected_total
    assert stats["total_kernel_zeros"] == expected_zeros
    assert stats["total_kernel_bits"] == expected_bits


def test_sparsity_in_zero_to_hundred(saved_hgq2_model):
    stats = extract_bitwidth_stats(saved_hgq2_model)
    assert 0.0 <= stats["sparsity_pct"] <= 100.0


def test_untrained_model_has_default_8_bit_kernel(saved_hgq2_model):
    """Right after build, HGQ2's KIFConfig defaults give 8-bit kernel
    quantizers (kif_weight_default: i0=2, f0=6 since hgq2 0.1.9; the 0.1.0
    default was 4 bits). Documents the prior we'd see if the model never
    trained."""
    stats = extract_bitwidth_stats(saved_hgq2_model)
    assert stats["avg_kernel_bitwidth"] == pytest.approx(8.0, rel=0.01)


def test_falls_back_gracefully_on_unloadable_path(tmp_path):
    """Pointing at a nonexistent file shouldn't crash."""
    stats = extract_bitwidth_stats(tmp_path / "does_not_exist.keras")
    assert stats["per_layer"] == []
    assert "error" in stats
