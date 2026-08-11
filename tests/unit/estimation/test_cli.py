"""Tests for ``scripts/estimate_resources.py`` — the CLI driver.

Exercises ``main(argv)`` directly (no subprocess): load a saved model,
auto-detect its family, run the applicable estimators, write
``estimate.json`` + ``estimate.txt``. Exit codes: 0 ok, 2 bad input.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

import keras  # noqa: E402

from estimate_resources import main  # noqa: E402


@pytest.fixture(scope="module")
def fp32_checkpoint(tmp_path_factory):
    from l1deepmet.models.custom_layers import BoundedWeight, SumOverParticles

    pxpy = keras.Input(shape=(16, 2), name="momentum_inputs")
    raw = keras.layers.Dense(1, name="met_weight")(pxpy)
    w = BoundedWeight(name="bounded")(raw)
    weighted = keras.layers.Multiply(name="weighted")([w, pxpy])
    out = SumOverParticles(name="met")(weighted)
    model = keras.Model(pxpy, out)

    path = tmp_path_factory.mktemp("fp32") / "best_model.keras"
    model.save(path)
    return path


@pytest.fixture(scope="module")
def hgq2_checkpoint(tmp_path_factory):
    pytest.importorskip("hgq")
    from l1deepmet.quantization import build_hgq2_model

    model = build_hgq2_model(width=8, depth=1, use_embeddings=True,
                             bounded_weight=True, n_particles=16)
    path = tmp_path_factory.mktemp("hgq2") / "best_model.keras"
    model.save(path)
    return path


class TestFp32Cli:
    def test_writes_artifacts_and_detects_family(self, fp32_checkpoint,
                                                 tmp_path, capsys):
        out_dir = tmp_path / "est"
        code = main(["--model", str(fp32_checkpoint),
                     "--output-dir", str(out_dir)])
        assert code == 0

        data = json.loads((out_dir / "estimate.json").read_text())
        assert data["model_family"] == "fp32"
        assert data["trace"] is None
        assert (out_dir / "estimate.txt").is_file()

        stdout = capsys.readouterr().out
        assert "met_weight" in stdout


class TestHgq2Cli:
    def test_no_trace_flag_and_ebops_sections(self, hgq2_checkpoint,
                                              tmp_path):
        out_dir = tmp_path / "est"
        code = main(["--model", str(hgq2_checkpoint),
                     "--output-dir", str(out_dir),
                     "--no-trace", "--n-calibration", "8"])
        assert code == 0

        data = json.loads((out_dir / "estimate.json").read_text())
        assert data["model_family"] == "hgq2"
        assert data["ebops"] is not None
        assert data["ebops_recomputed"]["total"] > 0

    def test_no_ebops_recompute_flag(self, hgq2_checkpoint, tmp_path):
        out_dir = tmp_path / "est"
        code = main(["--model", str(hgq2_checkpoint),
                     "--output-dir", str(out_dir),
                     "--no-trace", "--no-ebops-recompute"])
        assert code == 0
        data = json.loads((out_dir / "estimate.json").read_text())
        assert data["ebops_recomputed"] is None


class TestBadInput:
    def test_missing_model_exits_2(self, tmp_path):
        code = main(["--model", str(tmp_path / "nope.keras"),
                     "--output-dir", str(tmp_path / "out")])
        assert code == 2
