"""Integration test for the QAT validation experiment driver.

Runs the full pipeline (FP32 train → HGQ2 train → HLS conversion ×2 → report)
end-to-end at a tiny scale: 1 epoch each, 50 validation events. Slow (a few
minutes) so marked ``slow``; ``-m 'not slow'`` skips it during normal CI.

Verifies the orchestration produces the expected artifact tree, not the
physics performance.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("hgq")
pytest.importorskip("hls4ml")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path("/home/users/dprimosc/L1DeepMET/preprocessed/25Jul8_140X_v0")


@pytest.mark.slow
@pytest.mark.skipif(
    not DATA_DIR.is_dir(),
    reason=f"Real preprocessed data not present at {DATA_DIR}",
)
def test_full_pipeline_produces_expected_artifacts(tmp_path: Path):
    """Smoke-runs the driver and checks the artifact tree."""
    out = tmp_path / "qat_validation_smoke"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "experiments" / "qat_validation.py"),
        "--output-dir", str(out),
        "--data-dir", str(DATA_DIR),
        "--epochs", "1",
        "--seed", "42",
        "--n-validation-events", "50",
    ]
    # Use a minimal env that doesn't blow up TF threading on shared systems.
    env = {**os.environ,
           "TF_CPP_MIN_LOG_LEVEL": "3",
           "TF_NUM_INTEROP_THREADS": "1",
           "TF_NUM_INTRAOP_THREADS": "1"}
    subprocess.run(cmd, check=True, env=env)

    # FP32 artifacts.
    fp32_run = out / "fp32" / "bounded_mae_only_xy0_seed42"
    assert (fp32_run / "best_model.keras").is_file()
    assert (fp32_run / "result.json").is_file()
    assert (fp32_run / "hls").is_dir()
    assert (fp32_run / "hls" / "validation.json").is_file()
    assert (fp32_run / "hls" / "firmware").is_dir()

    # HGQ2 artifacts.
    hgq2_run = out / "hgq2"
    assert (hgq2_run / "best_model.keras").is_file()
    assert (hgq2_run / "result.json").is_file()
    assert (hgq2_run / "hls").is_dir()
    assert (hgq2_run / "hls" / "validation.json").is_file()
    assert (hgq2_run / "hls" / "firmware").is_dir()

    # Comparison + report.
    assert (out / "comparison.json").is_file()
    comp = json.loads((out / "comparison.json").read_text())
    assert "fp32" in comp and "hgq2" in comp
    assert "keras_card" in comp["fp32"] and "hls_card" in comp["fp32"]
    assert "keras_card" in comp["hgq2"] and "hls_card" in comp["hgq2"]

    report = out / "report.md"
    assert report.is_file()
    contents = report.read_text()
    assert "FP32 Keras" in contents and "HGQ2 Keras" in contents
