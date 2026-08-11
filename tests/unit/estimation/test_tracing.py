"""Tests for ``l1deepmet.estimation.tracing`` — da4ml LUT-cost/latency
tracing of HGQ2 models, no HLS run involved.

da4ml's converter registry is exact-class keyed and (as of 0.5.2) has no
handler for plain ``keras.layers.Activation`` — which is exactly what
``build_hgq2_model``'s relu/tanh layers are. ``register_extra_handlers``
adds one: linear passes through, relu uses da4ml's native op, and any
other function goes through ``FixedVariableArray.apply`` so the next
Q-layer's input quantizer materializes the LUT cost. The bounded-weight
model (tanh) is therefore the make-or-break case and is tested first.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

pytest.importorskip("hgq")
pytest.importorskip("da4ml")
try:
    import da4ml.converter.hgq2  # noqa: F401
except ImportError:
    pytest.skip("da4ml hgq2 converter requires hgq2>=0.1.9",
                allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

import keras  # noqa: E402

from l1deepmet.estimation.tracing import (  # noqa: E402
    TraceReport,
    check_traceable,
    register_extra_handlers,
    trace_costs,
)
from l1deepmet.quantization import build_hgq2_model  # noqa: E402

# Small particle axis keeps the symbolic replay fast; the real 128-slot
# model is exercised by the slow integration test.
N_TEST_PARTICLES = 16


def _tiny(**overrides):
    kwargs = dict(width=8, depth=1, use_embeddings=True, bounded_weight=True,
                  n_particles=N_TEST_PARTICLES)
    kwargs.update(overrides)
    return build_hgq2_model(**kwargs)


class TestRegisterExtraHandlers:
    def test_registers_plain_activation(self):
        register_extra_handlers()
        from da4ml.converter.hgq2.layers._base import _registry

        assert keras.layers.Activation in _registry

    def test_idempotent(self):
        register_extra_handlers()
        from da4ml.converter.hgq2.layers._base import _registry

        first = _registry[keras.layers.Activation]
        register_extra_handlers()
        assert _registry[keras.layers.Activation] is first

    def test_concatenate_override_wins_registry_slot(self):
        """da4ml's ReplayMerge broadcasts before concatenating, which
        breaks unequal channel widths — our handler must own the slot."""
        register_extra_handlers()
        from da4ml.converter.hgq2.layers._base import _registry

        handler = _registry[keras.layers.Concatenate]
        assert handler.__module__ == "l1deepmet.estimation.tracing"


class TestTraceCosts:
    def test_bounded_weight_model_traces(self):
        """tanh -> LUT path: the make-or-break case."""
        report = trace_costs(_tiny(bounded_weight=True))
        assert isinstance(report, TraceReport)
        assert report.cost > 0
        assert report.latency_max >= report.latency_min >= 0
        assert report.n_inputs > 0
        assert report.n_outputs == 2

    def test_unbounded_variant_traces(self):
        assert trace_costs(_tiny(bounded_weight=False)).cost > 0

    def test_no_embeddings_variant_traces_with_inputs_kif(self):
        """The zero-kernel dummy path Adds raw inputs before any Q-layer
        quantizer, so the input ranges must be bounded explicitly."""
        report = trace_costs(_tiny(use_embeddings=False),
                             inputs_kif=(1, 6, 10))
        assert report.cost > 0

    def test_no_embeddings_without_inputs_kif_raises_hint(self):
        with pytest.raises(ValueError, match="inputs_kif"):
            trace_costs(_tiny(use_embeddings=False))

    def test_hwconfig_passthrough(self):
        report = trace_costs(_tiny(), adder_size=2)
        assert report.hwconfig == (2, -1, -1.0)

    def test_per_layer_costs(self):
        report = trace_costs(_tiny(), per_layer=True)
        assert report.per_layer is not None
        names = [name for name, _ in report.per_layer]
        assert "qdense_0" in names
        costs = [cost for _, cost in report.per_layer]
        assert all(c >= 0 for c in costs)
        # The output tensor's cumulative cost is the whole model's cost.
        assert max(costs) == pytest.approx(report.cost, rel=1e-6)

    def test_to_dict_json_serializable(self):
        import json

        payload = json.loads(json.dumps(trace_costs(_tiny()).to_dict()))
        assert payload["cost"] > 0
        assert "latency_max" in payload


class TestUntraceableModels:
    @pytest.fixture(scope="class")
    def fp32_model(self):
        from arch_search import ArchConfig, build_model

        cfg = ArchConfig(name="tiny", width=8, depth=1, mode=1,
                         activation="relu", use_embeddings=True,
                         binned_weight=0.0, bounded_weight=True,
                         use_sum=True)
        return build_model(cfg)

    def test_check_traceable_lists_unsupported_layers(self, fp32_model):
        unsupported = check_traceable(fp32_model)
        # fp32 custom layers have no da4ml handlers.
        assert "bounded_weight" in unsupported
        assert "output" in unsupported  # SumOverParticles

    def test_trace_costs_raises_with_full_list(self, fp32_model):
        with pytest.raises(ValueError) as excinfo:
            trace_costs(fp32_model)
        message = str(excinfo.value)
        for name in check_traceable(fp32_model):
            assert name in message


class TestLazyImport:
    def test_package_import_does_not_pull_converter(self):
        """l1deepmet.estimation must stay importable on envs where the
        da4ml/hgq2 pairing is broken — so the converter import has to be
        deferred until trace time."""
        code = (
            "import sys; import l1deepmet.estimation; "
            "assert 'da4ml.converter.hgq2' not in sys.modules, "
            "'converter imported eagerly'"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
