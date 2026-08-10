"""EBOPs (effective bit operations) reporting for HGQ2 models.

EBOPs ~ sum of ``bits_in x bits_kernel`` over each layer's multiply
structure — HGQ's differentiable proxy for FPGA LUT+DSP cost. Each
Q-layer stores its value in a non-trainable ``uint32`` Keras weight that

- is populated during calls with ``training`` truthy or ``'tracing'``
  (i.e. during ``fit`` or ``hgq.utils.trace_minmax``), and
- is serialized into ``.keras`` checkpoints, so a *saved trained* model
  reports its EBOPs with no forward pass (``collect_ebops``), while a
  freshly *built* model reads all-zero until traced
  (``recompute_ebops``).

``hgq`` is imported lazily so ``l1deepmet.estimation`` stays importable
without it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class EbopsRow:
    layer: str
    class_name: str
    ebops: int
    beta: Optional[float]


@dataclass
class EbopsReport:
    per_layer: List[EbopsRow] = field(default_factory=list)
    total: int = 0
    n_q_layers: int = 0
    all_zero: bool = False
    source: str = "saved"  # "saved" | "recomputed"

    def to_dict(self) -> dict:
        return {
            "per_layer": [asdict(row) for row in self.per_layer],
            "total": int(self.total),
            "n_q_layers": int(self.n_q_layers),
            "all_zero": bool(self.all_zero),
            "source": self.source,
        }


def collect_ebops(model, *, source: str = "saved") -> EbopsReport:
    """Read the stored per-layer EBOPs without running the model.

    Mirrors ``hgq.utils.sugar.FreeEBOPs``: walks ``_flatten_layers()`` and
    sums ``_ebops`` over Q-layers with ``enable_ebops``. Returns an empty
    report (``n_q_layers == 0``) for plain Keras models — and likewise
    when ``hgq`` isn't installed, in which case the model cannot contain
    Q-layers anyway.
    """
    report = EbopsReport(source=source)
    try:
        from hgq.layers import QLayerBase
    except ImportError:
        return report

    from keras import ops

    for layer in model._flatten_layers():
        if not (isinstance(layer, QLayerBase) and layer.enable_ebops):
            continue
        ebops = int(ops.convert_to_numpy(layer._ebops))
        try:
            beta = float(ops.convert_to_numpy(layer.beta))
        except (AttributeError, TypeError, ValueError):
            beta = None
        report.per_layer.append(EbopsRow(
            layer=layer.name,
            class_name=type(layer).__name__,
            ebops=ebops,
            beta=beta,
        ))

    report.n_q_layers = len(report.per_layer)
    report.total = sum(row.ebops for row in report.per_layer)
    report.all_zero = report.n_q_layers > 0 and report.total == 0
    return report


def recompute_ebops(model, data, *, batch_size: int = 1024,
                    reset: bool = True) -> EbopsReport:
    """(Re)populate EBOPs by tracing ``model`` over calibration ``data``.

    Wraps ``hgq.utils.trace_minmax``, which calls the model with the
    special ``'tracing'`` training flag: EBOPs and quantizer min/max
    registers are recomputed, while BatchNorm uses (and does not update)
    its moving statistics.

    .. warning::
       This MUTATES model state: the ``_ebops`` weights always, and — for
       WRAP-mode activation quantizers with ``reset=True`` — the min/max
       registers. Load a throwaway instance if that matters.

    Args:
        data: calibration inputs — an array, or a list of arrays matching
            ``model.inputs`` order (see ``synthetic_calibration_batch``).
        batch_size: tracing batch size.
        reset: reset WRAP-mode min/max registers before tracing.
    """
    from hgq.utils import trace_minmax

    trace_minmax(model, data, reset=reset, batch_size=batch_size)
    return collect_ebops(model, source="recomputed")


def synthetic_calibration_batch(model, n: int = 32, seed: int = 42) -> List[np.ndarray]:
    """Random calibration inputs matching ``model.inputs`` (order and shape).

    Inputs whose name contains ``pdgid`` / ``charge`` get categorical
    values: one-hot rows when the input carries a vocab axis (rank >= 3,
    the HGQ2 convention), integer codes otherwise (the fp32 convention,
    vocab 6 for pdgid / 4 for charge). Everything else is standard normal.

    Synthetic data exercises every path for plumbing purposes; use real
    PUPPI candidates for physics-meaningful EBOPs (WRAP-mode ranges and
    activation statistics depend on the input distribution).
    """
    rng = np.random.default_rng(seed)
    batch: List[np.ndarray] = []
    for tensor in model.inputs:
        shape = [n] + [int(d) for d in tensor.shape[1:]]
        name = (getattr(tensor, "name", "") or "").lower()
        is_categorical = "pdgid" in name or "charge" in name
        if is_categorical and len(shape) >= 3:
            vocab = shape[-1]
            idx = rng.integers(0, vocab, size=shape[:-1])
            batch.append(np.eye(vocab, dtype=np.float32)[idx])
        elif is_categorical:
            vocab = 6 if "pdgid" in name else 4
            batch.append(rng.integers(0, vocab, size=shape).astype(np.float32))
        else:
            batch.append(rng.normal(0.0, 1.0, size=shape).astype(np.float32))
    return batch
