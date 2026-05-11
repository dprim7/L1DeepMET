"""Bit-accurate validation of an hls4ml model against its source Keras model.

Compares predictions on the test split and computes the full physics card
(resolution, response, AUC) for both. Also generates the standard L1METML-style
diagnostic plots: pairplots of (gen, PUPPI, Keras, hls4ml) MET / MET_x / MET_y,
response distributions, and per-layer numerical traces.

All outputs land under the same ``output_dir`` as the conversion.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import h5py  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Numbers + paths to written plots.

    Attributes:
        keras_card:   physics-card dict computed on Keras predictions.
        hls_card:     physics-card dict computed on hls4ml predictions.
        delta_card:   per-key (hls − keras); resolution deltas tell us how
                      much fixed-point quantization costs in physics terms.
        n_events:     number of test events used.
        plots:        dict of plot-name → absolute path.
    """

    keras_card: Dict[str, float]
    hls_card: Dict[str, float]
    delta_card: Dict[str, float]
    n_events: int
    plots: Dict[str, str] = field(default_factory=dict)


def _model_inputs_from_features(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Split the (N, 128, 9) preprocessed feature tensor into the dict the
    Keras model expects.

    Note: pdgid / charge are categorical, but we pass them as float32 because
    hls4ml's compiled bit-accurate library expects float32 for all inputs
    (it casts internally to whatever each input layer's declared precision is).
    Passing actual int32 arrays raises ``TypeError: array must have data type
    float32`` from the underlying ctypes call.
    """
    return {
        "continuous_inputs": X[:, :, 0:5].astype(np.float32),
        "momentum_inputs":   X[:, :, 5:7].astype(np.float32),
        "pdgid_inputs":      X[:, :, 7].astype(np.float32),
        "charge_inputs":     X[:, :, 8].astype(np.float32),
    }


def compare_keras_vs_hls(
    keras_model: tf.keras.Model,
    hls_model: Any,
    test_h5_path: str | Path,
    output_dir: str | Path,
    *,
    normfac: float = 100.0,
    n_events: int = 1000,
    make_plots: bool = True,
    trace_layers: bool = False,
) -> ValidationResult:
    """Predict on the test set with both models, compute physics card, plot.

    Args:
        keras_model:    the source full-precision Keras model.
        hls_model:      the compiled hls4ml model (call ``.predict(X)``).
        test_h5_path:   path to ``test.h5`` (preprocessed). Contains ``features``
                        and ``targets`` datasets.
        output_dir:     where to write plots + ``validation.json``.
        normfac:        training-time target normalization (predictions are in
                        normalized units; multiply by ``normfac`` to get GeV).
        n_events:       subsample the test set to this many events for speed.
                        hls_model.predict is the bottleneck (Python-bound sim).
        make_plots:     produce diagnostic plots.
        trace_layers:   also dump scatter plots of every intermediate layer
                        (Keras vs hls4ml). Expensive — pass ``False`` for quick
                        iteration.

    Returns:
        :class:`ValidationResult`.
    """
    from l1deepmet.metrics.physics import compute_puppi_baseline, full_physics_card

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(test_h5_path), "r") as f:
        X = f["features"][:n_events]
        Y = f["targets"][:n_events]

    logger.info("Validating on %d events from %s", len(X), test_h5_path)
    mi = _model_inputs_from_features(X)

    keras_pred = keras_model.predict(mi, batch_size=512, verbose=0) * normfac
    # hls4ml's .predict expects the inputs in the input-layer order, all as
    # float32. The C library casts internally per the declared precision.
    hls_pred = hls_model.predict(
        [mi[layer.name].astype(np.float32) for layer in keras_model.inputs]
    ) * normfac
    # Sanity: shapes match
    assert keras_pred.shape == hls_pred.shape, \
        f"Pred shape mismatch: keras {keras_pred.shape} vs hls {hls_pred.shape}"

    puppi_xy = compute_puppi_baseline(X)
    keras_card = full_physics_card(gen_xy=Y, reco_xy=keras_pred, puppi_xy=puppi_xy)
    hls_card = full_physics_card(gen_xy=Y, reco_xy=hls_pred, puppi_xy=puppi_xy)

    delta = {k: (hls_card[k] - keras_card[k])
             for k in keras_card if isinstance(keras_card[k], (int, float))
             and not isinstance(keras_card[k], bool) and k in hls_card}

    # Write JSON summary
    with open(output_dir / "validation.json", "w") as f:
        json.dump(
            {
                "n_events": len(X),
                "keras_card": keras_card,
                "hls_card": hls_card,
                "delta_card": delta,
            },
            f, indent=2, default=str,
        )

    plots: Dict[str, str] = {}
    if make_plots:
        plots = _make_plots(
            gen_xy=Y, puppi_xy=puppi_xy,
            keras_pred=keras_pred, hls_pred=hls_pred,
            output_dir=output_dir,
        )

    if trace_layers:
        plots.update(_trace_layers(keras_model, hls_model, mi, output_dir))

    logger.info("Validation complete: %s", output_dir / "validation.json")
    return ValidationResult(
        keras_card=keras_card,
        hls_card=hls_card,
        delta_card=delta,
        n_events=len(X),
        plots=plots,
    )


def _make_plots(
    gen_xy: np.ndarray,
    puppi_xy: np.ndarray,
    keras_pred: np.ndarray,
    hls_pred: np.ndarray,
    output_dir: Path,
) -> Dict[str, str]:
    """Three pairplots (MET, MET_x, MET_y) + one response histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn  # type: ignore
    import pandas as pd  # type: ignore

    plots: Dict[str, str] = {}

    gen_pt = np.hypot(gen_xy[:, 0], gen_xy[:, 1])
    puppi_pt = np.hypot(puppi_xy[:, 0], puppi_xy[:, 1])
    keras_pt = np.hypot(keras_pred[:, 0], keras_pred[:, 1])
    hls_pt = np.hypot(hls_pred[:, 0], hls_pred[:, 1])

    # ── pT pairplot ──
    df = pd.DataFrame({
        "Gen MET":    gen_pt,
        "PUPPI MET":  puppi_pt,
        "Keras MET":  keras_pt,
        "hls4ml MET": hls_pt,
    })
    plt.figure()
    seaborn.pairplot(df, corner=True, height=2.5)
    p = output_dir / "profiling_MET.png"
    plt.savefig(p, dpi=150)
    plt.close("all")
    plots["pt_pairplot"] = str(p)

    # ── X pairplot ──
    df = pd.DataFrame({
        "Gen MET x":    gen_xy[:, 0],
        "PUPPI MET x":  puppi_xy[:, 0],
        "Keras MET x":  keras_pred[:, 0],
        "hls4ml MET x": hls_pred[:, 0],
    })
    plt.figure()
    seaborn.pairplot(df, corner=True, height=2.5)
    p = output_dir / "profiling_MET_x.png"
    plt.savefig(p, dpi=150)
    plt.close("all")
    plots["x_pairplot"] = str(p)

    # ── Y pairplot ──
    df = pd.DataFrame({
        "Gen MET y":    gen_xy[:, 1],
        "PUPPI MET y":  puppi_xy[:, 1],
        "Keras MET y":  keras_pred[:, 1],
        "hls4ml MET y": hls_pred[:, 1],
    })
    plt.figure()
    seaborn.pairplot(df, corner=True, height=2.5)
    p = output_dir / "profiling_MET_y.png"
    plt.savefig(p, dpi=150)
    plt.close("all")
    plots["y_pairplot"] = str(p)

    # ── Response distribution (reco / gen) ──
    bins = np.linspace(0, 2, 25)
    plt.figure(figsize=(15, 4))
    for i, (name, pt) in enumerate(
        [("PUPPI", puppi_pt), ("Keras", keras_pt), ("hls4ml", hls_pt)], start=1
    ):
        valid = gen_pt > 1  # avoid divide-by-zero
        r = pt[valid] / gen_pt[valid]
        med = float(np.median(r))
        iqr = float(np.percentile(r, 75) - np.percentile(r, 25))
        plt.subplot(1, 3, i)
        plt.hist(r, bins=bins, label=f"{name}, median={med:.2f}, IQR={iqr:.2f}")
        plt.legend()
        plt.xlabel(r"MET response $\hat{y}/y$")
        plt.ylabel("Events")
    plt.tight_layout()
    p = output_dir / "response_MET.png"
    plt.savefig(p, dpi=150)
    plt.close("all")
    plots["response"] = str(p)

    return plots


def _trace_layers(
    keras_model: tf.keras.Model,
    hls_model: Any,
    mi: Dict[str, np.ndarray],
    output_dir: Path,
) -> Dict[str, str]:
    """Per-layer scatter plots (hls4ml vs Keras) for finding quantization
    pathologies. Wraps hls4ml's profiling helpers; one PNG per layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import hls4ml  # type: ignore

    plots: Dict[str, str] = {}
    # hls4ml's profiling expects list inputs in the input layer order.
    X_list = [mi[layer.name].astype(np.float32) for layer in keras_model.inputs]

    try:
        _, hls_trace = hls_model.trace(X_list)
        keras_trace = hls4ml.model.profiling.get_ymodel_keras(keras_model, X_list)
    except Exception as e:
        logger.warning("Layer-by-layer tracing unavailable: %s", e)
        return plots

    for layer_name, hls_vals in hls_trace.items():
        if layer_name not in keras_trace:
            continue
        keras_vals = keras_trace[layer_name]
        plt.figure(figsize=(5, 5))
        plt.scatter(hls_vals.flatten(), keras_vals.flatten(), s=0.2)
        mn = float(min(np.min(hls_vals), np.min(keras_vals)))
        mx = float(max(np.max(hls_vals), np.max(keras_vals)))
        plt.plot([mn, mx], [mn, mx], color="gray", linewidth=0.8)
        plt.xlabel(f"hls4ml {layer_name}")
        plt.ylabel(f"Keras  {layer_name}")
        plt.tight_layout()
        p = output_dir / f"trace_{layer_name}.png"
        plt.savefig(p, dpi=120)
        plt.close("all")
        plots[f"trace_{layer_name}"] = str(p)
    return plots
