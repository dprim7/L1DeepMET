"""Data-loader shim that adapts the standard H5 preprocessing pipeline for
HGQ2 training.

The HGQ2 model expects ``pdgid_inputs`` and ``charge_inputs`` as one-hot
encoded tensors (since ``hls4ml`` cannot synthesize ``tf.one_hot`` inside the
model graph). This module wraps the raw integer encodings produced by the
full-precision preprocessing into the format the HGQ2 model wants — and
provides the equivalent of ``H5DataLoader.create_tf_dataset`` for the QAT
trainer.

Public surface:

  - ``encode_split_for_hgq2(X, Y, normfac=1.0)`` — numpy in / numpy out for
    eager inference / evaluation.
  - ``make_hgq2_tf_dataset_from_features(X, Y, batch_size, shuffle, normfac)``
    — TF dataset wrapper used by the training loop.
  - ``make_hgq2_tf_dataset_from_h5(data_dir, split, ...)`` — convenience
    wrapper that loads the preprocessed H5 split for a given data dir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

from .hgq2_model import CHARGE_VOCAB, PDGID_VOCAB, one_hot_encode_features


def encode_split_for_hgq2(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    normfac: float = 1.0,
) -> Tuple[dict, np.ndarray]:
    """Split the preprocessed feature tensor into the dict the HGQ2 model
    consumes, one-hot encoding the categorical columns on the way.

    Args:
        X:        ``(N, 128, 9)`` preprocessed feature tensor. Layout matches
                  the full-precision pipeline (continuous 0:5, momentum 5:7,
                  pdgid 7, charge 8).
        Y:        ``(N, 2)`` MET (px, py) target in GeV.
        normfac:  divisor applied to targets (mirrors ``H5DataLoader.create_tf_dataset``).

    Returns:
        ``(inputs_dict, targets)`` where ``inputs_dict`` has keys
        ``continuous_inputs``, ``momentum_inputs``, ``pdgid_inputs``,
        ``charge_inputs`` and ``targets`` is ``Y`` (or ``Y / normfac``).
    """
    inputs_continuous = X[:, :, 0:5].astype(np.float32)
    inputs_momentum = X[:, :, 5:7].astype(np.float32)
    pdgid_codes = X[:, :, 7]
    charge_codes = X[:, :, 8]
    pdgid_oh, charge_oh = one_hot_encode_features(
        pdgid_codes, charge_codes,
        pdgid_vocab=PDGID_VOCAB, charge_vocab=CHARGE_VOCAB,
    )
    inputs = {
        "continuous_inputs": inputs_continuous,
        "momentum_inputs":   inputs_momentum,
        "pdgid_inputs":      pdgid_oh,
        "charge_inputs":     charge_oh,
    }
    targets = Y.astype(np.float32) if normfac == 1.0 else (Y / float(normfac)).astype(np.float32)
    return inputs, targets


def make_hgq2_tf_dataset_from_features(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
    normfac: float = 1.0,
    shuffle_buffer: int = 10_000,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Build a ``tf.data.Dataset`` of (inputs_dict, targets) pairs.

    Mirrors the threading / prefetch settings used by ``H5DataLoader.create_tf_dataset``:
    private threadpool of size 1 (shared-system safe), prefetch(1),
    num_parallel_calls=1.
    """
    inputs, targets = encode_split_for_hgq2(X, Y, normfac=normfac)
    # ``from_tensor_slices`` on a dict-of-arrays produces a dataset of dicts.
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    ds = ds.batch(batch_size)

    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)
    ds = ds.prefetch(1)
    return ds


def make_hgq2_tf_dataset_from_h5(
    data_dir: str | Path,
    split: str = "train",
    *,
    batch_size: int = 256,
    shuffle: bool | None = None,
    normfac: float = 1.0,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Convenience: load ``data_dir/<split>.h5`` and return a dataset.

    Defaults match the training-time semantics of the standard H5DataLoader:
    shuffle=True only for ``split == 'train'``.
    """
    import h5py  # local import
    path = Path(data_dir) / f"{split}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"No preprocessed split at {path}")
    with h5py.File(str(path), "r") as f:
        X = f["features"][:]
        Y = f["targets"][:]
    if shuffle is None:
        shuffle = (split == "train")
    return make_hgq2_tf_dataset_from_features(
        X, Y,
        batch_size=batch_size, shuffle=shuffle, normfac=normfac,
        seed=seed,
    )
