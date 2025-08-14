from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import awkward as ak  # type: ignore
import numpy as np  # type: ignore
import argparse
import sys
from pathlib import Path

import h5py  # type: ignore


def to_np_array(ak_array, maxN: int = 100, pad: float | int = 0):
    """
    Convert an Awkward array of variable-length lists to a fixed-size numpy array
    by padding to length `maxN` (clipping longer lists). Missing entries are filled
    with `pad`.
    """
    return ak.fill_none(ak.pad_none(ak_array, maxN, clip=True, axis=-1), pad).to_numpy()


def convertXY2PtPhi(arrayXY: np.ndarray) -> np.ndarray:
    """Convert [:,0]=x, [:,1]=y to [:,0]=pt and [:,1]=phi."""
    nevents = arrayXY.shape[0]
    arrayPtPhi = np.zeros((nevents, 2), dtype=arrayXY.dtype)
    arrayPtPhi[:, 0] = np.sqrt((arrayXY[:, 0] ** 2 + arrayXY[:, 1] ** 2))
    arrayPtPhi[:, 1] = np.arctan2(arrayXY[:, 1], arrayXY[:, 0])
    return arrayPtPhi


def compute_px_py(pt: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute px, py from pt and phi with broadcasting."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    return px, py


def HCalDepth(
    hcal_first1: np.ndarray, hcal_first3: np.ndarray, hcal_first5: np.ndarray
) -> np.ndarray:
    """Effective center of energy depth in the hadronic calorimeter of Phase-2 HGCal.

    Mirrors convertNanoToHDF5.HCalDepth with numerically safe handling of zeros.
    """
    epsilon = 1.0e-10
    hcal_first5_safe = np.where(hcal_first5 == 0, epsilon, hcal_first5)
    depth_weighted = (
        hcal_first1 * 1.0
        + (hcal_first3 - hcal_first1) * 3.0
        + (hcal_first5 - hcal_first3) * 5.0
    ) / hcal_first5_safe
    depth_weighted = np.where(hcal_first5 == 0, 0.0, depth_weighted)
    return depth_weighted


def encode_pdgId_charge(pdgid: np.ndarray, charge: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode pdgId and charge using the mapping used during HDF5 conversion."""
    d_encoding = {
        "L1PuppiCands_charge": {-999.0: 0, -1.0: 1, 0.0: 2, 1.0: 3},
        "L1PuppiCands_pdgId": {
            -999.0: 0,
            -211.0: 1,
            -130.0: 2,
            -22.0: 3,
            -13.0: 4,
            -11.0: 5,
            11.0: 5,
            13.0: 4,
            22.0: 3,
            130.0: 2,
            211.0: 1,
        },
    }
    enc_pdgid = np.vectorize(d_encoding["L1PuppiCands_pdgId"].__getitem__)(pdgid.astype(float))
    enc_charge = np.vectorize(d_encoding["L1PuppiCands_charge"].__getitem__)(charge.astype(float))
    return enc_pdgid, enc_charge



def normalize_targets(Y: np.ndarray, normFac: float, invert_sign: bool = False) -> np.ndarray:
    """Normalize regression targets by normFac. If invert_sign is True, divide by -normFac."""
    denom = -normFac if invert_sign else normFac
    return Y / denom


def sort_and_truncate_by_pt(X: np.ndarray, maxNPF: int) -> np.ndarray:
    """Sort PF candidates in each event by pt descending (X[:,:,0]) and keep first maxNPF."""
    if maxNPF >= X.shape[1]:
        return X
    order = X[:, :, 0].argsort(axis=1)[:, ::-1]
    X_sorted = X.copy()
    shape = np.shape(X_sorted)
    for i in range(shape[0]):
        X_sorted[i, :, :] = X_sorted[i, order[i], :]
    return X_sorted[:, 0:maxNPF, :]


def deltaR_calc(eta1: np.ndarray, phi1: np.ndarray, eta2: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """Calculate deltaR with phi wrapping to [-pi, pi]."""
    dphi = (phi1 - phi2).copy()
    gt_pi_idx = (dphi > np.pi)
    lt_mpi_idx = (dphi < -np.pi)
    dphi[gt_pi_idx] -= 2 * np.pi
    dphi[lt_mpi_idx] += 2 * np.pi
    deta = eta1 - eta2
    return np.hypot(deta, dphi)


def kT_calc(pti: np.ndarray, ptj: np.ndarray, dR: np.ndarray) -> np.ndarray:
    """Compute kT = min(pt_i, pt_j) * dR."""
    min_pt = np.minimum(pti, ptj)
    return min_pt * dR


def z_calc(pti: np.ndarray, ptj: np.ndarray) -> np.ndarray:
    """Compute z = min(pt_i, pt_j) / (pt_i + pt_j + eps)."""
    epsilon = 1.0e-12
    min_pt = np.minimum(pti, ptj)
    return min_pt / (pti + ptj + epsilon)


def mass2_calc(pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Invariant mass squared of two 4-vectors (E, px, py, pz)."""
    pij = pi + pj
    return pij[:, :, 0] ** 2 - pij[:, :, 1] ** 2 - pij[:, :, 2] ** 2 - pij[:, :, 3] ** 2


def build_edge_features(
    Xi: np.ndarray,
    Xp: np.ndarray,
    edge_list: Sequence[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build graph edge features for an event batch.

    Inputs:
    - Xi: continuous features with pt at [:,:,0], eta at [:,:,1], phi at [:,:,2]
    - Xp: momentum components with [:,:,0]=px, [:,:,1]=py
    - edge_list: subset of {"dR", "kT", "z", "m2"}

    Returns (ef, edge_idx):
    - ef:  (batch, N*(N-1), num_edge_features)
    - edge_idx: (N*(N-1), 2) array of (receiver, sender) pairs
    """
    if edge_list is None:
        edge_list = []

    N = Xi.shape[1]
    receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0] != i[1]]
    edge_idx = np.array(receiver_sender_list)

    eta = Xi[:, :, 1]
    phi = Xi[:, :, 2]
    pt = Xi[:, :, 0]

    edge_stack: List[np.ndarray] = []
    dR = None

    if ("dR" in edge_list) or ("kT" in edge_list):
        eta1 = eta[:, edge_idx[:, 0]]
        phi1 = phi[:, edge_idx[:, 0]]
        eta2 = eta[:, edge_idx[:, 1]]
        phi2 = phi[:, edge_idx[:, 1]]
        dR = deltaR_calc(eta1, phi1, eta2, phi2)
        if "dR" in edge_list:
            edge_stack.append(dR)

    if ("kT" in edge_list) or ("z" in edge_list):
        pt1 = pt[:, edge_idx[:, 0]]
        pt2 = pt[:, edge_idx[:, 1]]
        if "kT" in edge_list:
            if dR is None:
                # If dR not computed above (edge_list contains kT but not dR)
                eta1 = eta[:, edge_idx[:, 0]]
                phi1 = phi[:, edge_idx[:, 0]]
                eta2 = eta[:, edge_idx[:, 1]]
                phi2 = phi[:, edge_idx[:, 1]]
                dR = deltaR_calc(eta1, phi1, eta2, phi2)
            edge_stack.append(kT_calc(pt1, pt2, dR))
        if "z" in edge_list:
            edge_stack.append(z_calc(pt1, pt2))

    if "m2" in edge_list:
        # Reconstruct 4-vectors from (px, py, pt, eta)
        px = Xp[:, :, 0]
        py = Xp[:, :, 1]
        pz = pt * np.sinh(eta)
        energy = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
        p4 = np.stack((energy, px, py, pz), axis=-1)
        p1 = p4[:, edge_idx[:, 0], :]
        p2 = p4[:, edge_idx[:, 1], :]
        edge_stack.append(mass2_calc(p1, p2))

    if len(edge_stack) == 0:
        ef = np.empty((Xi.shape[0], edge_idx.shape[0], 0), dtype=Xi.dtype)
    else:
        ef = np.stack(edge_stack, axis=-1)

    return ef, edge_idx


def preprocess_batch(
    X: np.ndarray,
    normFac: float,
    maxNPF: int | None = None,
    edge_list: Sequence[str] | None = None,
) -> Tuple[List[np.ndarray], np.ndarray, dict]:
    """
    High-level preprocessing pipeline for a batch of events.

    - Optionally sort/truncate PF candidates by pt
    - Apply preProcessing to build inputs and categorical features
    - Optionally build edge features

    Returns: (Xr, Yr_placeholder, metadata)
    - Xr: list of model inputs [Xi, Xp, Xc0, Xc1[, ef]]
    - Yr_placeholder: zeros target (shape matches two-component regression)
    - metadata: dict with auxiliary info (e.g., emb_input_dim)
    """
    X_work = X
    if maxNPF is not None:
        X_work = sort_and_truncate_by_pt(X_work, maxNPF)

    Xi, Xp, Xc0, Xc1 = preProcessing(X_work, normFac)

    inputs: List[np.ndarray] = [Xi, Xp, Xc0, Xc1]
    metadata = {"emb_input_dim": {0: int(np.max(Xc0[0:1000])) + 1, 1: int(np.max(Xc1[0:1000])) + 1}}

    if edge_list:
        ef, edge_idx = build_edge_features(Xi, Xp, edge_list)
        inputs = [Xi, Xp, Xc0, Xc1, ef]
        metadata["edge_idx"] = edge_idx

    # Placeholder for targets in cases where only inputs are needed here
    Yr_placeholder = np.zeros((X_work.shape[0], 2), dtype=X_work.dtype)
    return inputs, Yr_placeholder, metadata



def _find_h5_files(input_path: Path, pattern: str = "*.h5") -> List[Path]:
    if input_path.is_file() and input_path.suffix == ".h5":
        return [input_path]
    if input_path.is_dir():
        return sorted(list(input_path.glob(pattern)))
    raise ValueError(f"Input path does not exist or is not a file/dir: {input_path}")


def _find_root_files(input_path: Path, pattern: str = "*.root") -> List[Path]:
    if input_path.is_file() and input_path.suffix == ".root":
        return [input_path]
    if input_path.is_dir():
        return sorted(list(input_path.glob(pattern)))
    return []


def _convert_root_to_h5(root_files: List[Path], output_dir: Path, maxevents: int = -1, is_data: bool = False) -> List[Path]:
    # Lazy import to avoid circulars and optional dependency issues
    from convertNanoToHDF5 import convert_single_file

    h5_dir = output_dir / "converted_h5"
    h5_dir.mkdir(parents=True, exist_ok=True)
    out_h5: List[Path] = []
    for rf in root_files:
        of = h5_dir / f"{rf.stem}.h5"
        convert_single_file(str(rf), str(of), maxevents, is_data)
        out_h5.append(of)
    return out_h5


def _save_npz(target: Path, arrays: dict):
    np.savez_compressed(str(target), **arrays)


def _save_h5(target: Path, arrays: dict):
    with h5py.File(str(target), "w") as h5f:
        for k, v in arrays.items():
            if isinstance(v, (np.ndarray, np.int64, np.int32, np.float32, np.float64)):
                h5f.create_dataset(k, data=v, compression="lzf")
            else:
                # store simple metadata dicts as attributes when possible
                try:
                    if isinstance(v, dict):
                        for ak_, av_ in v.items():
                            h5f.attrs[f"{k}.{ak_}"] = av_
                except Exception:
                    pass


def _parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Run the L1METML preprocessing chain")
    p.add_argument("-i", "--input", required=True, help="Input file or directory (.root or .h5)")
    p.add_argument("-o", "--output", required=True, help="Output directory for preprocessed files")
    p.add_argument("--pattern", default="*.h5", help="Pattern for input files if a directory is given")
    p.add_argument("--root-pattern", default="*.root", help="Pattern for ROOT files if converting")
    p.add_argument("--convert-root", action="store_true", help="If set and ROOT files are provided, convert to H5 first")
    p.add_argument("--data", action="store_true", help="ROOT input is data (no truth targets)")
    p.add_argument("--maxevents", type=int, default=-1, help="Max events per ROOT file when converting (-1 for all)")
    p.add_argument("--maxNPF", type=int, default=100, help="Max PF candidates per event after sorting")
    p.add_argument("--normFac", type=float, default=1.0, help="Normalization factor for inputs/targets")
    p.add_argument("--invert-target-sign", action="store_true", help="Divide targets by -normFac instead of normFac")
    p.add_argument(
        "--edge-features",
        nargs="*",
        default=[],
        choices=["dR", "kT", "z", "m2"],
        help="Edge features to compute",
    )
    p.add_argument("--format", choices=["h5", "npz"], default="h5", help="Output file format")
    p.add_argument("--suffix", default="_preproc", help="Suffix to append to output filenames")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = _parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input files
    h5_files: List[Path] = []
    if input_path.suffix == ".root" or (input_path.is_dir() and len(_find_root_files(input_path, args.root_pattern)) > 0):
        if not args.convert_root:
            print("No H5 files found and --convert-root not set. Nothing to do.")
            return 1
        root_files = _find_root_files(input_path, args.root_pattern)
        if len(root_files) == 0:
            print("No ROOT files found to convert.")
            return 1
        print(f"Converting {len(root_files)} ROOT files to H5...")
        h5_files = _convert_root_to_h5(root_files, output_dir, args.maxevents, args.data)
    else:
        h5_files = _find_h5_files(input_path, args.pattern)

    if len(h5_files) == 0:
        print("No H5 files to preprocess.")
        return 1

    print(f"Preprocessing {len(h5_files)} H5 file(s)...")

    for in_h5 in h5_files:
        with h5py.File(str(in_h5), "r") as h5f:
            X = h5f["X"][:]
            Y = h5f["Y"][:] if "Y" in h5f else np.zeros((X.shape[0], 2), dtype=X.dtype)

        # Sort/truncate, preprocess, normalize
        X_sorted = sort_and_truncate_by_pt(X, args.maxNPF)
        Xi, Xp, Xc0, Xc1 = preProcessing(X_sorted, args.normFac)
        Yr = normalize_targets(Y, args.normFac, invert_sign=args.invert_target_sign)

        arrays = {
            "Xi": Xi,
            "Xp": Xp,
            "Xc0": Xc0,
            "Xc1": Xc1,
            "Y": Yr,
        }

        # Edge features if requested
        if len(args.edge_features) > 0:
            ef, edge_idx = build_edge_features(Xi, Xp, args.edge_features)
            arrays["ef"] = ef
            arrays["edge_idx"] = edge_idx

        # Embedding dims metadata
        emb_input_dim = {0: int(np.max(Xc0[0:1000])) + 1, 1: int(np.max(Xc1[0:1000])) + 1}
        arrays["emb_input_dim"] = emb_input_dim

        out_name = output_dir / f"{in_h5.stem}{args.suffix}.{args.format}"
        if args.format == "npz":
            _save_npz(out_name, arrays)
        else:
            _save_h5(out_name, arrays)

        print(f"Saved {out_name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

