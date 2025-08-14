"""
Data preprocessing functions for L1DeepMET.

This module contains all functions for loading, preprocessing, and preparing
data for training/evaluation.
"""

from typing import Dict, Tuple, List, Optional, Union, Any
from pathlib import Path
import numpy as np # type: ignore
import awkward as ak # type: ignore
import uproot # type: ignore
import h5py # type: ignore
import logging

from l1deepmet.utils import to_np_array

logger = logging.getLogger(__name__)


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


def load_samples_to_numpy(
    data_root: Union[str, Path],
    sample_names: List[str],
    var_list: List[str],
    var_list_mc: Optional[List[str]] = None,
    *,
    max_pf: int,
    encoding: Dict[str, Dict[float, int]],
    include_mc: bool = True,
    step_size: str = "100 MB",
    file_pattern: str = "*.root",
    tree_name: str = "Events",
    dtype=np.float32,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]: 
    """
    Load ROOT samples and convert to numpy arrays.
    
    Returns a tuple of numpy arrays (features, targets) over all samples.
    """
    data_root = Path(data_root)
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    logger.info(f"Starting data loading from {data_root}")
    logger.info(f"Processing {len(sample_names)} samples: {sample_names}")

    for i, sample in enumerate(sample_names, 1):
        logger.info(f"[{i}/{len(sample_names)}] Processing sample: {sample}")
        
        files = sorted((data_root / sample).rglob(file_pattern))
        if not files:
            raise FileNotFoundError(f"No ROOT files found for sample '{sample}' under {data_root / sample}")

        logger.info(f"  Found {len(files)} ROOT files")
        
        branches = list(var_list)
        if include_mc and var_list_mc:
            branches += var_list_mc
        logger.info(f"  Loading {len(branches)} branches from tree '{tree_name}'")
        logger.debug(f"  Branches: {branches}")
        
        X_parts: List[np.ndarray] = []
        Y_parts: List[np.ndarray] = []

        # Specify the tree name by appending to file paths
        files_with_tree = [str(f) + f":{tree_name}" for f in files]
        
        logger.info(f"  Starting iteration over files with step_size={step_size}")
        batch_count = 0
        
        for arrays in uproot.iterate(files_with_tree, expressions=branches, step_size=step_size, library="ak"):
            batch_count += 1
            logger.info(f"  Processing batch {batch_count}...")
            # Per-candidate inputs (awkward jagged -> dense with padding)
            pt = to_np_array(arrays["L1PuppiCands_pt"], maxN=max_pf, pad=0.0)
            eta = to_np_array(arrays["L1PuppiCands_eta"], maxN=max_pf, pad=0.0)
            phi = to_np_array(arrays["L1PuppiCands_phi"], maxN=max_pf, pad=0.0)
            pdgid = to_np_array(arrays["L1PuppiCands_pdgId"], maxN=max_pf, pad=-999.0)
            charge = to_np_array(arrays["L1PuppiCands_charge"], maxN=max_pf, pad=-999.0)
            puppiw = to_np_array(arrays["L1PuppiCands_puppiWeight"], maxN=max_pf, pad=0.0)
            dxyErr = to_np_array(arrays["L1PuppiCands_dxyErr"], maxN=max_pf, pad=1000.0)

            h1 = to_np_array(arrays["HGCal3DCl_firstHcal1layers"], maxN=max_pf, pad=0.0)
            h3 = to_np_array(arrays["HGCal3DCl_firstHcal3layers"], maxN=max_pf, pad=0.0)
            h5 = to_np_array(arrays["HGCal3DCl_firstHcal5layers"], maxN=max_pf, pad=0.0)
            hcalDepth = HCalDepth(h1, h3, h5)

            px = pt * np.cos(phi)
            py = pt * np.sin(phi)

            enc_pdg = np.vectorize(encoding["L1PuppiCands_pdgId"].__getitem__)(pdgid.astype(float))
            enc_chg = np.vectorize(encoding["L1PuppiCands_charge"].__getitem__)(charge.astype(float))

            nevents = pt.shape[0]
            X = np.zeros((nevents, max_pf, 10), dtype=dtype, order="F")
            X[:, :, 0] = pt
            X[:, :, 1] = px
            X[:, :, 2] = py
            X[:, :, 3] = eta
            X[:, :, 4] = phi
            X[:, :, 5] = puppiw
            X[:, :, 6] = enc_pdg
            X[:, :, 7] = enc_chg
            X[:, :, 8] = hcalDepth

            if include_mc and var_list_mc:
                gen_pt = arrays["genMet_pt"].to_numpy()
                gen_phi = arrays["genMet_phi"].to_numpy()
                Y = np.stack([gen_pt * np.cos(gen_phi), gen_pt * np.sin(gen_phi)], axis=1).astype(dtype, copy=False)
            else:
                Y = np.zeros((nevents, 2), dtype=dtype)

            X_parts.append(X)
            Y_parts.append(Y)
            logger.info(f"Batch {batch_count} complete: {nevents} events processed")
        
        # TODO: don't hardcode the number of features
        features = np.concatenate(X_parts, axis=0) if X_parts else np.zeros((0, max_pf, 9), dtype=dtype)
        targets = np.concatenate(Y_parts, axis=0) if Y_parts else np.zeros((0, 2), dtype=dtype)
        results[sample] = (features, targets)
        
        logger.info(f"Sample '{sample}' complete: {features.shape[0]} total events, shape {features.shape}")

    logger.info(f"Data loading complete. Processed {len(results)} samples")
    return results


def select_events(results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                  samples: Dict[str, int]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Select desired number of events per sample.
    
    Args:
        results: Dict with sample_name -> (features, targets) 
        samples: Dict with sample_name -> n_desired_events
        
    Returns:
        selected_results: Dict with sample_name -> (selected_features, selected_targets)
    """
    logger.info("Selecting desired number of events per sample")

    selected_results = {}
    for sample_name, (features, targets) in results.items():
        n_desired = samples[sample_name]
        n_available = features.shape[0]
        
        if n_available >= n_desired:
            # Randomly select n_desired events
            indices = np.random.choice(n_available, size=n_desired, replace=False)
            selected_features = features[indices]
            selected_targets = targets[indices]
            logger.info(f"{sample_name}: Selected {n_desired} events from {n_available} available")
        else:
            # Use all available events if we don't have enough
            selected_features = features
            selected_targets = targets
            logger.warning(f"{sample_name}: Only {n_available} events available, requested {n_desired}")
        
        selected_results[sample_name] = (selected_features, selected_targets)
        print(f"{sample_name}")
        print(selected_features.shape)
    return selected_results


def preprocess_data(selected_results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply preprocessing steps to the selected data, based on preprocessing.py::preProcessing
    but without normalization factor. Returns single concatenated feature array.
    
    Steps:
    - Extract pt, px, py, eta, phi, puppi, dxyErr, hcalDepth
    - Remove outliers in pt/px/py and sanitize dxyErr/hcalDepth
    - Concatenate into single feature array for H5 storage
    
    Feature array layout (9 features):
    [:, :, 0] = pt, [:, :, 1] = eta, [:, :, 2] = phi, [:, :, 3] = puppi,
    [:, :, 4] = hcal_depth, [:, :, 5] = px, [:, :, 6] = py,
    [:, :, 7] = encoded_pdgId, [:, :, 8] = encoded_charge
    
    Returns:
    - processed_results: dict with sample_name -> (preprocessed_features, targets)
    """
    logger.info("Applying preprocessing steps to selected data")
    
    processed_results = {}
    
    for sample_name, (features, targets) in selected_results.items():
        logger.info(f"Preprocessing sample: {sample_name}")
        
        A = features  
        
        pt = A[:, :, 0:1]          
        px = A[:, :, 1:2]          
        py = A[:, :, 2:3]          
        eta = A[:, :, 3:4]         
        phi = A[:, :, 4:5]         
        puppi = A[:, :, 5:6]       
        dxyErr = A[:, :, 8:9]      
        hcalDepth = A[:, :, 9:10]  
        
        # Remove outliers in momentum (using 500 GeV cutoff, no normalization)
        pt[np.where(np.abs(pt) > 500.0)] = 0.0
        px[np.where(np.abs(px) > 500.0)] = 0.0  
        py[np.where(np.abs(py) > 500.0)] = 0.0
        
        # Sanitize dxyErr and hcalDepth
        dxyErr[np.where(dxyErr == -999)] = 0.0
        dxyErr[np.where(np.abs(dxyErr) > 100.0)] = 0.0
        hcalDepth[np.where(~np.isfinite(hcalDepth))] = 0.0
        hcalDepth[np.where(np.abs(hcalDepth) > 100.0)] = 0.0
        
        # Create single preprocessed feature array (gets split later in training)
        # Layout: [pt, eta, phi, puppi, hcal_depth, px, py, encoded_pdgId, encoded_charge]
        inputs_cat0_expanded = A[:, :, 6:7]  # encoded pdgId 
        inputs_cat1_expanded = A[:, :, 7:8]  # encoded charge 
        
        preprocessed_features = np.concatenate([
            pt,                    # [:, :, 0] 
            eta,                   # [:, :, 1]
            phi,                   # [:, :, 2] 
            puppi,                 # [:, :, 3]
            hcalDepth,             # [:, :, 4]
            px,                    # [:, :, 5]
            py,                    # [:, :, 6]
            inputs_cat0_expanded,  # [:, :, 7] 
            inputs_cat1_expanded   # [:, :, 8]
        ], axis=2)
        
        processed_results[sample_name] = (preprocessed_features, targets)
        
        logger.info(f"  {sample_name} processed shapes:")
        logger.info(f"    features: {preprocessed_features.shape} (all preprocessed features)")
        logger.info(f"    targets: {targets.shape}")
    
    logger.info("Preprocessing complete")
    return processed_results


def combine_shuffle_split(processed_results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                         data_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine samples and create train/val/test splits.
    
    Args:
        processed_results: Dict with sample_name -> (features, targets)
        data_cfg: Data configuration dict with train-val-test-split ratios
        
    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    logger.info("Combining samples and creating train/val/test splits")

    # Combine all samples
    all_features = []
    all_targets = []
    for sample_name, (features, targets) in processed_results.items():
        logger.info(f"Adding {sample_name}: {features.shape[0]} events")
        all_features.append(features)
        all_targets.append(targets)

    # Concatenate and shuffle
    X_combined = np.concatenate(all_features, axis=0)
    Y_combined = np.concatenate(all_targets, axis=0)
    logger.info(f"Combined dataset: {X_combined.shape[0]} total events")

    # Shuffle indices
    np.random.seed(42)  # For reproducibility
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)
    X_combined = X_combined[indices]
    Y_combined = Y_combined[indices]

    # Get split ratios from config
    split_config = data_cfg["train-val-test-split"]
    train_ratio = split_config["train"]
    val_ratio = split_config["val"]
    test_ratio = split_config["test"]

    # Calculate split indices
    train_split = int(train_ratio * len(X_combined))
    val_split = int((train_ratio + val_ratio) * len(X_combined))

    # Create splits
    X_train, Y_train = X_combined[:train_split], Y_combined[:train_split]
    X_val, Y_val = X_combined[train_split:val_split], Y_combined[train_split:val_split]
    X_test, Y_test = X_combined[val_split:], Y_combined[val_split:]

    logger.info(f"Train split: {X_train.shape[0]} events ({train_ratio:.1%})")
    logger.info(f"Val split: {X_val.shape[0]} events ({val_ratio:.1%})")
    logger.info(f"Test split: {X_test.shape[0]} events ({test_ratio:.1%})")

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def save_h5_files(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                  Y_train: np.ndarray, Y_val: np.ndarray, Y_test: np.ndarray,
                  output_dir: Union[str, Path], samples: Dict[str, int]) -> None:
    """
    Save preprocessed data to separate H5 files for train/val/test.
    
    Args:
        X_train, X_val, X_test: Feature arrays
        Y_train, Y_val, Y_test: Target arrays  
        output_dir: Directory to save files
        samples: Sample configuration dict for metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving preprocessed data to separate H5 files in {output_dir}")

    # Metadata to save in each file
    metadata = {
        'feature_layout': ['pt', 'eta', 'phi', 'puppi_weight', 'hcal_depth', 
                          'px', 'py', 'encoded_pdgId', 'encoded_charge'],
        'n_features': X_train.shape[2],
        'max_puppi_candidates': X_train.shape[1],
        'samples_used': list(samples.keys()),
        'sample_event_counts': list(samples.values())
    }

    # Save training data
    train_file_path = output_dir / "train.h5"
    with h5py.File(train_file_path, 'w') as f:
        f.create_dataset('features', data=X_train, compression='gzip', compression_opts=9)
        f.create_dataset('targets', data=Y_train, compression='gzip', compression_opts=9)
        
        # Save metadata as attributes
        for key, value in metadata.items():
            f.attrs[key] = value

    logger.info(f"Training data saved to {train_file_path}")
    logger.info(f"  Shape: features {X_train.shape}, targets {Y_train.shape}")
    logger.info(f"  File size: {train_file_path.stat().st_size / 1024**2:.1f} MB")

    # Save validation data
    val_file_path = output_dir / "val.h5"
    with h5py.File(val_file_path, 'w') as f:
        f.create_dataset('features', data=X_val, compression='gzip', compression_opts=9)
        f.create_dataset('targets', data=Y_val, compression='gzip', compression_opts=9)
        
        # Save metadata as attributes
        for key, value in metadata.items():
            f.attrs[key] = value

    logger.info(f"Validation data saved to {val_file_path}")
    logger.info(f"  Shape: features {X_val.shape}, targets {Y_val.shape}")
    logger.info(f"  File size: {val_file_path.stat().st_size / 1024**2:.1f} MB")

    # Save test data
    test_file_path = output_dir / "test.h5"
    with h5py.File(test_file_path, 'w') as f:
        f.create_dataset('features', data=X_test, compression='gzip', compression_opts=9)
        f.create_dataset('targets', data=Y_test, compression='gzip', compression_opts=9)
        
        # Save metadata as attributes
        for key, value in metadata.items():
            f.attrs[key] = value

    logger.info(f"Test data saved to {test_file_path}")
    logger.info(f"  Shape: features {X_test.shape}, targets {Y_test.shape}")
    logger.info(f"  File size: {test_file_path.stat().st_size / 1024**2:.1f} MB")

    # Summary
    total_size = (train_file_path.stat().st_size + val_file_path.stat().st_size + 
                  test_file_path.stat().st_size) / 1024**2
    logger.info(f"All H5 files saved successfully")
    logger.info(f"Total size: {total_size:.1f} MB")



def coerce_encoding(encoding_raw: Dict[str, Dict]) -> Dict[str, Dict[float, int]]:
    """Coerce YAML keys to numeric for robust mapping."""
    return {k: {float(kk): int(vv) for kk, vv in v.items()} for k, v in encoding_raw.items()}