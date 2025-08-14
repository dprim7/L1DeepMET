import numpy as np

from l1deepmet.data.preprocessing import preprocess_batch


def _make_batch_X(batch_size: int, num_pf: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    X = np.zeros((batch_size, num_pf, 10), dtype=np.float32)
    # 0 pt, 1 px, 2 py
    X[:, :, 0] = rng.uniform(0.0, 100.0, size=(batch_size, num_pf))
    X[:, :, 1] = rng.normal(0.0, 10.0, size=(batch_size, num_pf))
    X[:, :, 2] = rng.normal(0.0, 10.0, size=(batch_size, num_pf))
    # 3 eta, 4 phi, 5 puppi
    X[:, :, 3] = rng.uniform(-2.5, 2.5, size=(batch_size, num_pf))
    X[:, :, 4] = rng.uniform(-np.pi, np.pi, size=(batch_size, num_pf))
    X[:, :, 5] = rng.uniform(0.0, 1.0, size=(batch_size, num_pf))
    # 6 pdgId(enc), 7 charge(enc) — encoded categories (small integers)
    X[:, :, 6] = rng.integers(0, 6, size=(batch_size, num_pf))
    X[:, :, 7] = rng.integers(0, 4, size=(batch_size, num_pf))
    # 8 dxyErr, 9 hcalDepth
    X[:, :, 8] = rng.uniform(0.0, 10.0, size=(batch_size, num_pf))
    X[:, :, 9] = rng.uniform(0.0, 5.0, size=(batch_size, num_pf))
    return X


def test_preprocess_batch_without_edges_and_without_truncation():
    B, N = 3, 12
    X = _make_batch_X(B, N)
    norm = 50.0
    inputs, Yr, metadata = preprocess_batch(X, normFac=norm)

    # Inputs: [Xi, Xp, Xc0, Xc1]
    assert isinstance(inputs, list) and len(inputs) == 4
    Xi, Xp, Xc0, Xc1 = inputs
    assert Xi.shape == (B, N, 5)
    assert Xp.shape == (B, N, 2)
    assert Xc0.shape == (B, N)
    assert Xc1.shape == (B, N)

    # Placeholder targets
    assert Yr.shape == (B, 2)

    # Metadata contains embedding dims inferred from sample
    assert "emb_input_dim" in metadata
    dims = metadata["emb_input_dim"]
    assert 0 in dims and 1 in dims
    assert dims[0] >= int(np.max(X[:, :1000, 6])) + 1  # conservative bound
    assert dims[1] >= int(np.max(X[:, :1000, 7])) + 1


def test_preprocess_batch_with_truncation_and_edges():
    B, N = 2, 8
    X = _make_batch_X(B, N)
    norm = 20.0
    maxNPF = 5
    edge_list = ["dR", "kT", "z", "m2"]
    inputs, Yr, metadata = preprocess_batch(X, normFac=norm, maxNPF=maxNPF, edge_list=edge_list)

    assert isinstance(inputs, list) and len(inputs) == 5
    Xi, Xp, Xc0, Xc1, ef = inputs
    # Truncation applied
    assert Xi.shape[1] == maxNPF
    assert Xp.shape[1] == maxNPF
    assert Xc0.shape[1] == maxNPF
    assert Xc1.shape[1] == maxNPF
    # Edge features present and consistent with fully connected directed graph without self-loops
    expected_num_edges = maxNPF * (maxNPF - 1)
    assert ef.shape == (B, expected_num_edges, 4)

    # edge_idx stored in metadata
    assert "edge_idx" in metadata
    edge_idx = metadata["edge_idx"]
    assert edge_idx.shape == (expected_num_edges, 2)

    # Placeholder targets
    assert Yr.shape == (B, 2)

