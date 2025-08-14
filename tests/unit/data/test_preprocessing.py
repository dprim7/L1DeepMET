import numpy as np
import awkward as ak  # type: ignore

from l1deepmet.data.preprocessing import (
    to_np_array,
    convertXY2PtPhi,
    compute_px_py,
    HCalDepth,
    encode_pdgId_charge,
    preProcessing,
    normalize_targets,
    sort_and_truncate_by_pt,
    deltaR_calc,
    kT_calc,
    z_calc,
    mass2_calc,
    build_edge_features,
)


def test_to_np_array_padding_and_truncation():
    ak_arr = ak.Array([[1, 2, 3], [], [4]])
    out = to_np_array(ak_arr, maxN=2, pad=-1)
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out[0], np.array([1, 2]))
    np.testing.assert_array_equal(out[1], np.array([-1, -1]))
    np.testing.assert_array_equal(out[2], np.array([4, -1]))


def test_convertXY2PtPhi_simple():
    xy = np.array([[3.0, 4.0], [0.0, -1.0]], dtype=np.float32)
    ptphi = convertXY2PtPhi(xy)
    np.testing.assert_allclose(ptphi[:, 0], np.array([5.0, 1.0]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ptphi[:, 1], np.arctan2(xy[:, 1], xy[:, 0]), rtol=1e-6, atol=1e-6)


def test_compute_px_py_broadcasting():
    pt = np.array([[1.0, 2.0], [3.0, 4.0]])
    phi = np.array([[0.0, np.pi / 2], [np.pi, -np.pi / 2]])
    px, py = compute_px_py(pt, phi)
    np.testing.assert_allclose(px, pt * np.cos(phi))
    np.testing.assert_allclose(py, pt * np.sin(phi))


def test_HCalDepth_with_and_without_zero_total():
    h1 = np.array([1.0, 2.0, 0.0])
    h3 = np.array([3.0, 5.0, 0.0])
    h5 = np.array([6.0, 0.0, 0.0])
    depth = HCalDepth(h1, h3, h5)
    expected_nonzero = (h1[0] * 1.0 + (h3[0] - h1[0]) * 3.0 + (h5[0] - h3[0]) * 5.0) / h5[0]
    assert depth.shape == h1.shape
    np.testing.assert_allclose(depth[0], expected_nonzero)
    assert depth[1] == 0.0  # h5 == 0 leads to 0 depth
    assert depth[2] == 0.0


def test_encode_pdgId_charge_mapping():
    pdgid = np.array([-211.0, 22.0, -999.0, 13.0])
    charge = np.array([-1.0, 0.0, -999.0, 1.0])
    enc_pdg, enc_q = encode_pdgId_charge(pdgid, charge)
    np.testing.assert_array_equal(enc_pdg, np.array([1, 3, 0, 4]))
    np.testing.assert_array_equal(enc_q, np.array([1, 2, 0, 3]))


def _make_synthetic_A(batch_size: int, num_pf: int) -> np.ndarray:
    # A shape: (B, N, 10) with columns as expected by preProcessing
    rng = np.random.default_rng(42)
    A = np.zeros((batch_size, num_pf, 10), dtype=np.float32)
    # 0 pt, 1 px, 2 py
    A[:, :, 0] = rng.uniform(0.0, 100.0, size=(batch_size, num_pf))
    A[:, :, 1] = rng.normal(0.0, 10.0, size=(batch_size, num_pf))
    A[:, :, 2] = rng.normal(0.0, 10.0, size=(batch_size, num_pf))
    # 3 eta, 4 phi, 5 puppi
    A[:, :, 3] = rng.uniform(-2.5, 2.5, size=(batch_size, num_pf))
    A[:, :, 4] = rng.uniform(-np.pi, np.pi, size=(batch_size, num_pf))
    A[:, :, 5] = rng.uniform(0.0, 1.0, size=(batch_size, num_pf))
    # 6 pdgId(enc), 7 charge(enc)
    A[:, :, 6] = rng.integers(0, 6, size=(batch_size, num_pf))
    A[:, :, 7] = rng.integers(0, 4, size=(batch_size, num_pf))
    # 8 dxyErr, 9 hcalDepth
    A[:, :, 8] = rng.uniform(0.0, 10.0, size=(batch_size, num_pf))
    A[:, :, 9] = rng.uniform(0.0, 5.0, size=(batch_size, num_pf))
    return A


def test_preProcessing_shapes_and_sanitization():
    B, N = 2, 5
    A = _make_synthetic_A(B, N)
    # induce outliers and sentinel values
    A[0, 0, 0] = 600.0  # pt outlier
    A[0, 1, 1] = 600.0  # px outlier
    A[0, 2, 2] = -600.0  # py outlier
    A[0, 3, 8] = -999.0  # dxyErr sentinel
    A[0, 4, 9] = np.inf  # hcalDepth non-finite

    norm = 10.0
    Xi, Xp, Xc0, Xc1 = preProcessing(A, norm)

    assert Xi.shape == (B, N, 5)
    assert Xp.shape == (B, N, 2)
    assert Xc0.shape == (B, N)
    assert Xc1.shape == (B, N)

    # Outliers should be zeroed after normalization
    assert Xi[0, 0, 0] == 0.0  # pt -> in Xi at index 0
    assert Xp[0, 1, 0] == 0.0  # px
    assert Xp[0, 2, 1] == 0.0  # py

    # Sanitization
    # Xi components are [pt, eta, phi, puppi, hcalDepth]
    assert Xi[0, 3, 4] == 0.0  # hcalDepth non-finite -> 0

    # dxyErr is not returned, but ensure it would be sanitized internally
    # by checking that sanitization does not crash and shapes hold


def test_normalize_targets():
    Y = np.array([[10.0, -20.0], [30.0, 40.0]], dtype=np.float32)
    out = normalize_targets(Y, normFac=10.0)
    np.testing.assert_allclose(out, Y / 10.0)
    out_inv = normalize_targets(Y, normFac=10.0, invert_sign=True)
    np.testing.assert_allclose(out_inv, Y / -10.0)


def test_sort_and_truncate_by_pt_orders_desc_and_truncates():
    # X[:,:,0] is pt
    X = np.array([
        [[1.0, 0, 0], [3.0, 0, 0], [2.0, 0, 0]],
    ], dtype=np.float32)
    X_sorted = sort_and_truncate_by_pt(X, maxNPF=2)
    np.testing.assert_allclose(X_sorted[0, :, 0], np.array([3.0, 2.0]))


def test_deltaR_calc_phi_wrapping():
    eta1 = np.array([[0.0]])
    phi1 = np.array([[np.pi - 0.1]])
    eta2 = np.array([[0.0]])
    phi2 = np.array([[-np.pi + 0.1]])
    dR = deltaR_calc(eta1, phi1, eta2, phi2)
    np.testing.assert_allclose(dR, 0.2, atol=1e-6)


def test_kT_calc_and_z_calc():
    pti = np.array([[2.0, 5.0]])
    ptj = np.array([[3.0, 1.0]])
    dR = np.array([[0.5, 2.0]])
    kT = kT_calc(pti, ptj, dR)
    z = z_calc(pti, ptj)
    np.testing.assert_allclose(kT, np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(z, np.array([[2.0 / 5.0, 1.0 / 6.0]]), rtol=1e-6)


def test_mass2_calc_simple_back_to_back():
    # Two identical massless-like momenta back-to-back yield positive m2
    # Construct p4 = (E, px, py, pz)
    E = 10.0
    px = 6.0
    py = 8.0
    pz = 0.0
    p1 = np.array([E, px, py, pz])
    p2 = np.array([E, -px, -py, -pz])
    pi = np.stack([p1])  # (4,)
    pj = np.stack([p2])
    pi = np.broadcast_to(pi, (1, 1, 4))
    pj = np.broadcast_to(pj, (1, 1, 4))
    m2 = mass2_calc(pi, pj)
    # Combined p = (2E, 0, 0, 0) => m2 = (2E)^2
    np.testing.assert_allclose(m2, (2 * E) ** 2)


def test_build_edge_features_shapes_and_values():
    # Construct tiny batch with N=3
    B, N = 1, 3
    # Xi: [:,:,0]=pt, [:,:,1]=eta, [:,:,2]=phi
    Xi = np.zeros((B, N, 5), dtype=np.float32)
    Xi[0, :, 0] = np.array([1.0, 2.0, 3.0])
    Xi[0, :, 1] = np.array([0.0, 0.0, 0.0])
    Xi[0, :, 2] = np.array([0.0, np.pi / 2, np.pi])
    # Xp: [:,:,0]=px, [:,:,1]=py consistent with pt,phi for eta=0
    px = Xi[0, :, 0] * np.cos(Xi[0, :, 2])
    py = Xi[0, :, 0] * np.sin(Xi[0, :, 2])
    Xp = np.zeros((B, N, 2), dtype=np.float32)
    Xp[0, :, 0] = px
    Xp[0, :, 1] = py

    ef, edge_idx = build_edge_features(Xi, Xp, edge_list=["dR", "kT", "z", "m2"])
    assert ef.shape == (B, N * (N - 1), 4)
    assert edge_idx.shape == (N * (N - 1), 2)

    # Check a specific pair: (receiver=0, sender=1) present in edge_idx
    # Find its row
    pair = np.array([0, 1])
    rows = np.where((edge_idx == pair).all(axis=1))[0]
    assert len(rows) == 1
    r = rows[0]

    # dR between (0,1): eta equal, phi difference pi/2
    np.testing.assert_allclose(ef[0, r, 0], np.abs(np.pi / 2), rtol=1e-6)

    # kT uses min(pt) * dR: min(1,2) * pi/2
    np.testing.assert_allclose(ef[0, r, 1], 1.0 * np.abs(np.pi / 2), rtol=1e-6)

    # z = min / (sum)
    np.testing.assert_allclose(ef[0, r, 2], 1.0 / 3.0, rtol=1e-6)

    # m2 should be positive and non-zero for non-collinear vectors
    assert ef[0, r, 3] > 0.0

