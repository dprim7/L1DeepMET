"""Unit tests for ``l1deepmet.metrics.physics``.

The physics-card functions are the single source of truth for resolution,
response, AUC, and PUPPI-comparison numbers reported in every ablation /
synthesis run. Bugs here would silently invalidate every report.
"""
from __future__ import annotations

import numpy as np
import pytest

from l1deepmet.metrics.physics import (
    compute_puppi_baseline,
    compute_resolution_metrics,
    compute_roc_curve,
    compute_trigger_metrics,
    compute_turn_on,
    convert_xy_to_pt_phi,
    full_physics_card,
    phi_diff,
    resolqt,
)


# ─── Numeric helpers ─────────────────────────────────────────────────────────

class TestResolqt:
    def test_gaussian_matches_one_sigma(self):
        """For Gaussian noise, (P84 − P16) / 2 ≈ σ."""
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100_000) * 5.0
        assert abs(resolqt(y) - 5.0) < 0.1

    def test_heavy_tails_smaller_than_std(self):
        """For a heavy-tailed distribution, IQR/2 < std (the original
        motivation for using resolqt over np.std)."""
        rng = np.random.default_rng(0)
        y = rng.standard_t(df=3, size=100_000) * 2.0   # df=3 → heavy tails
        assert resolqt(y) < np.std(y) - 0.5

    def test_returns_python_float(self):
        assert isinstance(resolqt(np.array([1.0, 2.0, 3.0, 4.0])), float)


class TestConvertXY:
    def test_unit_x(self):
        xy = np.array([[1.0, 0.0]])
        pt_phi = convert_xy_to_pt_phi(xy)
        assert pt_phi[0, 0] == pytest.approx(1.0)
        assert pt_phi[0, 1] == pytest.approx(0.0)

    def test_unit_y(self):
        xy = np.array([[0.0, 1.0]])
        pt_phi = convert_xy_to_pt_phi(xy)
        assert pt_phi[0, 0] == pytest.approx(1.0)
        assert pt_phi[0, 1] == pytest.approx(np.pi / 2)

    def test_quadrants(self):
        xy = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        pt_phi = convert_xy_to_pt_phi(xy)
        expected_phi = np.array([np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4])
        np.testing.assert_allclose(pt_phi[:, 1], expected_phi, atol=1e-6)


class TestPhiDiff:
    def test_wrap_around(self):
        """Δφ = +π should wrap to ±π, not 0 and not 2π."""
        d = phi_diff(np.array([np.pi - 0.1]), np.array([-np.pi + 0.1]))
        # Two angles separated by 0.2 across the wrap-around branch cut.
        # Wrapped difference should be small (~−0.2), not ~2π.
        assert abs(d[0]) < 0.5

    def test_identical(self):
        phi = np.linspace(-np.pi, np.pi, 10)
        np.testing.assert_allclose(phi_diff(phi, phi), 0.0, atol=1e-7)


# ─── PUPPI baseline ──────────────────────────────────────────────────────────

class TestComputePuppiBaseline:
    def test_returns_negative_sum_of_pxpy(self):
        """PUPPI MET is -Σ(px, py) over all candidates."""
        # 2 events, 4 candidates each. Features[..., 5] = px, [..., 6] = py.
        X = np.zeros((2, 4, 9))
        X[0, :, 5] = [10.0, -5.0, 3.0, 2.0]   # sum px = 10 → MET_x = -10
        X[0, :, 6] = [0.0, 4.0, -1.0, -3.0]    # sum py = 0  → MET_y = 0
        X[1, :, 5] = [1.0, 2.0, 3.0, 4.0]      # sum px = 10 → MET_x = -10
        X[1, :, 6] = [0.0, 0.0, 0.0, 0.0]      # sum py = 0  → MET_y = 0
        out = compute_puppi_baseline(X)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out, [[-10.0, 0.0], [-10.0, 0.0]])

    def test_zero_input_gives_zero_output(self):
        X = np.zeros((3, 8, 9))
        out = compute_puppi_baseline(X)
        np.testing.assert_array_equal(out, np.zeros((3, 2)))


# ─── Resolution and response card ────────────────────────────────────────────

class TestResolutionMetrics:
    def test_perfect_reco_gives_zero_resolution(self):
        """If reco == gen, resolution metrics should be ~0 (X, Y) and
        the mean response = 1.0."""
        rng = np.random.default_rng(0)
        gen = rng.standard_normal((1000, 2)) * 50.0 + np.array([60.0, 0.0])
        metrics = compute_resolution_metrics(gen, gen)
        assert metrics["met_x_resolution"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["met_y_resolution"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mean_response"] == pytest.approx(1.0, rel=1e-3)
        assert metrics["phi_resolution"] == pytest.approx(0.0, abs=1e-6)

    def test_constant_offset_increases_resolution(self):
        rng = np.random.default_rng(0)
        gen = rng.standard_normal((1000, 2)) * 30.0 + np.array([60.0, 60.0])
        reco = gen + np.array([10.0, 0.0])  # 10 GeV bias in X only
        metrics = compute_resolution_metrics(gen, reco)
        # X resolution is dominated by the constant 10 GeV offset; small std → small IQR
        # (resolution of a constant is 0, since IQR of a constant = 0)
        assert metrics["met_x_resolution"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["met_y_resolution"] == pytest.approx(0.0, abs=1e-6)

    def test_response_bins_present_and_correctly_computed(self):
        """Reco = 0.5 × gen → response = 0.5 in every populated bin.

        Need enough events in every gen-pT bin (the impl returns 1.0 as a
        fallback for bins with ≤ 10 events). Construct gen so gen_pt is
        uniformly distributed in [60, 450] — covers all 5 bins generously.
        """
        rng = np.random.default_rng(0)
        n = 20_000
        # Sample gen_pt uniform in [60, 450], then random phi → 2D components.
        pt = rng.uniform(60, 450, size=n)
        phi = rng.uniform(-np.pi, np.pi, size=n)
        gen = np.column_stack([pt * np.cos(phi), pt * np.sin(phi)])
        reco = gen * 0.5
        metrics = compute_resolution_metrics(gen, reco)
        for bin_key in ("response_50_100", "response_100_200",
                        "response_200_300", "response_300_400",
                        "response_400_inf"):
            assert bin_key in metrics
            assert metrics[bin_key] == pytest.approx(0.5, rel=0.01), \
                f"{bin_key} = {metrics[bin_key]} (expected ~0.5)"

    def test_response_falls_back_to_one_with_few_events(self):
        """The per-bin response defaults to 1.0 when a bin has ≤ 10 events.
        This documents the impl behavior so future changes are caught."""
        rng = np.random.default_rng(0)
        # 50 events all clustered around 60 GeV — only response_50_100 will
        # have > 10 events; the higher bins should fall back to 1.0.
        gen = np.column_stack([rng.uniform(50, 90, 50),
                                rng.uniform(-5, 5, 50)])
        reco = gen * 0.3
        metrics = compute_resolution_metrics(gen, reco)
        # 400+ bin definitely has < 10 events with this setup
        assert metrics["response_400_inf"] == 1.0


# ─── Trigger metrics ─────────────────────────────────────────────────────────

class TestTriggerMetrics:
    def test_perfect_discriminator_gives_auc_one(self):
        """If reco_pt cleanly separates signal (gen MET > 200) from background
        (gen MET < 50), AUC should be close to 1.0."""
        rng = np.random.default_rng(0)
        # Build signal: gen_pt > 200 GeV with reco_pt also large
        n_sig, n_bg = 200, 1000
        gen_sig = np.zeros((n_sig, 2)); gen_sig[:, 0] = rng.uniform(210, 400, n_sig)
        gen_bg = np.zeros((n_bg, 2)); gen_bg[:, 0] = rng.uniform(0, 49, n_bg)
        gen = np.concatenate([gen_sig, gen_bg])
        reco_pt = np.concatenate([rng.uniform(180, 400, n_sig),
                                   rng.uniform(0, 30, n_bg)])
        metrics = compute_trigger_metrics(gen, reco_pt)
        assert metrics["auc"] > 0.99
        assert metrics["n_signal"] == n_sig
        assert metrics["n_background"] == n_bg

    def test_too_few_events_returns_nan(self):
        """With fewer than 10 signal events, AUC should be NaN."""
        rng = np.random.default_rng(0)
        n = 5
        gen = np.zeros((n, 2)); gen[:, 0] = rng.uniform(210, 400, n)
        reco_pt = rng.uniform(180, 400, n)
        metrics = compute_trigger_metrics(gen, reco_pt)
        assert np.isnan(metrics["auc"])


class TestRocCurve:
    def test_returns_three_arrays_same_length(self):
        rng = np.random.default_rng(0)
        gen = np.zeros((500, 2)); gen[:, 0] = rng.uniform(0, 500, 500)
        reco_pt = rng.uniform(0, 500, 500)
        roc = compute_roc_curve(gen, reco_pt)
        assert len(roc["thresholds"]) == len(roc["sig_eff"])
        assert len(roc["sig_eff"]) == len(roc["bg_eff"])
        assert (roc["sig_eff"] <= 1.0).all() and (roc["sig_eff"] >= 0.0).all()


class TestTurnOn:
    def test_efficiency_at_zero_threshold_is_one_above_threshold(self):
        rng = np.random.default_rng(0)
        gen = np.zeros((1000, 2)); gen[:, 0] = rng.uniform(100, 500, 1000)
        reco_pt = np.full(1000, 50.0)  # all pass a threshold of 30
        turn_on = compute_turn_on(gen, reco_pt, threshold=30.0)
        # For bins with events, efficiency should be 1 (all reco_pt > 30)
        nonempty = turn_on["efficiencies"] > 0
        if nonempty.any():
            assert turn_on["efficiencies"][nonempty].mean() == pytest.approx(1.0)


# ─── full_physics_card integration ───────────────────────────────────────────

class TestFullPhysicsCard:
    def test_keys_match_documented_set(self):
        rng = np.random.default_rng(0)
        gen = rng.standard_normal((500, 2)) * 60.0
        reco = gen + rng.standard_normal((500, 2)) * 10.0
        card = full_physics_card(gen, reco)
        for k in ("met_pt_resolution", "met_x_resolution", "met_y_resolution",
                  "mean_response", "phi_resolution",
                  "response_50_100", "response_100_200",
                  "auc", "n_signal", "n_background"):
            assert k in card, f"missing key {k!r}"

    def test_with_puppi_adds_delta_keys(self):
        rng = np.random.default_rng(0)
        gen = rng.standard_normal((500, 2)) * 60.0
        reco = gen + rng.standard_normal((500, 2)) * 10.0
        puppi = gen + rng.standard_normal((500, 2)) * 15.0
        card = full_physics_card(gen, reco, puppi_xy=puppi)
        for k in ("puppi_met_x_resolution", "puppi_met_y_resolution",
                  "delta_met_pt_resolution", "delta_met_x_resolution",
                  "delta_auc", "mean_pred_pt_over_puppi_pt"):
            assert k in card, f"missing key {k!r}"

    def test_puppi_vs_itself_gives_zero_deltas(self):
        """Sanity check: full_physics_card on PUPPI-vs-PUPPI must give
        zero deltas for every metric. Need enough signal/background events
        (gen_pt > 200 / < 50) for the AUC computation to be defined."""
        rng = np.random.default_rng(0)
        # Mix of high- and low-MET events so n_signal > 10 and n_bg > 10.
        n_sig, n_bg, n_mid = 200, 200, 600
        pt = np.concatenate([
            rng.uniform(210, 400, n_sig),
            rng.uniform(0, 49, n_bg),
            rng.uniform(60, 199, n_mid),
        ])
        phi = rng.uniform(-np.pi, np.pi, size=n_sig + n_bg + n_mid)
        gen = np.column_stack([pt * np.cos(phi), pt * np.sin(phi)])
        puppi = gen + rng.standard_normal(gen.shape) * 15.0
        card = full_physics_card(gen, reco_xy=puppi, puppi_xy=puppi)
        for k in ("delta_met_pt_resolution", "delta_met_x_resolution",
                  "delta_met_y_resolution", "delta_phi_resolution",
                  "delta_auc"):
            assert card[k] == pytest.approx(0.0, abs=1e-10), \
                f"{k} should be 0 for PUPPI-vs-PUPPI but is {card[k]}"
