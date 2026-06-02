"""FastPUPPI-style L1 MET trigger metrics: rate-vs-threshold + fixed-rate working point.

Definitions ported from FastPUPPI
``NtupleProducer/python/scripts/jetHtSuite.py`` (``makeCumulativeHTEff``,
``makeInclusiveEffRate``, ``effForRate``):

    rate[kHz] = NORM_KHZ * P(reco_MET > threshold)   on a minbias sample
    NORM_KHZ  = 2760 colliding bunches * 11246 Hz (LHC orbit) / 1000

The fixed-rate working point inverts this: the threshold giving ``target_rate``
is the ``1 - target/NORM`` quantile of the minbias reco-MET distribution, and the
deliverable is the SIGNAL efficiency at that threshold.

Toy datasets with analytically-known answers (test-first; these were written
before the implementation).
"""
import numpy as np
import pytest

from l1deepmet.metrics.physics import (
    NORM_KHZ,
    compute_rate_vs_threshold,
    compute_working_point,
)


def test_norm_constant_matches_fastpuppi():
    assert NORM_KHZ == pytest.approx(2760.0 * 11246.0 / 1000.0)
    assert NORM_KHZ == pytest.approx(31039.0, abs=1.0)


def test_rate_known_pass_fraction():
    reco = np.arange(100.0)  # 0..99
    out = compute_rate_vs_threshold(reco, thresholds_gev=np.array([50.0]))
    # P(reco > 50) = #{51..99}/100 = 49/100
    assert out["pass_fraction"][0] == pytest.approx(0.49)
    assert out["rate_khz"][0] == pytest.approx(NORM_KHZ * 0.49)


def test_rate_is_monotonically_non_increasing():
    rng = np.random.default_rng(0)
    reco = rng.exponential(40.0, size=20000)
    out = compute_rate_vs_threshold(reco, thresholds_gev=np.arange(0, 300, 5.0))
    assert np.all(np.diff(out["rate_khz"]) <= 1e-9)


def test_rate_all_pass_and_none_pass():
    reco = np.full(1000, 100.0)
    out = compute_rate_vs_threshold(reco, thresholds_gev=np.array([50.0, 150.0]))
    assert out["rate_khz"][0] == pytest.approx(NORM_KHZ)   # all pass
    assert out["rate_khz"][1] == pytest.approx(0.0)        # none pass


def test_rate_default_thresholds_present():
    out = compute_rate_vs_threshold(np.arange(50.0))
    assert out["thresholds"].min() == 0.0 and out["thresholds"].max() >= 500.0
    assert out["rate_khz"].shape == out["thresholds"].shape


def test_working_point_threshold_is_correct_quantile():
    reco_mb = np.arange(1000.0)                 # 0..999
    target = NORM_KHZ * 0.1                      # 10% pass -> 90th percentile
    wp = compute_working_point(reco_mb,
                               reco_pt_signal=np.full(50, 1e9),
                               gen_pt_signal=np.full(50, 300.0),
                               target_rate_khz=target)
    assert wp["threshold_gev"] == pytest.approx(np.quantile(reco_mb, 0.9), rel=1e-6)
    # achieved rate at that threshold should reproduce the requested target
    assert wp["achieved_rate_khz"] == pytest.approx(target, rel=0.02)


def test_working_point_plateau_efficiency_perfect_and_zero():
    reco_mb = np.arange(1000.0)
    target = NORM_KHZ * 0.05                     # threshold ~ 950
    wp_hi = compute_working_point(reco_mb, np.full(100, 1e9), np.full(100, 300.0), target)
    assert wp_hi["plateau_efficiency"] == pytest.approx(1.0)
    wp_lo = compute_working_point(reco_mb, np.zeros(100), np.full(100, 300.0), target)
    assert wp_lo["plateau_efficiency"] == pytest.approx(0.0)


def test_working_point_plateau_uses_gen_cut():
    reco_mb = np.arange(1000.0)
    target = NORM_KHZ * 0.5
    # signal passes the L1 threshold but gen is below the plateau cut -> excluded
    wp = compute_working_point(reco_mb, np.full(100, 1e9), np.full(100, 50.0),
                               target, plateau_gen_min=200.0)
    assert np.isnan(wp["plateau_efficiency"])
    assert wp["n_signal_plateau"] == 0


def test_working_point_impossible_rate_gives_zero_threshold():
    reco_mb = np.arange(1000.0)
    wp = compute_working_point(reco_mb, np.full(10, 500.0), np.full(10, 300.0),
                               target_rate_khz=NORM_KHZ * 2.0)  # > total rate
    assert wp["threshold_gev"] == pytest.approx(0.0)
