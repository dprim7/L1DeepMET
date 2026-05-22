"""Stub tests for the L1 physics card extensions.

These mirror the stubs in ``src/l1deepmet/metrics/physics.py`` and are
skipped until each function's design is fixed. Per CLAUDE.md test-first
standing order:

  1. Pick the function's contract (sign off on the design doc in
     ``reports/physics_card_design/DESIGN.md``).
  2. Replace ``pytest.skip(...)`` here with real expected-behaviour tests
     (use a synthetic toy where you know the answer — e.g., perfect
     prediction → fake_frac = 0, miss_frac = 0).
  3. Run them; they should fail (red).
  4. Implement in ``src/l1deepmet/metrics/physics.py``.
  5. Run; should pass (green). Commit.

Until then this file exists to document that the stubs *will* have tests,
not to assert anything.
"""
import pytest

from l1deepmet.metrics.physics import (
    compute_asymmetric_tails,
    compute_per_eta_card,
    compute_per_pu_card,
    compute_puppi_ablation,
    compute_rate_vs_threshold,
    compute_working_point,
)


@pytest.mark.skip(reason="design pending — see reports/physics_card_design/DESIGN.md")
def test_compute_rate_vs_threshold():
    pass


@pytest.mark.skip(reason="design pending — needs target_rate_khz fixed")
def test_compute_working_point():
    pass


@pytest.mark.skip(reason="design pending — fixed vs scaled Δ undecided")
def test_compute_asymmetric_tails():
    pass


@pytest.mark.skip(reason="design pending — PU bin edges undecided")
def test_compute_per_pu_card():
    pass


@pytest.mark.skip(reason="design pending — per-event eta definition undecided")
def test_compute_per_eta_card():
    pass


@pytest.mark.skip(reason="design pending — eval_fn API undecided")
def test_compute_puppi_ablation():
    pass


def test_stubs_raise_not_implemented():
    """Sanity: the stubs do raise NotImplementedError until implemented.

    This keeps the test file from being entirely skipped (which would be
    misleading) and gives a single passing assertion that the stub
    scaffold itself is wired correctly.
    """
    import numpy as np
    for fn, args in [
        (compute_rate_vs_threshold, (np.zeros(5),)),
        (compute_working_point,     ({}, {}, 4.0)),
        (compute_asymmetric_tails,  (np.zeros((5, 2)), np.zeros((5, 2)))),
        (compute_per_pu_card,       (np.zeros((5, 2)), np.zeros((5, 2)), np.zeros(5))),
        (compute_per_eta_card,      (np.zeros((5, 2)), np.zeros((5, 2)), np.zeros(5))),
        (compute_puppi_ablation,    (lambda x: x, np.zeros((5, 128, 4)), np.zeros((5, 2)))),
    ]:
        with pytest.raises(NotImplementedError):
            fn(*args)
