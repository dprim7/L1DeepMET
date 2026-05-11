"""Recipes for scripts/ablation.py.

Each recipe defines a `default_arch` and `default_loss` (kwargs for ArchConfig
and CorrectedCompositeLoss respectively), and a list of `cells` overriding
those defaults per-cell. The full kwargs per cell are saved to result.json.

Adding a new study = appending a recipe here; no new runner script.
"""

# Production baseline architecture used as default in most studies:
SCALAR_BASE = dict(
    width=64, depth=3, mode=1, activation="relu",
    use_embeddings=True, binned_weight=0.0,
    weight_minus_one=True, use_sum=True,
    with_bias=False, use_2d_weights=False,
    bounded_weight=False,
    xy_balance_weight=0.0,
)

# Default loss kwargs (passed to CorrectedCompositeLoss).
DEFAULT_LOSS = dict(
    mae_weight=1.0, mse_weight=1.0,
    huber_weight=0.0, huber_delta=0.5,
    binned_weight=0.0, phi_weight=0.0,
    xy_balance_weight=0.0,
)


RECIPES = {

    # ─────────────────────────────────────────────────────────────────────────
    "binned_weight": {
        "default_arch": SCALAR_BASE,
        "default_loss": dict(DEFAULT_LOSS, xy_balance_weight=10.0),
        "cells": [
            {"name": "bw0",   "loss": {"binned_weight": 0.0}},
            {"name": "bw50",  "loss": {"binned_weight": 50.0}},
            {"name": "bw200", "loss": {"binned_weight": 200.0}},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    "loss_form": {
        "default_arch": SCALAR_BASE,
        "default_loss": DEFAULT_LOSS,
        "cells": [
            {"name": "mse_only",       "loss": {"mae_weight": 0.0, "mse_weight": 1.0}},
            {"name": "mae_only",       "loss": {"mae_weight": 1.0, "mse_weight": 0.0}},
            {"name": "huber_d05",      "loss": {"mae_weight": 0.0, "mse_weight": 0.0,
                                                "huber_weight": 1.0, "huber_delta": 0.5}},
            {"name": "mae_mse_xy0",    "loss": {"mae_weight": 1.0, "mse_weight": 1.0}},
            {"name": "mae_mse_xy10",   "loss": {"mae_weight": 1.0, "mse_weight": 1.0,
                                                "xy_balance_weight": 10.0}},
            {"name": "huber_d05_xy10", "loss": {"mae_weight": 0.0, "mse_weight": 0.0,
                                                "huber_weight": 1.0, "huber_delta": 0.5,
                                                "xy_balance_weight": 10.0}},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    "residual_head": {
        # Output-head ablation. Loss fixed at MAE+MSE+xy=10 (matches the
        # original residual_ablation.py for backwards comparability).
        "default_arch": SCALAR_BASE,
        "default_loss": dict(DEFAULT_LOSS, xy_balance_weight=10.0),
        "cells": [
            {"name": "unbounded_no_bias",   "arch": {"weight_minus_one": True}},
            {"name": "bounded_no_bias",     "arch": {"weight_minus_one": False, "bounded_weight": True}},
            {"name": "unbounded_with_bias", "arch": {"weight_minus_one": True, "with_bias": True}},
            {"name": "bounded_with_bias",   "arch": {"weight_minus_one": False, "bounded_weight": True, "with_bias": True}},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    "combined_best": {
        # Cross the two ablation winners: bounded head × MAE-only × xy=0.
        # Plus reference points for direct comparison.
        "default_arch": SCALAR_BASE,
        "default_loss": DEFAULT_LOSS,
        "cells": [
            # The combined-best hypothesis: 31.5–32.5 GeV
            {"name": "bounded_mae_only_xy0",
             "arch": {"weight_minus_one": False, "bounded_weight": True},
             "loss": {"mae_weight": 1.0, "mse_weight": 0.0}},

            # Reference: bounded with MAE+MSE (xy=0)
            {"name": "bounded_mae_mse_xy0",
             "arch": {"weight_minus_one": False, "bounded_weight": True},
             "loss": {"mae_weight": 1.0, "mse_weight": 1.0}},

            # Reference: unbounded with MAE only (xy=0) — should reproduce loss winner
            {"name": "unbounded_mae_only_xy0",
             "arch": {"weight_minus_one": True},
             "loss": {"mae_weight": 1.0, "mse_weight": 0.0}},
        ],
    },
}
