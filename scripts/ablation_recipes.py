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


# Best-known loss/head settings as of the combined-best ablation:
BEST_LOSS = dict(DEFAULT_LOSS, mae_weight=1.0, mse_weight=0.0,
                 xy_balance_weight=0.0)


RECIPES = {

    # ─────────────────────────────────────────────────────────────────────────
    # Architecture sweep under the CORRECTED loss (mae_only, no xy_balance,
    # no bias, binned_weight=0). The prior dense_architecture_baseline_apr2026
    # report swept this but with binned_weight=200, so its rankings reflect
    # a now-known-bad loss. This sweep regenerates the architecture landscape
    # with the correct loss to confirm or refute the ~33.2 GeV "plateau"
    # claim from the combined_best ablation.
    "arch_sweep": {
        "default_arch": SCALAR_BASE,
        "default_loss": BEST_LOSS,
        "cells": [
            {"name": f"w{w}_d{d}", "arch": {"width": w, "depth": d}}
            for w in [32, 64, 128, 256]
            for d in [2, 3, 4]
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Transformer loss × head sweep. The arch_comparison_recon showed the
    # transformer over-shrinks: X/Y IQR/2 looks great (≈30 GeV) but pT
    # IQR/2 degrades by ~12 GeV vs PUPPI and AUC drops to PUPPI. The recon
    # used MAE loss with weight_minus_one init.
    #
    # Hypothesis: the over-shrinkage comes from the loss having a degenerate
    # "predict the median" optimum, not from architecture per se. Two
    # orthogonal interventions to test:
    #   - Loss: MAE (median-seeking, prone to shrinkage on zero-mean targets)
    #     vs MSE (mean-seeking, optimum is the conditional mean — much harder
    #     to satisfy by collapse).
    #   - Head: weight_minus_one (unbounded scalar weight × pxpy, free to
    #     shrink toward 0) vs bounded_weight (effective weight ∈ [-2, 0],
    #     model can't easily collapse contributions).
    #
    # 4 cells × 3 seeds. All same transformer body (w64 d2 h2 kd6 ff8,
    # ~9.9k params) so this is loss/head only.
    #
    # NOTE: an earlier attempt added a response-regularisation loss term to
    # fix this. We rejected it — the mean-drift hinge is structurally the
    # same instrument as BinnedDeviation (which we removed for hurting
    # resolution); at the parameters that would matter it would pull the
    # MLP off its 0.7 Wiener response toward 1.0 and lose ~1 GeV of X/Y.
    # See git history for the reverted commit.
    "transformer_loss_head": {
        "default_arch": dict(SCALAR_BASE, width=64, depth=2, mode=1,
                             use_sum=True,
                             body_type="transformer",
                             num_heads=2, key_dim=6, ffn_dim=8),
        "default_loss": DEFAULT_LOSS,  # MAE+MSE both 1.0 default
        "cells": [
            # MAE only + weight_minus_one — reproduces the over-shrinkage
            # baseline from the recon.
            {"name": "xformer_mae_wmo",
             "arch": {"weight_minus_one": True, "bounded_weight": False},
             "loss": {"mae_weight": 1.0, "mse_weight": 0.0}},
            # MSE only + weight_minus_one — does mean-seeking loss fix it?
            {"name": "xformer_mse_wmo",
             "arch": {"weight_minus_one": True, "bounded_weight": False},
             "loss": {"mae_weight": 0.0, "mse_weight": 1.0}},
            # MAE only + bounded weight — does architectural constraint fix it?
            {"name": "xformer_mae_bounded",
             "arch": {"weight_minus_one": False, "bounded_weight": True},
             "loss": {"mae_weight": 1.0, "mse_weight": 0.0}},
            # MSE only + bounded weight — both interventions combined.
            {"name": "xformer_mse_bounded",
             "arch": {"weight_minus_one": False, "bounded_weight": True},
             "loss": {"mae_weight": 0.0, "mse_weight": 1.0}},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Deep Sets with explicit ρ head (the CMS L1 jet-tagger structure,
    # arXiv:2509.24371): per-particle φ MLP → sum → ρ MLP → output.
    # mode=0 picks the post-aggregation Dense path; rho_depth > 0 turns the
    # default Dense(2) head into a proper ρ MLP. Compared against the current
    # mode-1 "scalar weight × pxpy → sum" head on equal terms.
    "deepsets_rho": {
        "default_arch": dict(SCALAR_BASE, mode=0, use_sum=True,
                             weight_minus_one=False, bounded_weight=False,
                             with_bias=False),
        "default_loss": BEST_LOSS,
        "cells": [
            # Baseline: mode=0 with no ρ head (= current mode-0 with sum pool).
            {"name": "deepsets_rho0",            "arch": {"width": 64, "depth": 3, "rho_depth": 0}},
            # 1-layer ρ head, ρ_width matching φ.
            {"name": "deepsets_rho1_w64_d3",     "arch": {"width": 64, "depth": 3, "rho_depth": 1, "rho_width": 64}},
            # 2-layer ρ head.
            {"name": "deepsets_rho2_w64_d3",     "arch": {"width": 64, "depth": 3, "rho_depth": 2, "rho_width": 64}},
            # Slimmer φ + slimmer ρ (closer to CMS L1 jet-tagger sizing).
            {"name": "deepsets_rho2_w32_d3",     "arch": {"width": 32, "depth": 3, "rho_depth": 2, "rho_width": 32}},
            # Reference: current production mode-1 head with the same φ body
            # (re-runs from existing combined_best winner for parity).
            {"name": "mode1_w64_d3_ref",
             "arch": dict(SCALAR_BASE, width=64, depth=3, mode=1,
                          weight_minus_one=True, use_sum=True),
             "loss": BEST_LOSS},
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # Architecture family comparison at matched parameter count (~10k), same
    # corrected loss (mae_only, no xy_balance, no bias). Three families:
    #   - mlp        : DeepMET / Deep Sets (Dense + BN per layer, weight-shared)
    #   - transformer: 2-head MHA encoder per arXiv:2402.01047 style;
    #                  key_dim=6 keeps params near 10k
    #   - smaller MLPs as a "less capacity, fewer LUTs" reference point
    "arch_comparison": {
        "default_arch": SCALAR_BASE,
        "default_loss": BEST_LOSS,
        "cells": [
            # MLP / Deep Sets family
            {"name": "mlp_w64_d3",   "arch": {"width": 64, "depth": 3, "body_type": "mlp"}},
            {"name": "mlp_w32_d3",   "arch": {"width": 32, "depth": 3, "body_type": "mlp"}},
            {"name": "mlp_w32_d4",   "arch": {"width": 32, "depth": 4, "body_type": "mlp"}},
            # Transformer family — ~10k params at d=2 kd=6
            {"name": "xformer_w64_d2_h2_kd6_ff8",
             "arch": {"width": 64, "depth": 2, "body_type": "transformer",
                      "num_heads": 2, "key_dim": 6, "ffn_dim": 8}},
            {"name": "xformer_w64_d3_h2_kd6_ff8",
             "arch": {"width": 64, "depth": 3, "body_type": "transformer",
                      "num_heads": 2, "key_dim": 6, "ffn_dim": 8}},
            # Bigger transformer for ceiling reference
            {"name": "xformer_w64_d3_h2_kd16_ff16",
             "arch": {"width": 64, "depth": 3, "body_type": "transformer",
                      "num_heads": 2, "key_dim": 16, "ffn_dim": 16}},
        ],
    },


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
