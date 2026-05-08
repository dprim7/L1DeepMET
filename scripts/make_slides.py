#!/usr/bin/env python3
"""Generate architecture search summary slides as a multi-page PDF.

Dependencies: matplotlib, numpy (no pandas required).
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "arch_search", "search_results.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "arch_search")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "arch_search_slides.pdf")

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
FIGSIZE = (11.69, 8.27)  # A4 landscape in inches
COLOR_BLUE = "#1f77b4"
COLOR_ORANGE = "#ff7f0e"
COLOR_LIGHT_BLUE = "#aec7e8"
COLOR_LIGHT_ORANGE = "#ffbb78"
COLOR_GREY = "#888888"
TITLE_FONTSIZE = 22
SUBTITLE_FONTSIZE = 14
BODY_FONTSIZE = 12
SMALL_FONTSIZE = 10


# ---------------------------------------------------------------------------
# CSV loading (no pandas)
# ---------------------------------------------------------------------------
def load_csv(path):
    """Load CSV into a list of dicts with typed values."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed = {}
            for k, v in row.items():
                # Type conversion
                if k in ("width", "depth", "mode", "param_count", "best_epoch", "epochs_trained"):
                    typed[k] = int(v)
                elif k in ("binned_weight", "val_loss", "val_mae", "val_mse",
                            "val_binned_deviation", "train_time_s"):
                    typed[k] = float(v)
                elif k == "use_embeddings":
                    typed[k] = v.strip() == "True"
                else:
                    typed[k] = v.strip()
            rows.append(typed)
    return rows


def col(rows, key):
    """Extract a column as numpy array."""
    return np.array([r[key] for r in rows])


def where(rows, **filters):
    """Filter rows by exact column matches."""
    out = []
    for r in rows:
        if all(r[k] == v for k, v in filters.items()):
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Slide helpers
# ---------------------------------------------------------------------------
def new_slide(title=None, title_y=0.92):
    """Create a new blank slide figure and return (fig, ax)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.text(
            0.5, title_y, title,
            ha="center", va="top", fontsize=TITLE_FONTSIZE,
            fontweight="bold", color="#222222",
        )
    return fig, ax


def add_bullet_block(ax, x, y, lines, fontsize=BODY_FONTSIZE, spacing=0.04, color="#333333"):
    """Add bullet-point lines. Returns y after last line."""
    for line in lines:
        ax.text(x, y, line, ha="left", va="top", fontsize=fontsize, color=color,
                family="monospace" if line.startswith("  ") else "sans-serif")
        y -= spacing
    return y


def save_slide(fig, pdf):
    fig.tight_layout(pad=0.5)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_csv(CSV_PATH)

    with PdfPages(OUTPUT_PDF) as pdf:
        # ==================================================================
        # SLIDE 1 -- Title
        # ==================================================================
        fig, ax = new_slide()
        ax.text(0.5, 0.62, "L1DeepMET Architecture Search",
                ha="center", va="center", fontsize=28, fontweight="bold", color="#1a1a1a")
        ax.text(0.5, 0.52, "Dense Model Optimization",
                ha="center", va="center", fontsize=22, color="#444444")
        ax.text(0.5, 0.40, "Per-particle MET weighting for L1 trigger at HL-LHC",
                ha="center", va="center", fontsize=SUBTITLE_FONTSIZE, color=COLOR_GREY,
                style="italic")
        ax.text(0.5, 0.28, "April 2, 2026", ha="center", va="center",
                fontsize=SUBTITLE_FONTSIZE, color=COLOR_GREY)
        ax.axhline(y=0.34, xmin=0.25, xmax=0.75, color=COLOR_BLUE, linewidth=2)
        save_slide(fig, pdf)

        # ==================================================================
        # SLIDE 2 -- Motivation & Setup
        # ==================================================================
        fig, ax = new_slide("Motivation & Experimental Setup")
        y = 0.82

        y = add_bullet_block(ax, 0.08, y, [
            "Goal",
        ], fontsize=15, spacing=0.045, color=COLOR_BLUE)
        y = add_bullet_block(ax, 0.10, y, [
            "\u2022  Find optimal Dense architecture for L1 MET reconstruction",
            "    within FPGA resource constraints (target: HLS4ML synthesis)",
        ], fontsize=BODY_FONTSIZE, spacing=0.04)

        y -= 0.02
        y = add_bullet_block(ax, 0.08, y, [
            "Data",
        ], fontsize=15, spacing=0.045, color=COLOR_BLUE)
        y = add_bullet_block(ax, 0.10, y, [
            "\u2022  118 k train / 14.8 k val events,  128 PUPPI candidates \u00d7 9 features",
            "\u2022  Targets: gen-level MET (px, py), normalized by factor 100",
        ], fontsize=BODY_FONTSIZE, spacing=0.04)

        y -= 0.02
        y = add_bullet_block(ax, 0.08, y, [
            "Search Space  (25 configurations)",
        ], fontsize=15, spacing=0.045, color=COLOR_BLUE)
        y = add_bullet_block(ax, 0.10, y, [
            "\u2022  Width: {32, 64}     Depth: {2, 3, 4}     Embeddings: {yes, no}",
            "\u2022  Mode 0: direct MET regression   |   Mode 1: per-particle weighting",
            "\u2022  Activation: {ReLU, ELU}     BinnedDeviation weight: {100, 200}",
        ], fontsize=BODY_FONTSIZE, spacing=0.04)

        y -= 0.02
        y = add_bullet_block(ax, 0.08, y, [
            "Training",
        ], fontsize=15, spacing=0.045, color=COLOR_BLUE)
        y = add_bullet_block(ax, 0.10, y, [
            "\u2022  Loss = MAE + MSE + w \u00b7 BinnedDeviation  (physics-informed pt-bin asymmetry)",
            "\u2022  30 epochs, AdamW lr=1e-3, early stopping patience 10",
        ], fontsize=BODY_FONTSIZE, spacing=0.04)

        save_slide(fig, pdf)

        # ==================================================================
        # SLIDE 3 -- Mode comparison
        # ==================================================================
        fig = plt.figure(figsize=FIGSIZE)
        ax_title = fig.add_axes([0, 0.88, 1, 0.12])
        ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1); ax_title.axis("off")
        ax_title.text(0.5, 0.5, "Mode 0 vs Mode 1: Per-Particle Weighting Wins Decisively",
                      ha="center", va="center", fontsize=TITLE_FONTSIZE, fontweight="bold",
                      color="#222222")

        mode0_rows = where(data, mode=0)
        mode1_rows = where(data, mode=1)

        ax1 = fig.add_axes([0.08, 0.15, 0.40, 0.65])
        ax2 = fig.add_axes([0.56, 0.15, 0.40, 0.65])

        for ax_sub, metric, label in [(ax1, "val_loss", "Validation Loss"),
                                       (ax2, "val_mae", "Validation MAE")]:
            m0 = col(mode0_rows, metric)
            m1 = col(mode1_rows, metric)

            positions = [0, 1]
            bp = ax_sub.boxplot([m0, m1], positions=positions, widths=0.5,
                                patch_artist=True, showmeans=True,
                                meanprops=dict(marker="D", markerfacecolor="white",
                                               markeredgecolor="black", markersize=6))
            bp["boxes"][0].set_facecolor(COLOR_LIGHT_ORANGE)
            bp["boxes"][1].set_facecolor(COLOR_LIGHT_BLUE)

            # Overlay points
            for i, (vals, clr) in enumerate([(m0, COLOR_ORANGE), (m1, COLOR_BLUE)]):
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                ax_sub.scatter(np.full_like(vals, float(i)) + jitter, vals, color=clr,
                               alpha=0.6, s=30, zorder=5)

            ax_sub.set_xticks(positions)
            ax_sub.set_xticklabels(["Mode 0\n(direct)", "Mode 1\n(per-particle)"],
                                    fontsize=BODY_FONTSIZE)
            ax_sub.set_ylabel(label, fontsize=BODY_FONTSIZE)
            ax_sub.grid(axis="y", alpha=0.3)

            if metric == "val_loss":
                best0 = m0.min()
                worst1 = m1.max()
                ax_sub.axhline(best0, color=COLOR_ORANGE, ls="--", alpha=0.5, lw=1)
                ax_sub.axhline(worst1, color=COLOR_BLUE, ls="--", alpha=0.5, lw=1)
                gap_text = f"Best Mode 0: {best0:.1f}\nWorst Mode 1: {worst1:.1f}"
                ax_sub.text(0.98, 0.98, gap_text, transform=ax_sub.transAxes,
                            ha="right", va="top", fontsize=SMALL_FONTSIZE,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                      edgecolor="#cccccc"))

        fig.text(0.5, 0.04,
                 "Physics insight: per-particle weighting exploits MET = \u2013\u03a3(w \u00b7 p) structure "
                 "\u2192 ALL mode-1 configs outperform ALL mode-0 configs",
                 ha="center", fontsize=BODY_FONTSIZE, style="italic", color="#555555")

        pdf.savefig(fig)
        plt.close(fig)

        # ==================================================================
        # SLIDE 4 -- Width x Depth heatmaps (mode 1 only)
        # ==================================================================
        fig = plt.figure(figsize=FIGSIZE)
        ax_title = fig.add_axes([0, 0.88, 1, 0.12])
        ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1); ax_title.axis("off")
        ax_title.text(0.5, 0.5,
                      "Width \u00d7 Depth: Validation MAE for Mode-1 Models (bw=200, ReLU)",
                      ha="center", va="center", fontsize=TITLE_FONTSIZE, fontweight="bold",
                      color="#222222")

        mode1_relu_bw200 = [r for r in data
                            if r["mode"] == 1 and r["activation"] == "relu"
                            and r["binned_weight"] == 200.0]
        widths = [32, 64]
        depths = [2, 3, 4]

        def build_grid(rows):
            """Build a 2D array [width_idx][depth_idx] of val_mae."""
            grid = np.full((len(widths), len(depths)), np.nan)
            for r in rows:
                wi = widths.index(r["width"]) if r["width"] in widths else None
                di = depths.index(r["depth"]) if r["depth"] in depths else None
                if wi is not None and di is not None:
                    grid[wi, di] = r["val_mae"]
            return grid

        noemb_rows = [r for r in mode1_relu_bw200 if not r["use_embeddings"]]
        emb_rows = [r for r in mode1_relu_bw200 if r["use_embeddings"]]

        def make_heatmap(ax_h, grid, title_text):
            vmin, vmax = 0.30, 0.85
            im = ax_h.imshow(grid, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
            ax_h.set_xticks(range(len(depths)))
            ax_h.set_xticklabels(depths, fontsize=BODY_FONTSIZE)
            ax_h.set_yticks(range(len(widths)))
            ax_h.set_yticklabels(widths, fontsize=BODY_FONTSIZE)
            ax_h.set_xlabel("Depth (# layers)", fontsize=BODY_FONTSIZE)
            ax_h.set_ylabel("Width", fontsize=BODY_FONTSIZE)
            ax_h.set_title(title_text, fontsize=SUBTITLE_FONTSIZE, pad=10)

            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    val = grid[i, j]
                    if np.isnan(val):
                        ax_h.text(j, i, "N/A", ha="center", va="center",
                                  fontsize=BODY_FONTSIZE, color="#999999")
                    else:
                        clr = "white" if val > 0.6 else "black"
                        ax_h.text(j, i, f"{val:.3f}", ha="center", va="center",
                                  fontsize=14, fontweight="bold", color=clr)
            return im

        ax_left = fig.add_axes([0.08, 0.18, 0.36, 0.58])
        ax_right = fig.add_axes([0.56, 0.18, 0.36, 0.58])

        make_heatmap(ax_left, build_grid(noemb_rows), "No Embeddings")
        im = make_heatmap(ax_right, build_grid(emb_rows), "With Embeddings")

        cbar_ax = fig.add_axes([0.08, 0.08, 0.84, 0.03])
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Validation MAE")

        fig.text(0.5, 0.03,
                 "Depth (3\u20134 layers) matters more than width for MET reconstruction",
                 ha="center", fontsize=BODY_FONTSIZE, style="italic", color="#555555")

        pdf.savefig(fig)
        plt.close(fig)

        # ==================================================================
        # SLIDE 5 -- Top 10 table
        # ==================================================================
        fig, ax = new_slide("Top 10 Configurations by Validation Loss")

        sorted_data = sorted(data, key=lambda r: r["val_loss"])
        top10 = sorted_data[:10]

        col_labels = ["Rank", "Config", "Emb", "W", "D", "Params",
                       "val_loss", "val_mae", "val_mse", "val_bd"]
        col_widths_tbl = [0.04, 0.24, 0.04, 0.04, 0.04, 0.06, 0.08, 0.08, 0.08, 0.08]

        x0 = 0.04
        y0 = 0.82
        row_h = 0.055
        header_color = COLOR_BLUE

        # Header
        cx = x0
        for lbl, w in zip(col_labels, col_widths_tbl):
            ax.text(cx + w / 2, y0, lbl, ha="center", va="center",
                    fontsize=SMALL_FONTSIZE, fontweight="bold", color="white",
                    bbox=dict(boxstyle="square,pad=0.3", facecolor=header_color,
                              edgecolor="none"))
            cx += w

        highlight_ranks = {1, 3}

        for i, row in enumerate(top10):
            y = y0 - (i + 1) * row_h
            rank = i + 1
            vals = [
                str(rank),
                row["config_name"].replace("_relu", "").replace("_bw", "/bw"),
                "Y" if row["use_embeddings"] else "N",
                str(row["width"]),
                str(row["depth"]),
                f"{row['param_count']:,}",
                f"{row['val_loss']:.2f}",
                f"{row['val_mae']:.3f}",
                f"{row['val_mse']:.3f}",
                f"{row['val_binned_deviation']:.4f}",
            ]

            bg = "#e8f0fe" if rank in highlight_ranks else ("white" if i % 2 == 0 else "#f5f5f5")
            cx = x0
            for val, w in zip(vals, col_widths_tbl):
                ax.text(cx + w / 2, y, val, ha="center", va="center",
                        fontsize=9,
                        bbox=dict(boxstyle="square,pad=0.25", facecolor=bg, edgecolor="#dddddd"))
                cx += w

        ax.text(0.04, 0.20,
                "\u2605  Rank 1: Best overall accuracy (emb/w64/d4/m1/bw100)",
                fontsize=BODY_FONTSIZE, color=COLOR_BLUE)
        ax.text(0.04, 0.15,
                "\u2605  Rank 3: Best FPGA-friendly (noemb/w32/d3/m1/bw200, only 2,721 params)",
                fontsize=BODY_FONTSIZE, color=COLOR_BLUE)

        save_slide(fig, pdf)

        # ==================================================================
        # SLIDE 6 -- Pareto plot
        # ==================================================================
        fig = plt.figure(figsize=FIGSIZE)
        ax_title = fig.add_axes([0, 0.88, 1, 0.12])
        ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1); ax_title.axis("off")
        ax_title.text(0.5, 0.5, "Accuracy vs Model Size: Pareto Front",
                      ha="center", va="center", fontsize=TITLE_FONTSIZE, fontweight="bold",
                      color="#222222")

        ax_p = fig.add_axes([0.10, 0.12, 0.82, 0.72])

        marker_map = {True: "^", False: "o"}
        color_map = {0: COLOR_ORANGE, 1: COLOR_BLUE}

        for mode_val in [0, 1]:
            for emb_val in [True, False]:
                subset = where(data, mode=mode_val, use_embeddings=emb_val)
                if not subset:
                    continue
                lbl = f"Mode {mode_val}, {'emb' if emb_val else 'noemb'}"
                ax_p.scatter(col(subset, "param_count"), col(subset, "val_mae"),
                             c=color_map[mode_val], marker=marker_map[emb_val],
                             s=80, alpha=0.7, edgecolors="black", linewidth=0.5,
                             label=lbl, zorder=5)

        # Pareto front (mode 1 only, lower-left is better)
        m1_sorted = sorted(mode1_rows, key=lambda r: r["param_count"])
        pareto_params = []
        pareto_mae = []
        best_mae = float("inf")
        for r in m1_sorted:
            if r["val_mae"] < best_mae:
                best_mae = r["val_mae"]
                pareto_params.append(r["param_count"])
                pareto_mae.append(r["val_mae"])

        ax_p.plot(pareto_params, pareto_mae, "k--", alpha=0.4, linewidth=1.5,
                  label="Pareto front (mode 1)", zorder=3)

        # Label Pareto-optimal points
        for pc, mae in zip(pareto_params, pareto_mae):
            matching = [r for r in data
                        if r["param_count"] == pc and abs(r["val_mae"] - mae) < 1e-6]
            if matching:
                short = matching[0]["config_name"].replace("_relu", "").replace("_bw", "/bw")
                ax_p.annotate(short, (pc, mae), textcoords="offset points",
                              xytext=(8, 8), fontsize=8, color="#333333",
                              arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))

        # Annotate recommended models
        best_acc = [r for r in data if r["config_name"] == "emb_w64_d4_m1_relu_bw100"][0]
        best_fpga = [r for r in data if r["config_name"] == "noemb_w32_d3_m1_relu_bw200"][0]

        for r, txt, clr in [
            (best_acc, "Best accuracy\n14,369 params", "#2ca02c"),
            (best_fpga, "Best FPGA\n2,721 params", "#d62728"),
        ]:
            ax_p.annotate(
                txt, (r["param_count"], r["val_mae"]),
                textcoords="offset points", xytext=(-60, -35),
                fontsize=SMALL_FONTSIZE, fontweight="bold", color=clr,
                arrowprops=dict(arrowstyle="->", color=clr, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=clr, alpha=0.9),
            )

        ax_p.set_xlabel("Parameter Count", fontsize=BODY_FONTSIZE)
        ax_p.set_ylabel("Validation MAE", fontsize=BODY_FONTSIZE)
        ax_p.legend(fontsize=SMALL_FONTSIZE, loc="upper right")
        ax_p.grid(alpha=0.3)
        ax_p.set_xscale("log")

        pdf.savefig(fig)
        plt.close(fig)

        # ==================================================================
        # SLIDE 7 -- Summary & Next Steps
        # ==================================================================
        fig, ax = new_slide("Summary & Next Steps")
        y = 0.82

        y = add_bullet_block(ax, 0.06, y, [
            "Key Findings",
        ], fontsize=16, spacing=0.05, color=COLOR_BLUE)
        y = add_bullet_block(ax, 0.08, y, [
            "\u2022  Per-particle weighting (mode 1) strongly outperforms direct regression",
            "\u2022  BinnedDeviation weight 100 > 200 (physics term was over-weighted at 200)",
            "\u2022  Depth (3\u20134 layers) matters more than width for MET reconstruction",
            "\u2022  Embeddings provide modest gains; dispensable for FPGA if needed",
        ], fontsize=BODY_FONTSIZE, spacing=0.042)

        y -= 0.03
        y = add_bullet_block(ax, 0.06, y, [
            "Recommended Models",
        ], fontsize=16, spacing=0.05, color=COLOR_BLUE)
        y = add_bullet_block(ax, 0.08, y, [
            "\u2022  Best accuracy:   emb / w64 / d4 / m1 / bw100  \u2014  14,369 params,  val_mae = 0.383",
            "\u2022  FPGA-optimized:  noemb / w32 / d3 / m1 / bw200  \u2014  2,721 params,  val_mae = 0.354",
        ], fontsize=BODY_FONTSIZE, spacing=0.042)

        y -= 0.03
        y = add_bullet_block(ax, 0.06, y, [
            "Next Steps",
        ], fontsize=16, spacing=0.05, color=COLOR_BLUE)
        y = add_bullet_block(ax, 0.08, y, [
            "\u2022  Longer training runs (100+ epochs) for top candidates",
            "\u2022  Fine-tune BinnedDeviation weight in range 50\u2013150",
            "\u2022  Quantization studies with QKeras / HGQ2",
            "\u2022  HLS4ML synthesis and FPGA resource estimates",
            "\u2022  Comparison with baseline (PuppiMET, firmware MET)",
        ], fontsize=BODY_FONTSIZE, spacing=0.042)

        save_slide(fig, pdf)

    print(f"PDF written to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
