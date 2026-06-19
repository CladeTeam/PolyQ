#!/usr/bin/env python3
"""Plot delta logP(ref) from Base — only Rice & Chlamy (balanced)."""
import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", required=True,
                        help="Path to polyQ_summary_all_models.csv")
    parser.add_argument("--output_dir", default="figures",
                        help="Output directory for figure")
    args = parser.parse_args()

    FDIR = Path(args.output_dir)
    FDIR.mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(args.summary_csv)
    base = s[s["model"] == "base"]
    q_vals = base["q_length"].values

    MODELS = ["rice_balanced", "chlamydomonas_balanced"]
    COLORS = {"rice_balanced": "#1b9e77", "chlamydomonas_balanced": "#9467bd"}
    LABELS = {"rice_balanced": "Rice (16 ep, balanced)",
              "chlamydomonas_balanced": "Chlamy (44 ep, balanced)"}

    fig, ax = plt.subplots(figsize=(11, 5.5))

    all_deltas = []
    for m in MODELS:
        sub = s[s["model"] == m]
        if sub.empty:
            print(f"WARNING: model {m} not found in {args.summary_csv}")
            continue
        delta = sub["mean_pll"].values - base["mean_pll"].values
        all_deltas.append(delta)
        # raw (faint)
        ax.plot(q_vals, delta, color=COLORS[m], linewidth=1.0, alpha=0.3,
                marker=".", markersize=1.5)
        # smoothed (bold)
        smooth = uniform_filter1d(delta, size=5)
        ax.plot(q_vals, smooth, color=COLORS[m], linewidth=2.5, alpha=0.95,
                label=LABELS[m])

    if all_deltas:
        # peak region
        avg_abs = np.mean(np.abs(all_deltas), axis=0)
        peak_q = q_vals[np.argmax(avg_abs)]
        peak_abs = np.max(avg_abs)

        ax.axvspan(20, 32, alpha=0.08, color="crimson", zorder=0)
        ax.axvline(peak_q, color="darkred", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.annotate(f"Peak deviation at\nQ$\\approx${int(peak_q)} (|$\\Delta$| = {peak_abs:.4f})",
                    xy=(peak_q, peak_abs * 0.55),
                    xytext=(peak_q + 10, peak_abs * 0.82),
                    fontsize=10, fontweight="bold", color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax.annotate("Q $\\approx$ 20–32: maximal divergence\n"
                "between species-adapted models\n"
                "and Base ESM2. This intermediate\n"
                "polyQ range may represent a\n"
                "transition zone where species-\n"
                "specific proteome context exerts\n"
                "the largest influence on sequence\n"
                "likelihood.",
                xy=(25, -0.017), fontsize=8.5, color="#333",
                bbox=dict(boxstyle="round", facecolor="lightyellow",
                          alpha=0.85, edgecolor="#bbb"))

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xlabel("PolyQ Length", fontsize=13)
    ax.set_ylabel("$\Delta$ logP(ref) from Base Model", fontsize=13)
    ax.set_title("Species-Adapted Model Deviation from Base ESM2\n"
                 "Shaded: Q $\\approx$ 20–32 max divergence region",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right", frameon=True)
    ax.set_xlim(8, 74)

    fig.tight_layout()
    out = FDIR / "fig_delta_clean_rice_chlamy.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
