"""Reviewer figure: conceptual schematic for the sliced 2-D sort-match
extension (Theorem 2). Pedagogical; no real data used.

Left panel: a 1-D axis with blue predicted points and orange target
points, joined by gray sort-match pairing lines.

Right panel: 2-D scatter with the same points, with 3-4 dashed lines
through the origin/center indicating K random projection directions
plus short arrows showing that each 1-D projection yields a 1-D
matching sub-problem.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "2d" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GRAY = "#666666"


def main():
    rng = np.random.default_rng(7)
    n = 8

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))

    # ---- Left panel: 1-D sort-match ----
    ax = axes[0]
    pred_1d = rng.uniform(0.5, 9.5, n)
    targ_1d = pred_1d + rng.normal(0, 0.7, n)
    # sort both to visualize the sort-match pairing
    sp = np.sort(pred_1d)
    st = np.sort(targ_1d)
    for p, t in zip(sp, st):
        ax.plot([0, 1], [p, t], color=GRAY, lw=0.9, alpha=0.75, zorder=1)
    ax.scatter(
        np.zeros(n), pred_1d, s=70, c=BLUE, zorder=3,
        edgecolors="black", linewidths=0.5,
        label=r"predicted  $\hat\delta$",
    )
    ax.scatter(
        np.ones(n), targ_1d, s=70, c=ORANGE, marker="s", zorder=3,
        edgecolors="black", linewidths=0.5,
        label=r"target  $\delta^\star$",
    )
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.7, 10.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["predicted", "target"])
    ax.set_ylabel("shift (ppm)")
    ax.set_title("(a) 1-D sort-match\n(sort both, pair in order)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # ---- Right panel: sliced 2-D sort-match schematic ----
    ax = axes[1]
    # 2-D point clouds roughly co-located to emphasize matching problem
    pH = rng.uniform(1.2, 7.8, n)
    pC = rng.uniform(25, 145, n)
    tH = pH + rng.normal(0, 0.55, n)
    tC = pC + rng.normal(0, 7.0, n)

    center = np.array([4.5, 85.0])
    # Choose aspect-matched half-extent so lines look visually through "center"
    ext_h = 5.0
    ext_c = 85.0
    # K=4 projection directions
    K = 4
    rs = np.random.default_rng(13)
    # Use evenly spaced angles plus a small jitter for a "random" feel
    angles = np.linspace(0, np.pi, K, endpoint=False) + rs.uniform(-0.1, 0.1, K)
    proj_colors = ["#555555"] * K
    for k, theta in enumerate(angles):
        dx = np.cos(theta) * ext_h
        dy = np.sin(theta) * ext_c
        ax.plot(
            [center[0] - dx, center[0] + dx],
            [center[1] - dy, center[1] + dy],
            linestyle="--",
            color=proj_colors[k],
            lw=0.9,
            alpha=0.75,
            zorder=1,
        )
        # Small arrow near the tip labeling the direction u_k
        tip_x = center[0] + dx * 0.95
        tip_y = center[1] + dy * 0.95
        ax.annotate(
            f"$u_{k+1}$",
            xy=(tip_x, tip_y),
            xytext=(tip_x + 0.18, tip_y + 3),
            fontsize=8,
            color="#333333",
            arrowprops=dict(
                arrowstyle="->",
                color="#333333",
                lw=0.6,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    ax.scatter(
        pH, pC, s=70, c=BLUE, zorder=3,
        edgecolors="black", linewidths=0.5,
        label=r"predicted  $\hat P$",
    )
    ax.scatter(
        tH, tC, s=70, c=ORANGE, marker="s", zorder=3,
        edgecolors="black", linewidths=0.5,
        label=r"target  $P^\star$",
    )

    ax.set_xlim(0.2, 9.0)
    ax.set_ylim(5, 170)
    ax.set_xlabel(r"$^{1}$H shift (ppm)")
    ax.set_ylabel(r"$^{13}$C shift (ppm)")
    ax.set_title("(b) Sliced 2-D sort-match (Theorem 2)\n"
                 r"$K$ directions $\to$ $K$ 1-D matchings")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    fig.tight_layout()
    out_png = OUT / "fig_theorem2_schematic.png"
    out_pdf = OUT / "fig_theorem2_schematic.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
