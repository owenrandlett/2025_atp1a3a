"""
Heatmap of deformation-based morphometry (DBM) results summarised across Z-Brain regions.

Takes the per-region "SignificantDeltaMedians ZBrain2Analysis" CSVs and plots the top
regions as a heatmap, using the same yellow (increased) / cyan (decreased) colour
scheme as the MAP-Mapping / DBM projections (see napari_viewMAPMap.py).

The plotted value is (Positive - Negative). Set METRIC below to choose between the
'Sum' columns (total significant signal in a region) and the 'Mean' columns, which are
Sum / region volume and therefore size-normalised.

Run from this directory:  python plot_DBM_region_heatmap.py
Outputs Figure_DBM_RegionHeatmap.svg/.png alongside the CSVs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, SymLogNorm
from matplotlib.ticker import FuncFormatter, NullFormatter

# ---------------------------------------------------------------- configuration
CSV_DIR = "."
COMPARISONS = [
    ("atp1a3aMUT_over_atp1a3aWT_SignificantDeltaMedians_ZBrain2Analysis.csv",  "$-/-$\nvs $+/+$"),
    ("atp1a3aHET_over_atp1a3aWT_SignificantDeltaMedians_ZBrain2Analysis.csv",  "$+/-$\nvs $+/+$"),
]
RANK_BY = 0        # index into COMPARISONS used to choose and order the top regions
N_REGIONS = 25        # enough to include the pallium, which ranks 23rd by |mean difference|

# "sum"  : total significant signal in the region. Reflects how much of a region changed,
#          so large regions with focal changes (e.g. the pallium) are represented.
# "mean" : Sum / region volume, i.e. size-normalised, comparable to the regional summaries
#          used by Moyer et al. Favours small structures with dense changes.
METRIC = "mean"

# "linear" shows effect sizes faithfully; because they span roughly two orders of
# magnitude, the largest regions (e.g. torus longitudinalis) then dominate the colour map
# and mid-ranked regions such as the pallium render as almost black. "symlog" keeps the
# same ordering but compresses that range so the smaller effects stay visible.
COLOUR_SCALE = "linear"

# Absolute value at which the colour scale saturates; None uses the largest effect in the
# data. Capping it keeps the mid-ranked regions readable on the linear scale, at the cost
# of the few largest regions (torus longitudinalis, superior raphe) being clipped.
VMAX = 150
OUT_BASE = "Figure_DBM_RegionHeatmap"

# yellow = increased volume in mutants, cyan = decreased, black = no significant signal,
# matching the additive display of the brain projections.
CMAP = LinearSegmentedColormap.from_list(
    "cyan_black_yellow", ["#00FFFF", "#000000", "#FFFF00"], N=512
)


def tidy(name):
    """'-Forebrain-Diencephalon-Thalamus-Ventral Thalamus' -> 'Ventral Thalamus (FB)'."""
    # ' - ' occurs inside region names (e.g. 'Raphe - Superior'); protect it before
    # splitting on the '-' used as the atlas hierarchy separator.
    protected = name.replace(" - ", "–")
    parts = [p.replace("–", " - ") for p in protected.split("-") if p]
    leaf = parts[-1]
    top = parts[0] if parts else ""
    short = {
        "Forebrain": "FB", "Midbrain": "MB", "Hindbrain": "HB",
        "Spinal Cord": "SC", "Ganglia": "Gg", "Retina": "Ret",
    }.get(top, top[:3])
    if leaf == top:
        return leaf
    return f"{leaf} ({short})"


def load(fname):
    d = pd.read_csv(os.path.join(CSV_DIR, fname), index_col=0)
    col = "Sum" if METRIC == "sum" else "Mean"
    return d[f"{col} Positive"] - d[f"{col} Negative"]


def colourbar_ticks(vmax, norm):
    """Zero, the decades the colour scale spans, and the rounded end points.

    Only meaningful on the symlog scale; on a linear one the default evenly spaced
    ticks are already round, so return None and leave the locator alone.
    """
    linthresh = getattr(norm, "linthresh", None)
    if linthresh is None:
        return None
    top = 10 ** int(np.floor(np.log10(vmax)))
    decades, d = [], top
    while d >= linthresh:
        decades.append(d)
        d /= 10
    end = np.round(vmax, -int(np.floor(np.log10(vmax))))
    if end > 1.5 * top:
        decades.insert(0, end)
    return [-t for t in decades] + [0] + decades[::-1]


def main():
    series = {label: load(f) for f, label in COMPARISONS}
    df = pd.DataFrame(series)

    rank_label = COMPARISONS[RANK_BY][1]
    order = df[rank_label].abs().sort_values(ascending=False).index[:N_REGIONS]
    df = df.loc[order]

    # split the selected regions into increases then decreases, each ordered by
    # effect size, so the two directions read as separate blocks
    rank = df[rank_label]
    up = rank[rank > 0].sort_values(ascending=False).index
    down = rank[rank <= 0].sort_values().index
    df = df.loc[list(up) + list(down)]
    n_up = len(up)

    scale = 1e6 if METRIC == "sum" else 1.0
    df = df / scale
    data_max = np.abs(df.values).max()
    vmax = VMAX if VMAX is not None else data_max
    if COLOUR_SCALE == "symlog":
        norm = SymLogNorm(linthresh=0.05 * vmax, vmin=-vmax, vmax=vmax, base=10)
    else:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    im = ax.imshow(df.values, cmap=CMAP, norm=norm, aspect="auto")
    # arrowheads on the colourbar where the largest effects run off the capped scale
    extend = "both" if data_max > vmax else "neither"

    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, fontsize=9, linespacing=1.3)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels([tidy(i) for i in df.index], fontsize=9)
    ax.tick_params(length=2, pad=1.5)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # thin separators so individual cells read as discrete regions
    ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", length=0)

    # divider between the increased and decreased blocks, with a label for each
    if 0 < n_up < df.shape[0]:
        ax.axhline(n_up - 0.5, color="black", linewidth=1.5)
    x_lab = df.shape[1] - 0.35
    for label, lo, hi in (("increased", 0, n_up), ("decreased", n_up, df.shape[0])):
        if hi > lo:
            ax.text(x_lab, (lo + hi - 1) / 2, label, rotation=90, fontsize=6,
                    va="center", ha="left")

    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.12, extend=extend)
    unit = "summed volume difference\n($\\times10^{6}$ A.U.)" if METRIC == "sum" \
        else "volume difference per unit\nregion volume (A.U.)"
    cb.set_label(f"{unit}\ncyan = decreased, yellow = increased", fontsize=6)
    # plain round numbers rather than the default 10^n labels of the symlog locator
    ticks = colourbar_ticks(vmax, norm)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)

    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(f"{OUT_BASE}.{ext}", dpi=600, bbox_inches="tight",
                    facecolor="white")
    print(f"wrote {OUT_BASE}.svg / .png")
    print(f"ranked by {rank_label}; colour scale +/- {vmax:.1f} "
          f"(largest effect {data_max:.1f})")


if __name__ == "__main__":
    main()
