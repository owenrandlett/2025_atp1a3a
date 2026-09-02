"""
Heatmap of deformation-based morphometry (DBM) results summarised across Z-Brain regions.

Takes the per-region "SignificantDeltaMedians ZBrain2Analysis" CSVs and plots the top
regions as a heatmap, using the same green (increased) / magenta (decreased) colour
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

# Effects span roughly two orders of magnitude, so on a linear scale the largest regions
# (e.g. torus longitudinalis) saturate the colour map and mid-ranked regions such as the
# pallium render as almost black. "symlog" keeps the ordering but makes those visible;
# "linear" is the faithful-but-less-legible alternative.
COLOUR_SCALE = "symlog"
OUT_BASE = "Figure_DBM_RegionHeatmap"

# green = increased volume in mutants, magenta = decreased, black = no significant signal,
# matching the additive display of the brain projections.
CMAP = LinearSegmentedColormap.from_list(
    "magenta_black_green", ["#FF00FF", "#000000", "#00FF00"], N=512
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


def main():
    series = {label: load(f) for f, label in COMPARISONS}
    df = pd.DataFrame(series)

    rank_label = COMPARISONS[RANK_BY][1]
    order = df[rank_label].abs().sort_values(ascending=False).index[:N_REGIONS]
    df = df.loc[order]

    scale = 1e6 if METRIC == "sum" else 1.0
    df = df / scale
    vmax = np.abs(df.values).max()
    if COLOUR_SCALE == "symlog":
        norm = SymLogNorm(linthresh=0.05 * vmax, vmin=-vmax, vmax=vmax, base=10)
    else:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(4.4, 6.2))
    im = ax.imshow(df.values, cmap=CMAP, norm=norm, aspect="auto")

    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, fontsize=7, linespacing=1.3)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels([tidy(i) for i in df.index], fontsize=6)
    ax.tick_params(length=2, pad=1.5)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # thin separators so individual cells read as discrete regions
    ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    unit = "summed volume difference\n($\\times10^{6}$ A.U.)" if METRIC == "sum" \
        else "volume difference per unit\nregion volume (A.U.)"
    cb.set_label(f"{unit}\nmagenta = decreased, green = increased", fontsize=6)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)

    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(f"{OUT_BASE}.{ext}", dpi=600, bbox_inches="tight",
                    facecolor="white")
    print(f"wrote {OUT_BASE}.svg / .png")
    print(f"ranked by {rank_label}; colour scale +/- {vmax:.1f}")


if __name__ == "__main__":
    main()
