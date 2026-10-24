"""Figure 3: regional landfall rate by MJO phase and LMI category.

Panels: (a) all China-landfalling TCs, (b) Weak, (c) Moderate, (d) Super.
Bars give the phase-day-normalized rate per region as a ratio to the
no-modulation expectation; each bar prints its count.

Error bars are the 2.5-97.5 percentile interval of the year-block permutation
null. Exact p and BH q for the 48 comparisons are in Table S1.
Output: fig/fig3-regional_mjo.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import config as C
from lib import plot_style as P

REGION_COLORS = {'South China': '#2E7D32', 'East China': '#1976D2',
                 'North China': '#7B1FA2'}
YLIM_LO, YLIM_HI = 0.0, 3.25     # frame headroom so count labels stay inside;
                                 # ticks every 0.5 up to 3.0 (1 decimal)
WHISK_CAP = 2.95        # null whiskers clip here (below the frame) leaving
                        # room above for the count label
N_FS = 20               # count labels (match the y-tick size)
LEGEND_FS = 22          # legend text (larger than axis HOUSE_FS=18)
TITLE_FS = 22
CAT_FULL = {0: 'All TCs', **{c: C.CATEGORY_FULL[c] for c in C.CATEGORY_ORDER}}


def regional_panel(ax, df, sample, stats, title, is_bottom):
    """Grouped bars: residence-normalized regional landfall rate ratio.

    Error bars = 2.5-97.5 percentile interval of the year-block permutation
    null (from r2_stats_pergrp.csv, family fig3). With shared axes
    (sharex/sharey), only the bottom row prints x-tick labels; the y-axis
    label is shared figure-wide via fig.supylabel (set in main).
    """
    x = np.arange(len(C.GROUP_KEY)); w = 0.26
    for ri, region in enumerate(C.REGION_ORDER):
        sub = stats[(stats['sample'] == sample) & (stats['stratum'] == region)]
        sub = sub.set_index('phase_group').reindex(C.GROUP_KEY)
        ratios = sub['est'].values
        # null-interval error bars, clamped to the panel frame
        los = [max(0.0, ratios[i] - sub['null_lo'].values[i])
               for i in range(4)]
        his = [max(0.0, min(sub['null_hi'].values[i] - ratios[i],
                            WHISK_CAP - ratios[i])) for i in range(4)]
        ax.bar(x + (ri - 1) * w, ratios, w, yerr=[los, his], capsize=3,
               color=REGION_COLORS[region], edgecolor='black', lw=0.5,
               label=region, error_kw=dict(alpha=0.8, elinewidth=2))
        for j, g in enumerate(C.GROUP_KEY):
            xc = x[j] + (ri - 1) * w
            top = max(ratios[j], min(sub['null_hi'].values[j], WHISK_CAP))
            # count above the error-bar top
            ax.text(xc, top + 0.05, f"{int(sub['n'].values[j])}",
                    ha='center', va='bottom', fontsize=N_FS)
    ax.axhline(1.0, color='gray', ls='--', lw=1.0, alpha=0.8, zorder=1)
    ax.set_xticks(x)
    if is_bottom:
        # sharex shares the tick formatter; labels on bottom row only
        ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY],
                           fontsize=P.HOUSE_FS)
        ax.tick_params(axis='x', labelbottom=True)
    else:
        ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold')
    ax.set_ylim(YLIM_LO, YLIM_HI)
    ax.set_yticks(np.arange(YLIM_LO, 3.001, 0.5))


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)
    stats = pd.read_csv(C.data_path("r2_stats_pergrp.csv"))
    stats = stats[stats['family'] == 'fig3']
    assert len(stats) == 48, f"fig3 family has {len(stats)} rows != 48"

    fig, axs = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    axs[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    n_by_sample = {0: len(df), **{c: int((df['lmi_category'] == c).sum())
                                  for c in C.CATEGORY_ORDER}}
    regional_panel(axs[0, 0], df, CAT_FULL[0], stats,
                   f'(a) All landfalling TCs (n={n_by_sample[0]})',
                   is_bottom=False)
    for k, cat in enumerate(C.CATEGORY_ORDER):
        flat_idx = k + 1                       # 1,2,3 -> panels (b),(c),(d)
        regional_panel(axs.flat[flat_idx], df[df['lmi_category'] == cat],
                       CAT_FULL[cat], stats,
                       f'({chr(98 + k)}) {C.CATEGORY_FULL[cat]} '
                       f'(LMI, n={n_by_sample[cat]})',
                       is_bottom=flat_idx >= 2)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    # legend nudged toward the panel corner
    axs[0, 0].legend(handles, labels, loc='upper left', ncol=1,
                     fontsize=LEGEND_FS, frameon=False, borderaxespad=0.1)
    fig.supylabel('Landfalls / active day  (ratio to no-modulation)',
                  fontsize=22)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.10)
    tot = {r: int((df['landfall_region'] == r).sum()) for r in C.REGION_ORDER}
    print("regional landfall totals:", tot)
    sig = stats[stats['q_fdr'] < 0.05]
    print(f"fig3: {len(sig)}/48 comparisons with q < 0.05"
          + (f": {list(zip(sig['sample'], sig['stratum'], sig['phase_group']))}"
             if len(sig) else ""))
    marg = stats[(stats['p_perm'] < 0.05) & (stats['q_fdr'] >= 0.05)]
    print(f"fig3: {len(marg)} comparisons with p < 0.05 but q >= 0.05")
    P.save(fig, 'fig3-regional_mjo.png')


if __name__ == "__main__":
    main()
