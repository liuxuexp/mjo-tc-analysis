"""03 — Figure 3: regional landfall rate of China-landfalling TCs by
paired MJO phase, overall and by LMI category (Methods residence-time test).

(a) All China-landfalling TCs; (b) Weak, (c) Moderate, (d) Super TCs by LMI.
For each phase pair, bars show the phase-day-normalized landfall RATE per region
(South/East/North) as a RATIO to the no-modulation expectation (1.0 = the rate
expected from MJO residence time alone), using lib/mjo.residence_ratio_test
(multinomial active-day null, Methods: 1000 resamples, 5-95th pct, alpha = 0.10).
Storms hitting >1 region use the landfall-point rule. Every bar prints its count
n; error bars are the residence-time null band; asterisks mark alpha = 0.10.
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
from lib import mjo as mj
from lib import plot_style as P

REGION_COLORS = {'South China': '#2E7D32', 'East China': '#1976D2',
                 'North China': '#7B1FA2'}
YLIM_LO, YLIM_HI = 0.0, 3.0      # ratio frame; ticks every 0.5, 1 decimal
LEGEND_FS = 22          # legend text (larger than axis HOUSE_FS=18)
NDAYS = mj.active_days_per_group(months=C.SEASON_MONTHS)


def regional_panel(ax, df, title, is_bottom):
    """Grouped bars: residence-time-normalized regional landfall rate ratio.

    With shared axes (sharex/sharey), only the bottom row prints x-tick labels;
    the y-axis label is shared figure-wide via fig.supylabel (set in main), so
    panels don't repeat axis text.  is_bottom controls the x-tick-label gating.
    """
    x = np.arange(len(C.GROUP_KEY)); w = 0.26
    for ri, region in enumerate(C.REGION_ORDER):
        obs = {g: int(((df['group_landfall'] == g) &
                       (df['landfall_region'] == region)).sum()) for g in C.GROUP_KEY}
        res = mj.residence_ratio_test(obs, NDAYS, rng=np.random.default_rng(20240601))
        ratios = [res[g]['ratio'] for g in C.GROUP_KEY]
        los = [max(0.0, res[g]['ratio'] - res[g]['ci_lo']) for g in C.GROUP_KEY]
        his = [max(0.0, res[g]['ci_hi'] - res[g]['ratio']) for g in C.GROUP_KEY]
        his = [max(0.0, min(h, YLIM_HI - ratios[i] - 0.02)) for i, h in enumerate(his)]
        ax.bar(x + (ri - 1) * w, ratios, w, yerr=[los, his], capsize=3,
               color=REGION_COLORS[region], edgecolor='black', lw=0.5, label=region,
               error_kw=dict(alpha=0.8, elinewidth=2))
        for j, g in enumerate(C.GROUP_KEY):
            xc = x[j] + (ri - 1) * w
            top = max(res[g]['ratio'], min(res[g]['ci_hi'], YLIM_HI))
            # count above the error-bar top. fig3-local: count font = HOUSE_FS
            ax.text(xc, top + 0.03, f"{res[g]['n']}", ha='center', va='bottom',
                    fontsize=P.HOUSE_FS)
            if res[g]['stars']:
                # star above the count; offset widened 0.10 -> 0.22 so the enlarged
                # 18pt count clears the star
                ax.text(xc, top + 0.22, res[g]['stars'], ha='center', va='bottom',
                        fontsize=P.STAR_FS, fontweight='bold')
    ax.axhline(1.0, color='gray', ls='--', lw=1.0, alpha=0.8, zorder=1)
    ax.set_xticks(x)
    if is_bottom:
        # sharex shares the tick formatter; labels on bottom row only
        ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY],
                           fontsize=P.HOUSE_FS)
        ax.tick_params(axis='x', labelbottom=True)
    else:
        ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_title(title, fontsize=P.HOUSE_FS, fontweight='bold')
    ax.set_ylim(YLIM_LO, YLIM_HI)
    ax.set_yticks(np.arange(YLIM_LO, YLIM_HI + 0.001, 0.5))


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)

    fig, axs = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    axs[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    regional_panel(axs[0, 0], df, f'(a) All landfalling TCs (n={len(df)})',
                   is_bottom=False)
    for k, cat in enumerate(C.CATEGORY_ORDER):
        sub = df[df['lmi_category'] == cat]
        flat_idx = k + 1                       # 1,2,3 -> panels (b),(c),(d)
        regional_panel(axs.flat[flat_idx], sub,
                       f'({chr(98 + k)}) {C.CATEGORY_FULL[cat]} (LMI, n={len(sub)})',
                       is_bottom=flat_idx >= 2)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(handles, labels, loc='upper left', ncol=1,
                     fontsize=LEGEND_FS, frameon=False)
    fig.supylabel('Landfalls / active day  (ratio to no-modulation)',
                  fontsize=P.HOUSE_FS)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08)
    tot = {r: int((df['landfall_region'] == r).sum()) for r in C.REGION_ORDER}
    print("NDAYS:", NDAYS)
    print("regional landfall totals:", tot)
    P.save(fig, 'fig3-regional_mjo.png')


if __name__ == "__main__":
    main()
