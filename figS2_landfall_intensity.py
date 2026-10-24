"""Figure S2: lifetime-maximum intensity against intensity at first landfall.

Panels:
  (a) Contingency counts of LMI by intensity at landfall, no-change diagonal
      highlighted.
  (b) Regional landfall proportions by intensity at landfall.
  (c) Event-total TCP by intensity at landfall, median with 95% year-block
      bootstrap interval and n.

Output: fig/figS2-lmi_landfall_intensity.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from lib import stats_yb as S
from lib import plot_style as P

TITLE_FS = 22


def _boot_prop(sub, region):
    """Year-block bootstrap 95% CI for the share of `sub` landfalling in
    `region` (resampling calendar years of the landfall dates)."""
    yi, _ = S.day_index(pd.to_datetime(sub['landfall_time']))
    n_y = C.YEAR_END - C.YEAR_START + 1
    multi = S.make_multisets(S.B_DEFAULT, n_years=n_y)
    succ = (sub['landfall_region'] == region).values.astype(float)
    cnt = multi @ np.bincount(yi, minlength=n_y)
    s = multi @ np.bincount(yi, weights=succ, minlength=n_y)
    prop = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    lo, hi = np.nanpercentile(prop, [2.5, 97.5])
    return float(succ.mean()), float(lo), float(hi)


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso']].reset_index(drop=True)
    lab = {0: 'TD', 1: 'Weak', 2: 'Moderate', 3: 'Super'}
    TICK_FS = 15   # figS2 multi-panel: ticks smaller than the 18pt house style to declutter
    CELL_FS = 20   # heatmap cell counts

    # Two-row layout: (a)(b) on top, (c) centered below; equal
    # widths via a 2x4 grid.
    fig = plt.figure(figsize=(12.0, 11.0))
    gs = fig.add_gridspec(2, 4)

    # (a) contingency heatmap: LMI (rows) x intensity at landfall (cols)
    from matplotlib.patches import Patch
    ax = fig.add_subplot(gs[0, :2])
    # columns = intensity AT landfall [0=TD, 1=Weak, 2=Moderate, 3=Super]; cat 0 (TD)
    # is included so the cross-tab covers all 464 storms.
    ct = pd.crosstab(df['lmi_category'], df['landfall_wind_category']).reindex(
        index=[1, 2, 3], columns=[0, 1, 2, 3], fill_value=0).values
    im = ax.imshow(ct, cmap='Blues', aspect='auto')
    for i in range(3):
        for j in range(4):
            n = ct[i, j]
            if n == 0:
                continue          # blank the (physically empty) intensification triangle
            ax.text(j, i, str(n), ha='center', va='center',
                    color='white' if n > ct.max() / 2 else 'black',
                    fontsize=CELL_FS, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'{lab[c]}\nat landfall' for c in [0, 1, 2, 3]], fontsize=TICK_FS)
    ax.set_yticks(range(3))
    ax.set_yticklabels([lab[c] for c in [1, 2, 3]], fontsize=18,
                       rotation=90, va='center')
    ax.tick_params(axis='both', labelsize=TICK_FS)
    # red outline on the no-change cells: landfall == LMI, i.e. col j == row i + 1
    # (row i -> LMI i+1; col j -> landfall j). Intensification (j > i+1) is physically
    # empty and left blank by the n==0 rule above.
    for i in range(3):
        ax.add_patch(plt.Rectangle((i + 1 - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', lw=2))
    cb = plt.colorbar(im, ax=ax, shrink=1.0, pad=0.02)   # full plot height
    # colorbar text: title at axis-title size, numbers at 16
    cb.set_label('n storms', fontsize=P.HOUSE_FS, labelpad=-6)
    cb.ax.tick_params(labelsize=16, pad=0)   # numbers hug the colorbar
    ax.set_title('(a) LMI × intensity at landfall', fontsize=TITLE_FS, fontweight='bold')
    ax.legend(handles=[Patch(fc='none', ec='red', lw=2, label='No change')],
              loc='upper right', frameon=False, fontsize=P.BARVAL_FS)

    # (b) regional proportions by landfall-intensity
    ax = fig.add_subplot(gs[0, 2:])
    x = np.arange(3); w = 0.26; bw = 0.20   # bw < w -> fig1-(b) gap within each trio
    for ri, region in enumerate(C.REGION_ORDER):
        pcts, los, his, ns = [], [], [], []
        for cat in [1, 2, 3]:
            sub = df[df['landfall_wind_category'] == cat]
            obs, lo, hi = _boot_prop(sub, region)
            ns.append(int((sub['landfall_region'] == region).sum()))
            pcts.append(obs * 100); los.append(max(0, obs - lo) * 100)
            his.append((hi - obs) * 100)
        ax.bar(x + (ri - 1) * w, pcts, bw, yerr=[los, his], capsize=3,
               color={'South China': '#2E7D32', 'East China': '#1976D2',
                      'North China': '#7B1FA2'}[region], edgecolor='black', lw=0.5,
               label=region, error_kw=dict(alpha=0.8, elinewidth=1))
    ax.set_xticks(x); ax.set_xticklabels([f'{lab[c]}\nat landfall' for c in [1, 2, 3]], fontsize=TICK_FS)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    ax.set_ylabel('Landfalls in region (%)', fontsize=P.HOUSE_FS, labelpad=-7)
    ax.set_title('(b) Regional landfall proportion', fontsize=TITLE_FS, fontweight='bold')
    ax.legend(fontsize=TICK_FS, frameon=False, loc='upper left'); ax.set_ylim(0, 100)

    # (c) event TCP by landfall-intensity
    ax = fig.add_subplot(gs[1, 1:3])
    data, labels = [], []
    for cat in [1, 2, 3]:
        v = df[df['landfall_wind_category'] == cat]['tcp_total'].values
        data.append(v); labels.append(f'{lab[cat]} (n={len(v)})')
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.55)
    for patch, cat in zip(bp['boxes'], [1, 2, 3]):
        patch.set_facecolor(C.COLOR_MAP[lab[cat]]); patch.set_alpha(0.45)
    rng = np.random.default_rng(3)
    for i, cat in enumerate([1, 2, 3]):
        sub = df[df['landfall_wind_category'] == cat]
        v = sub['tcp_total'].values
        ax.scatter(np.full(len(v), i + 1) + rng.uniform(-0.1, 0.1, len(v)), v,
                   s=8, alpha=0.5, color=C.COLOR_MAP[lab[cat]])
        m = S.boot_scalar_stats(v, pd.to_datetime(sub['landfall_time']))
        ax.errorbar(i + 1, m['median'],
                    yerr=[[max(0, m['median'] - m['median_lo'])],
                          [m['median_hi'] - m['median']]],
                    fmt='D', color='black', ms=6, capsize=3, zorder=5)
    ax.set_yscale('log')
    # y-axis title kept clear of the tick labels
    ax.set_ylabel('Event-total TCP (10$^6$ m$^3$)', fontsize=P.HOUSE_FS, labelpad=8)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.set_title('(c) Event-total TCP', fontsize=TITLE_FS, fontweight='bold')

    plt.tight_layout(w_pad=0.1)   # reduce horizontal gap between subplots
    # tight_layout + the (a) colorbar leaves no room above the top row and the
    # (a) title gets sliced by the canvas edge — hold explicit top headroom.
    fig.subplots_adjust(top=0.94)
    P.save(fig, 'figS2-lmi_landfall_intensity.png')


if __name__ == "__main__":
    main()
