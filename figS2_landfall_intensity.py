"""Figure S2 — comparison of LMI and intensity at first Chinese landfall.
Companion to Fig 2 (whose symbols encode LMI).

(a) Contingency counts of LMI (rows) x intensity-at-landfall (columns), with the
    'no-change' diagonal highlighted.
(b) Regional landfall proportions (South/East/North) by INTENSITY-AT-LANDFALL
    category (rather than LMI).
(c) Storm-level event-total TCP distributions by intensity-at-landfall category
    (median, 90% CI, n). TD-at-landfall (cat 0) is included so the cross-tab covers
    all storms; rows sum to 100%.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from lib import bootstrap as bs
from lib import plot_style as P

TITLE_FS = 22


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso']].reset_index(drop=True)
    lab = {0: 'TD', 1: 'Weak', 2: 'Moderate', 3: 'Super'}
    TICK_FS = 15   # figS2 multi-panel: ticks smaller than the 18pt house style to declutter

    fig = plt.figure(figsize=(18, 5.5))

    # (a) contingency heatmap: LMI (rows) x intensity at landfall (cols)
    from matplotlib.patches import Patch
    ax = fig.add_subplot(1, 3, 1)
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
                    fontsize=P.BARVAL_FS, fontweight='bold')
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
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)   # colorbar tight to the plot
    cb.set_label('n storms', fontsize=P.BARVAL_FS, labelpad=-10)   # vertical label, tight to the colorbar
    cb.ax.tick_params(labelsize=P.BARVAL_FS, pad=0)   # numbers hug the colorbar
    ax.set_title('(a) LMI × intensity at landfall', fontsize=TITLE_FS, fontweight='bold')
    ax.legend(handles=[Patch(fc='none', ec='red', lw=2, label='no change')],
              loc='upper right', frameon=False, fontsize=P.BARVAL_FS)

    # (b) regional proportions by landfall-intensity
    ax = fig.add_subplot(1, 3, 2); ax_b = ax
    overall = df['landfall_region'].value_counts(normalize=True).to_dict()
    x = np.arange(3); w = 0.26; bw = 0.20   # bw < w -> fig1-(b) gap within each trio
    for ri, region in enumerate(C.REGION_ORDER):
        pcts, los, his, ns = [], [], [], []
        for cat in [1, 2, 3]:
            sub = df[df['landfall_wind_category'] == cat]
            tot = len(sub); succ = int((sub['landfall_region'] == region).sum())
            ov = overall.get(region, 0)
            obs, sig, lo, hi = bs.proportion_ci(succ, tot, ov)
            pcts.append(obs * 100); los.append(max(0, obs - lo) * 100)
            his.append((hi - obs) * 100); ns.append(succ)
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
    ax = fig.add_subplot(1, 3, 3); ax_c = ax
    data, labels = [], []
    for cat in [1, 2, 3]:
        v = df[df['landfall_wind_category'] == cat]['tcp_total'].values
        data.append(v); labels.append(f'{lab[cat]}\nn={len(v)}')
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.55)
    for patch, cat in zip(bp['boxes'], [1, 2, 3]):
        patch.set_facecolor(C.COLOR_MAP[lab[cat]]); patch.set_alpha(0.45)
    rng = np.random.default_rng(3)
    for i, cat in enumerate([1, 2, 3]):
        v = df[df['landfall_wind_category'] == cat]['tcp_total'].values
        ax.scatter(np.full(len(v), i + 1) + rng.uniform(-0.1, 0.1, len(v)), v,
                   s=8, alpha=0.5, color=C.COLOR_MAP[lab[cat]])
        m = bs.summarize_median(v)
        ax.errorbar(i + 1, m['median'],
                    yerr=[[max(0, m['median'] - m['ci_lo'])], [m['ci_hi'] - m['median']]],
                    fmt='D', color='black', ms=6, capsize=3, zorder=5)
    ax.set_yscale('log'); ax.set_ylabel('Event-total TCP (10$^6$ m$^3$)', fontsize=P.HOUSE_FS, labelpad=-3)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.set_title('(c) Event-total TCP', fontsize=TITLE_FS, fontweight='bold')

    plt.tight_layout(w_pad=0.1)   # reduce horizontal gap between subplots
    # Close ONLY the (a)-(b) gap: shift panels b and c left together so the
    # (b)-(c) gap stays the same while (a)-(b) tightens.
    for _ax in (ax_b, ax_c):
        _box = _ax.get_position()
        _ax.set_position([_box.x0 - 0.012, _box.y0, _box.width, _box.height])
    P.save(fig, 'figS2-lmi_landfall_intensity.png')


if __name__ == "__main__":
    main()
