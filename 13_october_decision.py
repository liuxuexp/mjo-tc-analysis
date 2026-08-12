"""13 — October decision analysis.

Computes October event-normalized TCP and compares it with Jun-Sep, to decide
whether Figure 10 (October) carries a defensible per-event signal or should be
moved to the Supporting Information only. Writes a small comparison figure
(data03/october_decision.txt with the recommendation).
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


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['active_landfall']].copy()
    jj = df[df['month'].between(6, 9)]
    oct_ = df[df['month'] == 10]

    s_jj = bs.summarize(jj['tcp_total'].values)
    s_oct = bs.summarize(oct_['tcp_total'].values)
    pd_ = bs.pct_diff_ci(oct_['tcp_total'].values, jj['tcp_total'].values)

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    # (a) mean event TCP Jun-Sep vs Oct
    ax = axs[0]
    labels = [f'Jun-Sep\nn={s_jj["n"]}', f'October\nn={s_oct["n"]}']
    means = [s_jj['mean'], s_oct['mean']]
    los = [s_jj['mean'] - s_jj['ci_lo'], max(0, s_oct['mean'] - s_oct['ci_lo'])]
    his = [s_jj['ci_hi'] - s_jj['mean'], s_oct['ci_hi'] - s_oct['mean']]
    ax.bar([0, 0.3], means, 0.15, yerr=[los, his], capsize=5,
           color=['#1976D2', '#EF6C00'], edgecolor='black', lw=0.5)
    ax.set_xticks([0, 0.3]); ax.set_xticklabels(labels, fontsize=P.HOUSE_FS)
    ax.set_xlim(-0.15, 0.45)   # ticks land at 1/4 and 3/4 -> equal end margins
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylabel('Mean event TCP (10$^6$ m$^3$)', fontsize=P.HOUSE_FS)
    ax.set_title('(a) Mean event-total TCP per storm', fontsize=P.HOUSE_FS, fontweight='bold')
    # (b) distributions
    ax = axs[1]
    data = [jj['tcp_total'].values, oct_['tcp_total'].values]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], ['#1976D2', '#EF6C00']):
        patch.set_facecolor(c); patch.set_alpha(0.45)
    rng = np.random.default_rng(11)
    for i, v in enumerate(data):
        ax.scatter(np.full(len(v), i + 1) + rng.uniform(-0.08, 0.08, len(v)), v,
                   s=8, alpha=0.5)
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylabel('Event-total TCP (10$^6$ m$^3$)', fontsize=P.HOUSE_FS)
    ax.set_title('(b) Storm-level distributions', fontsize=P.HOUSE_FS, fontweight='bold')
    fig.subplots_adjust(wspace=0.35)
    P.save(fig, 'figS10-october_tcp.png')

    # recommendation
    robust_increase = (pd_['pct'] > 0) and (pd_['ci_lo'] > 0)
    rec = ('KEEP in SI only' if not robust_increase else 'retain October-specific result')
    txt = (f"October decision analysis\n"
           f"  Jun-Sep : n={s_jj['n']}, mean event TCP={s_jj['mean']:.0f} (90% CI {s_jj['ci_lo']:.0f}-{s_jj['ci_hi']:.0f})\n"
           f"  October : n={s_oct['n']}, mean event TCP={s_oct['mean']:.0f} (90% CI {s_oct['ci_lo']:.0f}-{s_oct['ci_hi']:.0f})\n"
           f"  Oct-vs-JJAS mean: {pd_['pct']:+.1f}% (90% CI {pd_['ci_lo']:+.1f} to {pd_['ci_hi']:+.1f}%)\n"
           f"  Robust October increase (CI > 0): {robust_increase}\n"
           f"  Recommendation: {rec} — October circulation is presented in SI (Fig S8) "
           f"rather than the main Fig 9 composite, unless the above shows a robust "
           f"per-event increase.\n")
    print(txt)
    with open(C.data_path("october_decision.txt"), 'w') as f:
        f.write(txt)


if __name__ == "__main__":
    main()
