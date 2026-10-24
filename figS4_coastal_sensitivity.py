"""Figure S4: coastal and inland event-total TCP at 100, 200 and 300 km.

Companion to Figure 4h, which uses the 200-km split. Mean event-total TCP is
shown for Weak, Moderate and Super storms at each threshold, with 95%
year-block bootstrap intervals. Units: 10^9 m^3.
Output: fig/figS4-coastal_sensitivity.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config as C
from lib import tcp as tcp_lib
from lib import stats_yb as S
from lib import plot_style as P

THRESHOLDS = [100.0, 200.0, 300.0]
TICK_FS = 20     # ticks, titles, y-axis labels
VAL_FS = 16      # bar-top values


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)
    codes = [f"{c:04d}" for c in df['chinese_code']]
    cats = df['lmi_category'].values
    dates = pd.to_datetime(df['landfall_time'])

    # per-storm coastal/inland totals at each threshold (one field read per storm)
    per = {thr: {'coastal': {c: [] for c in C.CATEGORY_ORDER},
                 'inland': {c: [] for c in C.CATEGORY_ORDER}} for thr in THRESHOLDS}
    per_d = {thr: {'coastal': {c: [] for c in C.CATEGORY_ORDER},
                   'inland': {c: [] for c in C.CATEGORY_ORDER}}
             for thr in THRESHOLDS}
    n_ok = 0
    for k, (code, cat) in enumerate(zip(codes, cats)):
        if cat not in C.CATEGORY_ORDER:
            continue
        try:
            res = tcp_lib.coastal_inland_by_threshold(code, THRESHOLDS)
        except Exception:
            continue
        n_ok += 1
        for thr in THRESHOLDS:
            coast, inl = res[thr]
            per[thr]['coastal'][cat].append(coast)
            per[thr]['inland'][cat].append(inl)
            per_d[thr]['coastal'][cat].append(dates.iloc[k])
            per_d[thr]['inland'][cat].append(dates.iloc[k])

    # summarize (mean + 95% year-block bootstrap CI, 10^9 m^3) and y ceiling
    sums = {thr: {} for thr in THRESHOLDS}
    ymax = 0.0
    for thr in THRESHOLDS:
        for zone in ('coastal', 'inland'):
            for cat in C.CATEGORY_ORDER:
                arr = np.array(per[thr][zone][cat], dtype=float)
                if arr.size:
                    s = S.boot_scalar_stats(arr, pd.Series(per_d[thr][zone][cat]))
                else:
                    s = dict(mean=0.0, mean_lo=0.0, mean_hi=0.0, n=0)
                sums[thr][(zone, cat)] = s
                ymax = max(ymax, s['mean_hi'] / 1e3)

    # Two-row layout: (a)(b) on top, (c) centered below; equal
    # widths via a 2x4 grid. y scale shared across all three panels — tick
    # labels on (a) and on (c) ((c) sits alone in its row), hidden on (b).
    fig = plt.figure(figsize=(10.0, 10.0))
    gs = fig.add_gridspec(2, 4)
    ax0 = fig.add_subplot(gs[0, :2])
    axs = [ax0,
           fig.add_subplot(gs[0, 2:], sharey=ax0),
           fig.add_subplot(gs[1, 1:3], sharey=ax0)]
    axs[1].tick_params(axis='y', labelleft=False)
    x = np.arange(len(C.CATEGORY_ORDER))
    w = 0.38; bw = 0.30   # bw < w -> gap within each coastal/inland pair
    zones = [('coastal', '#2E7D32', 'Coastal (≤ threshold)'),
             ('inland', '#EF6C00', 'Inland (> threshold)')]
    for k, thr in enumerate(THRESHOLDS):
        ax = axs[k]
        for zi, (zone, col, lab) in enumerate(zones):
            means = [sums[thr][(zone, cat)]['mean'] / 1e3
                     for cat in C.CATEGORY_ORDER]
            lo = [max(0.0, (sums[thr][(zone, cat)]['mean']
                            - sums[thr][(zone, cat)]['mean_lo']) / 1e3)
                  for cat in C.CATEGORY_ORDER]
            hi = [(sums[thr][(zone, cat)]['mean_hi']
                   - sums[thr][(zone, cat)]['mean']) / 1e3
                  for cat in C.CATEGORY_ORDER]
            ax.bar(x + (zi - 0.5) * w, means, bw, yerr=[lo, hi], capsize=3,
                   color=col, edgecolor='black', lw=0.5, label=lab,
                   error_kw=dict(alpha=0.8, elinewidth=2))
            for j, cat in enumerate(C.CATEGORY_ORDER):
                s = sums[thr][(zone, cat)]
                xc = x[j] + (zi - 0.5) * w
                ax.text(xc, s['mean_hi'] / 1e3 + 0.02 * ymax,
                        f"{s['mean'] / 1e3:.1f}",
                        ha='center', va='bottom', fontsize=VAL_FS)
        ax.set_xticks(x)
        ax.set_xticklabels([C.CATEGORY_LABEL[c] for c in C.CATEGORY_ORDER],
                           fontsize=TICK_FS)
        ax.tick_params(axis='y', labelsize=TICK_FS)
        ax.tick_params(axis='both', length=3)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        if k == 0:
            ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False)
        # (c) sits alone in its row with its own tick labels -> it needs the
        # y-axis title too ((b) shares (a)'s scale, tick labels hidden)
        if k in (0, 2):
            ax.set_ylabel('Mean event TCP (10$^9$ m$^3$)', fontsize=TICK_FS)
        ax.set_title(f'({chr(97 + k)}) {int(thr)} km threshold',
                     fontsize=TICK_FS, fontweight='bold')
        ax.set_ylim(0, 28)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.06)
    P.save(fig, 'figS4-coastal_sensitivity.png')
    print(f"coastal/inland sensitivity over {n_ok} storms; "
          f"ymax={ymax:.1f} 10^9 m^3")
    # remove a stale S9 output left by an earlier run
    old = Path(C.fig_path('figS9-coastal_sensitivity.png'))
    if old.exists():
        old.unlink()


if __name__ == "__main__":
    main()
