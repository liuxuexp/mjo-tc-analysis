"""Figure S9 — coastal vs inland event-total TCP sensitivity to the distance
threshold. Companion to Fig 4(h), which uses the primary 200-km split.

Repeats the coastal/inland MEAN event-total TCP for Weak / Moderate / Super
storms at 100, 200 and 300 km, to verify the coastal-inland contrast is not an
artifact of the chosen threshold. Error bars are the 5th-95th percentile bootstrap
interval; sample sizes match Fig 4(g)/(h).

Style mirrors Fig 4(h): coastal green / inland orange, edgecolor black, fontsize
18, value text 14 above each bar, dpi 600 tight.
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
from lib import bootstrap as bs
from lib import plot_style as P

THRESHOLDS = [100.0, 200.0, 300.0]


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)
    codes = [f"{c:04d}" for c in df['chinese_code']]
    cats = df['lmi_category'].values

    # per-storm coastal/inland totals at each threshold (one field read per storm)
    per = {thr: {'coastal': {c: [] for c in C.CATEGORY_ORDER},
                 'inland':  {c: [] for c in C.CATEGORY_ORDER}} for thr in THRESHOLDS}
    n_ok = 0
    for code, cat in zip(codes, cats):
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

    # summarize (mean + bootstrap CI) and resolve a common y-axis ceiling
    sums = {thr: {} for thr in THRESHOLDS}
    ymax = 0.0
    for thr in THRESHOLDS:
        for zone in ('coastal', 'inland'):
            for cat in C.CATEGORY_ORDER:
                arr = np.array(per[thr][zone][cat], dtype=float)
                s = bs.summarize(arr) if arr.size else dict(mean=0.0, ci_lo=0.0, ci_hi=0.0, n=0)
                sums[thr][(zone, cat)] = s
                ymax = max(ymax, s['ci_hi'])

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    x = np.arange(len(C.CATEGORY_ORDER))
    w = 0.38; bw = 0.30   # bw < w -> gap within each coastal/inland pair
    zones = [('coastal', '#2E7D32', 'Coastal (≤ threshold)'),
             ('inland', '#EF6C00', 'Inland (> threshold)')]
    for k, thr in enumerate(THRESHOLDS):
        ax = axs[k]
        for zi, (zone, col, lab) in enumerate(zones):
            means = [sums[thr][(zone, cat)]['mean'] for cat in C.CATEGORY_ORDER]
            lo = [max(0.0, sums[thr][(zone, cat)]['mean'] - sums[thr][(zone, cat)]['ci_lo'])
                  for cat in C.CATEGORY_ORDER]
            hi = [sums[thr][(zone, cat)]['ci_hi'] - sums[thr][(zone, cat)]['mean']
                  for cat in C.CATEGORY_ORDER]
            ax.bar(x + (zi - 0.5) * w, means, bw, yerr=[lo, hi], capsize=3,
                   color=col, edgecolor='black', lw=0.5, label=lab,
                   error_kw=dict(alpha=0.8, elinewidth=2))
            for j, cat in enumerate(C.CATEGORY_ORDER):
                s = sums[thr][(zone, cat)]
                xc = x[j] + (zi - 0.5) * w
                ax.text(xc, s['ci_hi'] + 0.02 * ymax, f"{s['mean']:.0f}",
                        ha='center', va='bottom', fontsize=P.BARVAL_FS - 2)
        ax.set_xticks(x)
        ax.set_xticklabels([C.CATEGORY_LABEL[c] for c in C.CATEGORY_ORDER], fontsize=P.HOUSE_FS)
        ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
        ax.tick_params(axis='both', length=3)
        fmt = mticker.ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5000))
        ax.yaxis.get_offset_text().set_fontsize(P.HOUSE_FS)
        if k == 0:
            ax.set_ylabel('Mean event TCP (10$^6$ m$^3$)', fontsize=P.HOUSE_FS)
            ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False)
        ax.set_title(f'({chr(97 + k)}) {int(thr)} km threshold',
                     fontsize=P.HOUSE_FS, fontweight='bold')
        ax.set_ylim(0, 27000)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.06)
    P.save(fig, 'figS9-coastal_sensitivity.png')
    print(f"coastal/inland sensitivity over {n_ok} storms; ymax={ymax:.0f} 10^6 m^3")


if __name__ == "__main__":
    main()
