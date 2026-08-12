"""Figure S1 — monthly distribution of China-landfalling TCs by LMI category
(Jan-Dec, 1960-2024). Provides the basis for restricting the primary analysis
to June-October.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from lib import plot_style as P


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    months = np.arange(1, 13)
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(12)
    for cat in C.CATEGORY_ORDER:
        cnt = np.array([((df['month'] == m) & (df['lmi_category'] == cat)).sum() for m in months])
        ax.bar(x, cnt, bottom=bottom, color=C.COLOR_MAP[C.CATEGORY_LABEL[cat]],
               edgecolor='black', lw=0.4, label=C.CATEGORY_FULL[cat])
        bottom += cnt
    # shade Jun-Oct (analysis season); no inline label — the description is the filename
    ax.axvspan(4.5, 9.5, color='gold', alpha=0.12, zorder=0)
    for i, m in enumerate(months):
        tot = int(((df['month'] == m)).sum())
        if tot:
            ax.text(i, bottom[i] + 1, str(tot), ha='center', fontsize=P.BARVAL_FS)
    ax.set_xticks(x); ax.set_xticklabels(
        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.tick_params(axis='both', labelsize=P.HOUSE_FS)
    ax.set_ylabel('Number of China-landfalling TCs', fontsize=P.HOUSE_FS)
    ax.legend(fontsize=P.HOUSE_FS, frameon=False)
    P.save(fig, 'figS1-monthly.png')


if __name__ == "__main__":
    main()
