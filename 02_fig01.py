"""Figure 1: basin activity, China-landfall frequency and TCP by MJO phase.

Panels:
  (a) Phase-day-normalized ACE, all-WNP TCs and the China-landfall subset.
  (b) Phase-day-normalized landfall count by LMI category.
  (c) Total TCP over China (10^9 m^3).
  (d) Mean event-total TCP per storm (10^9 m^3).

Error bars are 95% year-block bootstrap intervals. Permutation p and BH q are
reported in Table S2. Output: fig/fig1-landfall_ace_tcp.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import config as C
from lib import plot_style as P

TITLE_FS = 22
N_FS = 17    # bar count annotations
N_FS_D = 20  # (d) n labels, larger than the other panels


def _stats(family, sample=None):
    """Query r2_stats_pergrp.csv for one family/sample, indexed by phase."""
    st = pd.read_csv(C.data_path("r2_stats_pergrp.csv"))
    st = st[st['family'] == family]
    if sample is not None:
        st = st[st['sample'] == sample]
    st = st.set_index('phase_group').reindex(C.GROUP_KEY)
    return st


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_ace(ax):
    res = {'All WNP': _stats('fig1a', 'All WNP TCs'),
           'China-landfall': _stats('fig1a', 'China-landfall subset')}
    x = np.arange(len(C.GROUP_KEY)); w = 0.38; bw = 0.30
    for i, ((lab, res_g), col) in enumerate(zip(
            res.items(), ['#888888', P.PHASE_COLORS['5-6']])):
        vv = res_g['est'].values
        lo = np.maximum(0.0, vv - res_g['est_lo'].values)
        hi = res_g['est_hi'].values - vv
        ax.bar(x + (i - 0.5) * w, vv, bw, yerr=[lo, hi], capsize=3, color=col,
               edgecolor='black', lw=0.5, label=f"{lab} (95% CI)",
               error_kw=dict(alpha=0.8, elinewidth=2))
        for j, g in enumerate(C.GROUP_KEY):
            ax.text(x[j] + (i - 0.5) * w, vv[j] + hi[j] + 0.035,
                    f"{int(res_g['n'].values[j])}",
                    ha='center', va='bottom', fontsize=N_FS)
    ax.axhline(1.0, color='gray', ls='--', lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY],
                       fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    # clean symmetric 0-2 range: 1.0 no-modulation centred, matches panel (b)
    ax.set_ylim(0, 2.0); ax.set_ylabel('ACE / no-modulation', fontsize=P.HOUSE_FS)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title('(a) Phase-normalized ACE', fontsize=TITLE_FS, fontweight='bold')
    # legend nudged toward the panel corner
    ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False,
              borderaxespad=0.1)


def panel_landfall_by_lmi(ax):
    x = np.arange(len(C.GROUP_KEY)); w = 0.26; bw = 0.20
    for i, cat in enumerate(C.CATEGORY_ORDER):
        st = _stats('fig1b', C.CATEGORY_FULL[cat])
        vv = st['est'].values
        lo = np.maximum(0.0, vv - st['est_lo'].values)
        hi = st['est_hi'].values - vv
        col = C.COLOR_MAP[C.CATEGORY_LABEL[cat]]
        ax.bar(x + (i - 1) * w, vv, bw, yerr=[lo, hi], capsize=3, color=col,
               edgecolor='black', lw=0.5, label=C.CATEGORY_FULL[cat],
               error_kw=dict(alpha=0.8, elinewidth=1.5))
        for j, g in enumerate(C.GROUP_KEY):
            xc = x[j] + (i - 1) * w
            ax.text(xc, vv[j] + hi[j] + 0.035, f"{int(st['n'].values[j])}",
                    ha='center', va='bottom', fontsize=N_FS)
    ax.axhline(1.0, color='gray', ls='--', lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY],
                       fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    # clean 0-2 range shared with panel (a); headroom to 2.12 so the enlarged
    # count labels above the tallest error bar (Super 3-4, ~1.96) stay inside
    ax.set_ylim(0, 2.12)
    ax.set_ylabel('Landfalls / no-modulation', fontsize=P.HOUSE_FS)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title('(b) Landfall count by LMI', fontsize=TITLE_FS, fontweight='bold')
    # legend nudged toward the panel corner
    ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False,
              labelspacing=0.12, borderaxespad=0.1)


def panel_total_tcp(ax):
    st = _stats('tcp_total', 'All TCs')
    vv = st['est'].values / 1e3                       # 10^6 -> 10^9 m^3
    lo = np.maximum(0.0, vv - st['est_lo'].values / 1e3)
    hi = st['est_hi'].values / 1e3 - vv
    x = np.arange(len(C.GROUP_KEY))
    ax.bar(x, vv, 0.45, yerr=[lo, hi], capsize=4,
           color=[P.PHASE_COLORS[g] for g in C.GROUP_KEY],
           edgecolor='black', lw=0.5, error_kw=dict(alpha=0.85, elinewidth=2))
    ax.axhline(vv.sum() / 4, color='gray', ls='--', lw=1.2,
               label='No-modulation')
    for i, g in enumerate(C.GROUP_KEY):
        ax.text(i, vv[i] + hi[i] + 90, f"{vv[i]:.0f}\nn={int(st['n'].values[i])}",
                ha='center', va='bottom', fontsize=N_FS)
    ax.set_xticks(x)
    ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY],
                       fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylabel('Total TCP (10$^9$ m$^3$)', fontsize=P.HOUSE_FS)
    ax.set_title('(c) Total TCP over China', fontsize=TITLE_FS, fontweight='bold')
    ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False)
    ax.set_ylim(0, (vv + hi).max() * 1.30)
    ax.yaxis.set_major_locator(MultipleLocator(500))


def panel_mean_tcp(ax):
    st = _stats('tcp_mean', 'All TCs')
    overall = st['est'].mean()
    vv = st['est'].values / 1e3                       # 10^6 -> 10^9 m^3
    lo = np.maximum(0.0, vv - st['est_lo'].values / 1e3)
    hi = st['est_hi'].values / 1e3 - vv
    x = np.arange(len(C.GROUP_KEY))
    ax.bar(x, vv, 0.45, yerr=[lo, hi], capsize=4,
           color=[P.PHASE_COLORS[g] for g in C.GROUP_KEY],
           edgecolor='black', lw=0.5, error_kw=dict(alpha=0.85, elinewidth=2))
    ax.axhline(overall / 1e3, color='gray', ls='--', lw=1.2, label='Overall mean')
    for i, g in enumerate(C.GROUP_KEY):
        ax.text(i, vv[i] + hi[i] + 1.1, f"n={int(st['n'].values[i])}",
                ha='center', va='bottom', fontsize=N_FS_D)
    ax.set_xticks(x)
    ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY],
                       fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylabel('Mean event TCP (10$^9$ m$^3$)', fontsize=P.HOUSE_FS)
    ax.set_title('(d) Mean event TCP per storm', fontsize=TITLE_FS,
                 fontweight='bold')
    ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False)
    ax.set_ylim(0, (vv + hi).max() * 1.35)
    ax.yaxis.set_major_locator(MultipleLocator(10))


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)
    assert len(df) == 258

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True,
                             gridspec_kw=dict(hspace=0.11, wspace=0.15))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]
    panel_ace(ax_a)
    panel_landfall_by_lmi(ax_b)
    panel_total_tcp(ax_c)
    panel_mean_tcp(ax_d)
    # shared x-axis: phase labels only on the bottom row
    for ax in (ax_a, ax_b):
        ax.tick_params(labelbottom=False)
    P.save(fig, 'fig1-landfall_ace_tcp.png')


if __name__ == "__main__":
    main()
