"""Figure 5: decomposition of China-landfalling TCP by MJO phase.

Panels:
  (a) Phase-day-normalized landfall frequency.
  (b) Total TCP by phase.
  (c) Mean event-total TCP per storm.
  (d) Median event-total TCP per storm.

Error bars are 95% year-block bootstrap intervals; inference is in Table S2.
TCP is shown in 10^9 m^3. Output: fig/fig5-phase_decomp.png.

(a,b) scale with how many storms occupy a phase; (c,d) do not. (c,d) bars keep
absolute TCP but are labelled with the % deviation from the overall mean/median
(the dashed reference), so the modest cross-phase contrast is legible.

Style: 2x2 layout @ 18x12, sharex with phase labels on bottom row only,
fontsize 18 for axis labels/ticks/titles/legend, bar-value text 14,
elinewidth 2, ratio ticks via MultipleLocator/FormatStrFormatter, dpi 600.
No suptitle — the descriptive title is the filename.
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
N_FS = 20    # on-bar annotations (counts / values / %-deviation labels)


def _stats(family):
    st = pd.read_csv(C.data_path("r2_stats_pergrp.csv"))
    st = st[st['family'] == family]
    return st.set_index('phase_group').reindex(C.GROUP_KEY)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_frequency(ax, st):
    """(a) Phase-normalized landfall frequency (multiples of no-modulation)
    with 95% year-block bootstrap CI; count above the CI."""
    x = np.arange(len(C.GROUP_KEY))
    ratio = st['est'].values
    lo = np.maximum(0.0, ratio - st['est_lo'].values)
    hi = st['est_hi'].values - ratio
    ax.bar(x, ratio, 0.45, color='#455A64', edgecolor='black', linewidth=0.6,
           yerr=[lo, hi], capsize=4, error_kw=dict(elinewidth=2, alpha=0.8))
    ax.axhline(1.0, color='gray', ls='--', lw=1.3, alpha=0.8)
    for i, g in enumerate(C.GROUP_KEY):
        ax.text(x[i], ratio[i] + hi[i] + 0.05, f"{int(st['n'].values[i])}",
                ha='center', va='bottom', fontsize=N_FS)
    ax.set_xticks(x)
    ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY],
                       fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylim(0, 1.9)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel('Landfalls / no-modulation', fontsize=P.HOUSE_FS)
    ax.set_title('(a) Phase-normalized landfall frequency',
                 fontsize=TITLE_FS, fontweight='bold')


def panel_total_tcp(ax, st):
    """(b) Total TCP over China by phase (10^9 m^3) with 95% bootstrap CI."""
    tot = st['est'].values / 1e3
    lo = np.maximum(0.0, tot - st['est_lo'].values / 1e3)
    hi = st['est_hi'].values / 1e3 - tot
    x = np.arange(len(C.GROUP_KEY))
    ax.bar(x, tot, 0.45, yerr=[lo, hi], capsize=4,
           color=[P.PHASE_COLORS[g] for g in C.GROUP_KEY],
           edgecolor='black', lw=0.5, error_kw=dict(alpha=0.85, elinewidth=2))
    ax.axhline(tot.sum() / 4, color='gray', ls='--', lw=1.2,
               label='No-modulation')
    for i, g in enumerate(C.GROUP_KEY):
        ax.text(i, tot[i] + hi[i] + 90,
                f"{tot[i]:.0f}\nn={int(st['n'].values[i])}",
                ha='center', va='bottom', fontsize=N_FS)
    ax.set_xticks(x)
    ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY],
                       fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylabel('Total TCP (10$^9$ m$^3$)', fontsize=P.HOUSE_FS)
    ax.set_title('(b) Total TCP over China', fontsize=TITLE_FS,
                 fontweight='bold')
    ax.legend(fontsize=20, loc='upper left', frameon=False)
    ax.set_ylim(0, (tot + hi).max() * 1.25)
    ax.yaxis.set_major_locator(MultipleLocator(500))


def _per_storm_panel(ax, st, key, overall, title, ylabel, legend_label,
                     ylim=None, freeze_title_y=False):
    """Shared (c)/(d) per-storm TCP bar plot with 95% year-block bootstrap CI.

    `key` is 'est' plus est_lo/est_hi columns from r2_stats (family tcp_mean /
    tcp_median). Styled to match 02_fig01.panel_mean_tcp. `ylim` overrides the
    default max(vv)*1.35 upper bound (e.g. (0, 30) for a clean 0-30 range in
    10^9 m^3).

    Contrast: bars keep absolute TCP, but each is labelled with its % deviation
    from `overall` (the dashed reference) — no significance markers; exact
    p/q values live in Table S2.
    """
    vv = st[key].values / 1e3                       # 10^6 -> 10^9 m^3
    lo = np.maximum(0.0, vv - st['est_lo'].values / 1e3)
    hi = st['est_hi'].values / 1e3 - vv
    x = np.arange(len(C.GROUP_KEY))
    ax.bar(x, vv, 0.45, yerr=[lo, hi], capsize=4,
           color=[P.PHASE_COLORS[g] for g in C.GROUP_KEY],
           edgecolor='black', lw=0.5, error_kw=dict(alpha=0.85, elinewidth=2))
    ax.axhline(overall / 1e3, color='gray', ls='--', lw=1.2, label=legend_label)
    for i, g in enumerate(C.GROUP_KEY):
        dev = (vv[i] - overall / 1e3) / (overall / 1e3) * 100.0
        ax.text(i, vv[i] + hi[i] + 0.8, f"{dev:+.0f}%",
                ha='center', va='bottom', fontsize=N_FS)
    ax.set_xticks(x)
    ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY],
                       fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylabel(ylabel, fontsize=P.HOUSE_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold')
    if freeze_title_y:
        ax._autotitlepos = False
        ax.title.set_position((0.53, 1.0))
    ax.legend(fontsize=20, loc='upper left', frameon=False)
    ax.set_ylim(0, ylim[1] if ylim is not None else max(vv) * 1.35)
    ax.yaxis.set_major_locator(MultipleLocator(5 if (ylim and ylim[1] <= 40)
                                               else 10))


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)
    assert len(df) == 258

    st_a = _stats('fig5a')
    st_tot = _stats('tcp_total')
    st_mean = _stats('tcp_mean')
    st_med = _stats('tcp_median')
    overall_mean = df['tcp_total'].mean()           # 10^6 m^3
    overall_med = df['tcp_total'].median()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True,
                             gridspec_kw=dict(hspace=0.11, wspace=0.22))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]
    panel_frequency(ax_a, st_a)
    panel_total_tcp(ax_b, st_tot)
    _per_storm_panel(ax_c, st_mean, 'est', overall_mean,
                     '(c) Mean event TCP per storm',
                     'Mean event TCP (10$^9$ m$^3$)', 'Overall mean',
                     ylim=(0, 37))
    _per_storm_panel(ax_d, st_med, 'est', overall_med,
                     '(d) Median event TCP per storm',
                     'Median event TCP (10$^9$ m$^3$)', 'Overall median',
                     ylim=(0, 30), freeze_title_y=True)
    # shared x-axis: phase labels only on the bottom row
    for ax in (ax_a, ax_b):
        ax.tick_params(labelbottom=False)
    P.save(fig, 'fig5-phase_decomp.png')
    print("mean TCP/storm by phase (10^9 m^3):",
          {g: round(st_mean.loc[g, 'est'] / 1e3, 1) for g in C.GROUP_KEY})
    print("median TCP/storm by phase (10^9 m^3):",
          {g: round(st_med.loc[g, 'est'] / 1e3, 1) for g in C.GROUP_KEY})


if __name__ == "__main__":
    main()
