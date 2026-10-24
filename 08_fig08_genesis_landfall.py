"""Figure 8: WNP TC genesis and subsequent China landfall by MJO phase.

Panels:
  (a) Phase-normalized genesis rate of all WNP TCs, storms per 1,000 active
      phase-days, with phase assigned on the genesis date.
  (b) Same for the subset that later makes landfall in China.
  (c) Proportion of TCs in each genesis-phase group that reach China, with the
      overall proportion as a dashed reference.

Error bars are 95% year-block bootstrap intervals; inference is in Table S2.
Output: fig/fig8-genesis_landfall.png.
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

TITLE_FS = 22     # panel titles, larger than HOUSE_FS
COUNT_FS = 20     # bar-top counts


def _tick_labels():
    """X tick labels without the space: 'Phases1-2' (fig8-only convention)."""
    return [C.group_to_label(g).replace('Phases ', 'Phases') for g in C.GROUP_KEY]


def _stats(family):
    st = pd.read_csv(C.data_path("r2_stats_pergrp.csv"))
    st = st[st['family'] == family]
    return st.set_index('phase_group').reindex(C.GROUP_KEY)


def main():
    st_a = _stats('fig8a')
    st_b = _stats('fig8b')
    st_c = _stats('fig8c')
    n_wnp = int(st_a['n'].sum())
    n_lf = int(st_c['n'].sum())
    assert (n_wnp, n_lf) == (685, 254), (n_wnp, n_lf)
    overall = 100.0 * n_lf / n_wnp

    # Two-row layout: (a)(b) side by side on top, (c) centered
    # below — each panel spans two of four grid columns, so all stay equal-width.
    fig = plt.figure(figsize=(13.0, 12.4))
    gs = fig.add_gridspec(2, 4)
    axs = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:]),
           fig.add_subplot(gs[1, 1:3])]
    x = np.arange(len(C.GROUP_KEY))

    # (a) all-WNP genesis rate
    vv = st_a['est'].values
    lo = np.maximum(0.0, vv - st_a['est_lo'].values)
    hi = st_a['est_hi'].values - vv
    axs[0].bar(x, vv, 0.55, color='#888888', edgecolor='black', lw=0.5,
               yerr=[lo, hi], capsize=4, error_kw=dict(alpha=0.85, elinewidth=2))
    for i, g in enumerate(C.GROUP_KEY):
        axs[0].text(i, vv[i] + hi[i] + 4, f"{int(st_a['n'].values[i])}",
                    ha='center', va='bottom', fontsize=COUNT_FS)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(_tick_labels(), fontsize=P.HOUSE_FS)
    axs[0].tick_params(axis='y', labelsize=P.HOUSE_FS)
    axs[0].set_ylabel('Genesis events / 1000 phase-days', fontsize=P.HOUSE_FS)
    axs[0].set_title(f'(a) All WNP TCs (n={n_wnp})', fontsize=TITLE_FS,
                     fontweight='bold')
    axs[0].set_ylim(0, (vv + hi).max() * 1.25)

    # (b) China-landfall subset genesis rate
    vv = st_b['est'].values
    lo = np.maximum(0.0, vv - st_b['est_lo'].values)
    hi = st_b['est_hi'].values - vv
    axs[1].bar(x, vv, 0.55, color=[P.PHASE_COLORS[g] for g in C.GROUP_KEY],
               edgecolor='black', lw=0.5, yerr=[lo, hi], capsize=4,
               error_kw=dict(alpha=0.85, elinewidth=2))
    for i, g in enumerate(C.GROUP_KEY):
        axs[1].text(i, vv[i] + hi[i] + 1.5, f"{int(st_b['n'].values[i])}",
                    ha='center', va='bottom', fontsize=COUNT_FS)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(_tick_labels(), fontsize=P.HOUSE_FS)
    axs[1].tick_params(axis='y', labelsize=P.HOUSE_FS)
    axs[1].set_ylabel('Genesis events / 1000 phase-days', fontsize=P.HOUSE_FS)
    axs[1].set_title(f'(b) China-landfall subset (n={n_lf})',
                     fontsize=TITLE_FS, fontweight='bold')
    axs[1].set_ylim(0, (vv + hi).max() * 1.25)

    # (c) conditional landfall proportion
    vv = st_c['est'].values
    lo = np.maximum(0.0, vv - st_c['est_lo'].values)
    hi = st_c['est_hi'].values - vv
    axs[2].bar(x, vv, 0.55, color=[P.PHASE_COLORS[g] for g in C.GROUP_KEY],
               edgecolor='black', lw=0.5, yerr=[lo, hi], capsize=4,
               error_kw=dict(alpha=0.85, elinewidth=2))
    axs[2].axhline(overall, color='gray', ls='--', lw=1.4,
                   label=f'Overall {overall:.1f}%')
    for i, g in enumerate(C.GROUP_KEY):
        # n_landfall / n_all above each CI
        lab = f"{int(st_c['n'].values[i])}/{int(st_c['D'].values[i])}"
        axs[2].text(i, vv[i] + hi[i] + 1.0, lab, ha='center', va='bottom',
                    fontsize=COUNT_FS)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(_tick_labels(), fontsize=P.HOUSE_FS)
    axs[2].tick_params(axis='y', labelsize=P.HOUSE_FS)
    axs[2].set_ylabel('Landfalling fraction of WNP TCs (%)', fontsize=P.HOUSE_FS)
    axs[2].set_title('(c) Conditional landfall proportion', fontsize=TITLE_FS,
                     fontweight='bold')
    axs[2].legend(fontsize=P.HOUSE_FS, frameon=False, loc='upper right')
    axs[2].set_ylim(0, (vv + hi).max() * 1.30)

    fig.tight_layout()
    P.save(fig, 'fig8-genesis_landfall.png')
    print("fig8a rates:", {g: round(st_a.loc[g, 'est'], 1) for g in C.GROUP_KEY})
    print("fig8b rates:", {g: round(st_b.loc[g, 'est'], 1) for g in C.GROUP_KEY})
    print("fig8c proportions %:", {g: round(st_c.loc[g, 'est'], 1)
                                   for g in C.GROUP_KEY},
          f"overall {overall:.1f}%")


if __name__ == "__main__":
    main()
