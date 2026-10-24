"""Figure 7: genesis region, landfall region and MJO phase.

Genesis uses the four regions in config.GENESIS_ORDER.
  (a) Genesis-region composition of Weak TCs, conditional on China landfall.
  (b) Same for Moderate and Super TCs.
  (c) MJO-phase distribution of all WNP TCs by genesis region, with 95%
      year-block bootstrap intervals.
  (d) Same for the China-landfalling subset.

Permutation p and BH q for (c) and (d) are in Table S2.
Output: fig/fig7-genesis_mjo.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.transforms import ScaledTranslation

import config as C
from lib import mjo as mj
from lib import plot_style as P

TITLE_FS = 22
YLABEL_FS = 20   # y-axis labels, slightly larger than HOUSE_FS
COUNT_FS = 20    # bar-top counts, larger than BARVAL_FS

# Genesis-region display names shown on the figure (config keys -> figure labels).
_ORIG_GENESIS = {'South China Sea': 'SCS',
                 'Western Tropical WNP': 'South WNP',
                 'Eastern Tropical WNP': 'Open WNP',
                 'North WNP': 'North WNP'}
_LEGEND_GENESIS = {'South China Sea': 'South China Sea',
                   'Western Tropical WNP': 'South WNP',
                   'Eastern Tropical WNP': 'Open WNP',
                   'North WNP': 'North WNP'}


def _genesis_all_wnp():
    """Return genesis region and genesis-date MJO phase for every WNP storm."""
    tr = pd.read_csv(C.TYPHOON_OUT / "all_typhoons_tracks.csv")
    tr['TIME'] = pd.to_datetime(tr['TIME'])
    tr = tr.sort_values(['chinese_code', 'TIME'])
    rows = []
    d2g = mj.date_to_phase_map(months=C.SEASON_MONTHS)
    for code, g in tr.groupby('chinese_code'):
        g = g.sort_values('TIME')
        r = g.iloc[0]
        gdate = pd.Timestamp(r['TIME']).normalize()
        rows.append(dict(chinese_code=int(code),
                         genesis_lat=float(r['LAT']), genesis_lon=float(r['LONG']),
                         genesis_date=gdate,
                         wind_category=int(g['wind_category'].iloc[0]) if 'wind_category' in g else 0,
                         group=d2g.get(gdate)))
    d = pd.DataFrame(rows)
    d['formation_region'] = d.apply(
        lambda r: C.formation_region_from_genesis(r['genesis_lat'], r['genesis_lon']), axis=1)
    d = d.dropna(subset=['group'])     # active MJO at genesis, Jun-Oct
    return d


def _ab_max(sub):
    """Max percentage reaching a landfall region across genesis regions (for shared ylim)."""
    best = 0.0
    for lr in ('South China', 'East/North China'):
        for gr in C.GENESIS_ORDER:
            gsub = sub[sub['formation_region'] == gr]
            tot = len(gsub)
            if tot:
                best = max(best, 100 * int((gsub['landfall_region2'] == lr).sum()) / tot)
    return best


def panel_ab(ax, sub, title, top, ylabel=True):
    """Panel (a)/(b): genesis region by landfall destination (percent conditional).

    `top` is the shared (row) y-limit ceiling; counts are offset relative to it.
    """
    regions2 = ['South China', 'East/North China']
    x = np.arange(len(C.GENESIS_ORDER))
    off, bw = 0.38, 0.30          # bw < off -> fig1-style gap between the two bars
    cols2 = ['#2E7D32', '#EF6C00']
    labs2 = ['South China', 'East/North China']
    for i, lr in enumerate(regions2):
        pcts, labs = [], []
        for gr in C.GENESIS_ORDER:
            gsub = sub[sub['formation_region'] == gr]
            tot = len(gsub)
            succ = int((gsub['landfall_region2'] == lr).sum())
            pcts.append(100 * succ / tot if tot else 0)
            labs.append(f"{succ}/{tot}" if tot else "")      # count / denominator
        ax.bar(x + (i - 0.5) * off, pcts, bw, color=cols2[i],
               edgecolor='black', lw=0.5, label=labs2[i])
        for j, lab in enumerate(labs):
            if lab:
                ax.text(x[j] + (i - 0.5) * off, pcts[j] + 0.015 * top, lab,
                        ha='center', va='bottom', fontsize=COUNT_FS)
    ax.set_xticks(x)
    # Genesis x-labels show the figure display names.
    ax.set_xticklabels([_ORIG_GENESIS[r].replace('China Sea', 'China\nSea')
                        for r in C.GENESIS_ORDER], fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_ylim(0, top * 1.10)
    if ylabel:
        ax.set_ylabel('Landfalls reaching region (%)', fontsize=YLABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', loc='left')
    ax.legend(fontsize=P.HOUSE_FS, loc='upper right', frameon=False)


def _cd_rate_max(d):
    """Max residence-normalized rate (storms / 1000 phase-days) in `d` for the
    shared ylim — computed without the permutation (rate = count / ND * 1000)."""
    ND = mj.active_days_per_group(months=C.SEASON_MONTHS)
    best = 0.0
    for gr in C.GENESIS_ORDER:
        sub = d[d['formation_region'] == gr]
        for gg in C.GROUP_KEY:
            if ND[gg] > 0:
                best = max(best, int((sub['group'] == gg).sum()) / ND[gg] * 1000)
    return best


def panel_cd(ax, fam, sample, d, title, ymax, ylabel=True):
    """Panel (c)/(d): MJO phase by genesis region, residence-normalized rate
    with 95% year-block bootstrap CI (r2_stats_pergrp family fig7c/fig7d).

    `ymax` is the shared (row) y-limit ceiling. Counts are offset in POINTS
    from the bar top (ScaledTranslation) so they hug the CI cap at a fixed,
    scale-independent gap.
    """
    st = pd.read_csv(C.data_path("r2_stats_pergrp.csv"))
    st = st[(st['family'] == fam) & (st['sample'] == sample)]
    st = st.set_index(['stratum', 'phase_group'])
    ND = mj.active_days_per_group(months=C.SEASON_MONTHS)
    x = np.arange(len(C.GROUP_KEY))
    off, bw = 0.20, 0.15          # bw < off -> fig1-style gaps between the four bars
    cols = ['#8E44AD', '#2980B9', '#E67E22', '#27AE60']
    # count 3 pt above the CI cap -> tight, scale-independent cluster.
    count_off = ScaledTranslation(0, 3 / 72, ax.figure.dpi_scale_trans)
    for i, gr in enumerate(C.GENESIS_ORDER):
        vv = np.array([st.loc[(gr, g), 'est'] for g in C.GROUP_KEY])
        lo = np.maximum(0.0, vv - np.array([st.loc[(gr, g), 'est_lo']
                                            for g in C.GROUP_KEY]))
        hi = np.array([st.loc[(gr, g), 'est_hi'] for g in C.GROUP_KEY]) - vv
        ax.bar(x + (i - (len(C.GENESIS_ORDER) - 1) / 2) * off, vv, bw,
               yerr=[lo, hi], capsize=2, color=cols[i], edgecolor='black',
               lw=0.4, label=_LEGEND_GENESIS[gr],
               error_kw=dict(alpha=0.8, elinewidth=1.2))
        for j, g in enumerate(C.GROUP_KEY):
            xc = x[j] + (i - (len(C.GENESIS_ORDER) - 1) / 2) * off
            ax.text(xc, vv[j] + hi[j], f"{int(st.loc[(gr, g), 'n'])}",
                    ha='center', va='bottom',
                    fontsize=COUNT_FS, transform=ax.transData + count_off)
    ax.set_ylim(0, ymax * 1.30)
    # x-tick carries the phase-day denominator on a second line (ND[g] is per-phase,
    # shared by all four genesis-region bars in that column): each bar prints its
    # count above; rate = count / ND * 1000.
    ax.set_xticks(x)
    ax.set_xticklabels([f"{C.group_to_label(g)}\n({ND[g]} d)" for g in C.GROUP_KEY],
                       fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    if ylabel:
        ax.set_ylabel('Storms / 1000 phase-days', fontsize=YLABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', loc='left')
    ax.legend(fontsize=P.HOUSE_FS, ncol=1, loc='upper left', frameon=False)


def _cd_ci_max(fam, sample):
    """Max 95%-CI upper bound across genesis regions and phases."""
    st = pd.read_csv(C.data_path("r2_stats_pergrp.csv"))
    st = st[(st['family'] == fam) & (st['sample'] == sample)]
    return float(st['est_hi'].max())


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)
    allwnp = _genesis_all_wnp()
    weak = df[df['lmi_category'] == 1]
    modmaj = df[df['lmi_category'].isin([2, 3])]
    landfall = df[df['active_genesis']].rename(columns={'group_genesis': 'group'})

    # shared y per row -> one combined ceiling so (a)=(b) and (c)=(d) compare fairly
    top_ab = max(_ab_max(weak), _ab_max(modmaj), 1.0)
    ymax_cd = max(_cd_ci_max('fig7c', 'All WNP TCs'),
                  _cd_ci_max('fig7d', 'China-landfall subset'), 1.0)

    fig, axs = plt.subplots(2, 2, figsize=(18, 12), sharey='row',
                            gridspec_kw=dict(hspace=0.18, wspace=0.02))
    panel_ab(axs[0, 0], weak, '(a) Weak TCs', top=top_ab)
    panel_ab(axs[0, 1], modmaj, '(b) Moderate + Super TCs', top=top_ab, ylabel=False)
    panel_cd(axs[1, 0], 'fig7c', 'All WNP TCs', allwnp, '(c) All WNP TCs',
             ymax=ymax_cd)
    panel_cd(axs[1, 1], 'fig7d', 'China-landfall subset', landfall,
             '(d) China-landfalling TCs', ymax=ymax_cd, ylabel=False)
    # sharey='row': drop the redundant right-column y tick labels.
    for ax in (axs[0, 1], axs[1, 1]):
        ax.tick_params(labelleft=False)
    fig.align_ylabels(axs[:, 0])
    P.save(fig, 'fig7-genesis_mjo.png')


if __name__ == "__main__":
    main()
