"""02 — Figure 1: basin activity -> China-landfall frequency -> total TCP
-> mean TCP per storm, all phase-day-normalized with n + uncertainty.

Panels:
  (a) Phase-day-normalized ACE (ratio to no-modulation): all-WNP TCs vs the
      subset that later makes landfall in China.
  (b) Phase-day-normalized number of China-landfalling TCs by LMI category
      (ratio to no-modulation).
  (c) Total TCP over China by phase (10^6 m^3) with n.
  (d) Mean event-total TCP per storm by phase (10^6 m^3) with 90% bootstrap CI.

Dashed line = no-modulation expectation after residence-time normalisation.
Asterisks = two-sided alpha = 0.10.
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
from lib import mjo as mj
from lib import bootstrap as bs
from lib import plot_style as P

ND = mj.active_days_per_group(months=C.SEASON_MONTHS)
TITLE_FS = 22


# ---------------------------------------------------------------------------
# ACE per phase (storm-level), residence-normalized ratio with bootstrap CI
# ---------------------------------------------------------------------------
def _storm_ace_by_phase(tracks):
    """Return DataFrame: one row per storm with ACE per phase group + LMI cat."""
    tracks = tracks.copy()
    tracks['TIME'] = pd.to_datetime(tracks['TIME'])
    tracks['date'] = tracks['TIME'].dt.normalize()
    tracks = tracks[(tracks['TIME'].dt.year.between(C.YEAR_START, C.YEAR_END)) &
                    (tracks['TIME'].dt.month.between(6, 10))]
    tracks = tracks.dropna(subset=['WND'])
    tracks = tracks[tracks['WND'] > 0]
    d2g = mj.date_to_phase_map(months=C.SEASON_MONTHS)
    tracks['group'] = tracks['date'].map(d2g)
    tracks = tracks.dropna(subset=['group'])
    tracks['ace'] = (tracks['WND'] * C.MS_TO_KT) ** 2 / C.ACE_UNIT
    g = tracks.groupby(['chinese_code', 'group', 'wind_category'], observed=True)['ace'].sum().reset_index()
    return g


def ace_ratio(tracks, rng):
    """ratio per group (multiples of no-modulation) with storm-bootstrap CI."""
    g = _storm_ace_by_phase(tracks)
    storms = g['chinese_code'].unique()
    ace_g = {gg: float(g[g['group'] == gg]['ace'].sum()) for gg in C.GROUP_KEY}
    tot = sum(ace_g.values()); totd = sum(ND.values())
    null = tot / totd
    ratio = {gg: (ace_g[gg] / ND[gg]) / null if null > 0 else 0.0 for gg in C.GROUP_KEY}
    # bootstrap over storms
    boot = np.zeros((C.BOOT_NITER, 4))
    storm_idx = {s: i for i, s in enumerate(storms)}
    mat = np.zeros((len(storms), 4))
    for _, r in g.iterrows():
        mat[storm_idx[r['chinese_code']], C.GROUP_KEY.index(r['group'])] += r['ace']
    for b in range(C.BOOT_NITER):
        idx = rng.integers(0, len(storms), len(storms))
        s = mat[idx].sum(axis=0)
        bt = s.sum(); bn = bt / totd if totd else 1.0
        boot[b] = np.where(ND_arr > 0, (s / ND_arr) / bn, 0.0)
    lo = np.percentile(boot, 5, axis=0); hi = np.percentile(boot, 95, axis=0)
    return {gg: dict(ratio=ratio[gg], ci_lo=float(lo[i]), ci_hi=float(hi[i]),
                     n=int((g['group'] == gg).sum() if False else (g[g['group']==gg]['chinese_code'].nunique())))
            for i, gg in enumerate(C.GROUP_KEY)}


ND_arr = np.array([ND[g] for g in C.GROUP_KEY], dtype=float)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_ace(ax, rng):
    alltr = pd.read_csv(C.TYPHOON_OUT / "all_typhoons_tracks.csv")
    lftr = pd.read_csv(C.TYPHOON_OUT / "landfall_typhoons_tracks.csv")
    ra = ace_ratio(alltr, rng); rl = ace_ratio(lftr, rng)
    x = np.arange(len(C.GROUP_KEY)); w = 0.38; bw = 0.30
    ymax = 0.0
    for i, (res, col, lab) in enumerate([(ra, '#888888', 'All WNP'),
                                         (rl, P.PHASE_COLORS['5-6'], 'China-landfall')]):
        vv = [res[g]['ratio'] for g in C.GROUP_KEY]
        lo = [max(0.0, res[g]['ratio'] - res[g]['ci_lo']) for g in C.GROUP_KEY]
        hi = [res[g]['ci_hi'] - res[g]['ratio'] for g in C.GROUP_KEY]
        ax.bar(x + (i - 0.5) * w, vv, bw, yerr=[lo, hi], capsize=3, color=col,
               edgecolor='black', lw=0.5, label=lab, error_kw=dict(alpha=0.8, elinewidth=2))
        # storm count above each bar; star = two-sided alpha = 0.10 (90% bootstrap CI excludes 1.0)
        for j, g in enumerate(C.GROUP_KEY):
            top = vv[j] + hi[j]
            ymax = max(ymax, top)
            ax.text(x[j] + (i - 0.5) * w, top + 0.03, f"{res[g]['n']}",
                    ha='center', va='bottom', fontsize=P.BARVAL_FS)
            if res[g]['ci_lo'] > 1.0 or res[g]['ci_hi'] < 1.0:
                ax.text(x[j] + (i - 0.5) * w, top + 0.10, '*',
                        ha='center', va='bottom', fontsize=P.STAR_FS, fontweight='bold')
                ymax = max(ymax, top + 0.13)
    ax.axhline(1.0, color='gray', ls='--', lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY], fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    # clean symmetric 0-2 range: 1.0 no-modulation centred, matches panel (b)
    ax.set_ylim(0, 2.0); ax.set_ylabel('ACE / no-modulation', fontsize=P.HOUSE_FS)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title('(a) Phase-normalized ACE', fontsize=TITLE_FS, fontweight='bold')
    ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False)
    print("ACE ratio all-WNP:", {g: round(ra[g]['ratio'], 2) for g in C.GROUP_KEY})
    print("ACE ratio landfall:", {g: round(rl[g]['ratio'], 2) for g in C.GROUP_KEY})


def panel_landfall_by_lmi(ax, df):
    res = {}; counts_by_cat = {}
    for cat in C.CATEGORY_ORDER:
        sub = df[df['lmi_category'] == cat]
        counts = {gg: int((sub['group_landfall'] == gg).sum()) for gg in C.GROUP_KEY}
        counts_by_cat[cat] = counts
        res[cat] = mj.residence_ratio_test(counts, ND)
    x = np.arange(len(C.GROUP_KEY)); w = 0.26; bw = 0.20
    ymax = 0.0
    for i, cat in enumerate(C.CATEGORY_ORDER):
        vv = [res[cat][g]['ratio'] for g in C.GROUP_KEY]
        # error bars = residence-time null band (clamped >= 0)
        lo = [max(0.0, res[cat][g]['ratio'] - res[cat][g]['ci_lo']) for g in C.GROUP_KEY]
        hi = [max(0.0, res[cat][g]['ci_hi'] - res[cat][g]['ratio']) for g in C.GROUP_KEY]
        col = C.COLOR_MAP[C.CATEGORY_LABEL[cat]]
        ax.bar(x + (i - 1) * w, vv, bw, yerr=[lo, hi], capsize=3, color=col,
               edgecolor='black', lw=0.5, label=C.CATEGORY_FULL[cat],
               error_kw=dict(alpha=0.8, elinewidth=1.5))
        for j, g in enumerate(C.GROUP_KEY):
            xc = x[j] + (i - 1) * w
            top = vv[j] + hi[j]                       # visible top (bar + upper null extent)
            ymax = max(ymax, top)
            # storm count above the error bar; star stacked just above
            ax.text(xc, top + 0.03, f"{counts_by_cat[cat][g]}", ha='center',
                    va='bottom', fontsize=P.BARVAL_FS)
            if res[cat][g]['stars']:
                ax.text(xc, top + 0.10, '*', ha='center', va='bottom',
                        fontsize=P.STAR_FS, fontweight='bold')
                ymax = max(ymax, top + 0.13)
    ax.axhline(1.0, color='gray', ls='--', lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY], fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    # clean symmetric 0-2 range: 1.0 no-modulation centred, shared with panel (a)
    ax.set_ylim(0, 2.0); ax.set_ylabel('Landfalls / no-modulation', fontsize=P.HOUSE_FS)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title('(b) Landfall count by LMI', fontsize=TITLE_FS, fontweight='bold')
    ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False, labelspacing=0.12)


def panel_total_tcp(ax, df):
    tot = {gg: float(df[df['group_landfall'] == gg]['tcp_total'].sum()) for gg in C.GROUP_KEY}
    n = {gg: int((df['group_landfall'] == gg).sum()) for gg in C.GROUP_KEY}
    x = np.arange(len(C.GROUP_KEY))
    grand = sum(tot.values())
    bars = ax.bar(x, [tot[g] for g in C.GROUP_KEY], 0.45,
                  color=[P.PHASE_COLORS[g] for g in C.GROUP_KEY], edgecolor='black', lw=0.5)
    ax.axhline(grand / 4, color='gray', ls='--', lw=1.2, label='no-modulation')
    for i, g in enumerate(C.GROUP_KEY):
        ax.text(i, tot[g], f"{tot[g]:.0f}\nn={n[g]}", ha='center', va='bottom', fontsize=P.BARVAL_FS)
    ax.set_xticks(x); ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY], fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylabel('Total TCP (10$^6$ m$^3$)', fontsize=P.HOUSE_FS)
    ax.set_title('(c) Total TCP over China', fontsize=TITLE_FS, fontweight='bold')
    ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False)
    ax.set_ylim(0, max(tot.values()) * 1.25)
    ax.yaxis.get_offset_text().set_fontsize(P.HOUSE_FS)


def panel_mean_tcp(ax, df, rng):
    means = {}; los = {}; his = {}; ns = {}
    for g in C.GROUP_KEY:
        v = df[df['group_landfall'] == g]['tcp_total'].values
        s = bs.summarize(v)
        means[g] = s['mean']; los[g] = s['ci_lo']; his[g] = s['ci_hi']; ns[g] = s['n']
    overall = df['tcp_total'].mean()
    x = np.arange(len(C.GROUP_KEY))
    bars = ax.bar(x, [means[g] for g in C.GROUP_KEY], 0.45,
                  yerr=[[means[g]-los[g] for g in C.GROUP_KEY],
                        [his[g]-means[g] for g in C.GROUP_KEY]],
                  capsize=4, color=[P.PHASE_COLORS[g] for g in C.GROUP_KEY],
                  edgecolor='black', lw=0.5, error_kw=dict(alpha=0.85, elinewidth=2))
    ax.axhline(overall, color='gray', ls='--', lw=1.2, label='overall mean')
    for i, g in enumerate(C.GROUP_KEY):
        ax.text(i, means[g] + (his[g]-means[g]) + 300, f"n={ns[g]}", ha='center', va='bottom', fontsize=P.BARVAL_FS)
    ax.set_xticks(x); ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY], fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylabel('Mean event TCP (10$^6$ m$^3$)', fontsize=P.HOUSE_FS)
    ax.set_title('(d) Mean event TCP per storm', fontsize=TITLE_FS, fontweight='bold')
    ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False)
    ax.set_ylim(0, max(means.values()) * 1.35)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(P.HOUSE_FS)


def main():
    rng = np.random.default_rng(20240601)
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True,
                             gridspec_kw=dict(hspace=0.11, wspace=0.15))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]
    panel_ace(ax_a, rng)
    panel_landfall_by_lmi(ax_b, df)
    panel_total_tcp(ax_c, df)
    panel_mean_tcp(ax_d, df, rng)
    # shared x-axis: phase labels only on the bottom row
    for ax in (ax_a, ax_b):
        ax.tick_params(labelbottom=False)
    P.save(fig, 'fig1-landfall_ace_tcp.png')


if __name__ == "__main__":
    main()
