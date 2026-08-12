"""05 — Figure 5: MJO-phase decomposition of China-landfalling TCP.

Addresses whether MJO modulation of total precipitation comes from more storms
or wetter storms by decomposing TCP into phase-normalized frequency,
total accumulation, and per-storm means/medians.

(a) Phase-day-normalized landfall frequency (ratio to no-modulation) -> storm NUMBER.
(b) Total TCP by phase -> cumulative impact.
(c) Mean event-total TCP per storm -> per-storm wetness (mean).
(d) Median event-total TCP + 90% CI -> per-storm wetness (typical storm).

(a,b) scale with how many storms occupy a phase; (c,d) do not. (c,d) bars keep
absolute TCP but are labelled with the % deviation from the overall mean/median
(the dashed reference) + '*' where the phase CI excludes it, so the modest
cross-phase contrast is legible. Sample sizes and alpha=0.10 results are printed.

Style: 2x2 layout @ 18x12, sharex with phase labels on bottom row only,
fontsize 18 for axis labels/ticks/titles/legend, bar-value text 14,
elinewidth 2, ratio ticks via MultipleLocator/FormatStrFormatter, per-storm TCP
in scientific notation, dpi 600. No suptitle — the descriptive title is the
filename.
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


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_frequency(ax, res):
    """(a) Phase-normalized landfall frequency (multiples of no-modulation).

    Bare storm count + significance star, drawn manually (not phase_ratio_bars)
    so the annotation matches the fig1-(a) idiom: compact, hugging the error-bar
    top — count just above the CI, star just above the count. No 'n=' prefix,
    count font at HOUSE_FS.
    """
    x = np.arange(len(C.GROUP_KEY))
    ratio = [res[g]['ratio'] for g in C.GROUP_KEY]
    lo = [max(0.0, res[g]['ratio'] - res[g]['ci_lo']) for g in C.GROUP_KEY]
    hi = [max(0.0, res[g]['ci_hi'] - res[g]['ratio']) for g in C.GROUP_KEY]
    ax.bar(x, ratio, 0.45, color='#455A64', edgecolor='black', linewidth=0.6,
           yerr=[lo, hi], capsize=4, error_kw=dict(elinewidth=2, alpha=0.8))
    ax.axhline(1.0, color='gray', ls='--', lw=1.3, alpha=0.8)
    top_max = max(ratio[i] + hi[i] for i in range(len(C.GROUP_KEY)))
    for i, g in enumerate(C.GROUP_KEY):
        top = ratio[i] + hi[i]
        ax.text(x[i], top + 0.04, f"{res[g]['n']}", ha='center', va='bottom',
                fontsize=P.HOUSE_FS)
        if res[g]['stars']:
            ax.text(x[i], top + 0.22, res[g]['stars'], ha='center', va='bottom',
                    fontsize=P.STAR_FS, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY], fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylim(0, 2.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel('Landfalls / no-modulation', fontsize=P.HOUSE_FS)
    ax.set_title('(a) Phase-normalized landfall frequency', fontsize=P.HOUSE_FS, fontweight='bold')


def panel_total_tcp(ax, tot, counts):
    """(b) Total TCP over China by phase (10^6 m^3)."""
    x = np.arange(len(C.GROUP_KEY))
    grand = sum(tot.values())
    ax.bar(x, [tot[g] for g in C.GROUP_KEY], 0.45,
           color=[P.PHASE_COLORS[g] for g in C.GROUP_KEY], edgecolor='black', lw=0.5)
    ax.axhline(grand / 4, color='gray', ls='--', lw=1.2, label='no-modulation')
    for i, g in enumerate(C.GROUP_KEY):
        ax.text(i, tot[g], f"{tot[g]:.0f}\nn={counts[g]}", ha='center', va='bottom',
                fontsize=P.HOUSE_FS)
    ax.set_xticks(x)
    ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY], fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylabel('Total TCP (10$^6$ m$^3$)', fontsize=P.HOUSE_FS)
    ax.set_title('(b) Total TCP over China', fontsize=P.HOUSE_FS, fontweight='bold')
    ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False)
    ax.set_ylim(0, max(tot.values()) * 1.25)
    ax.yaxis.get_offset_text().set_fontsize(P.HOUSE_FS)


def _per_storm_panel(ax, summ, key, overall, title, ylabel, legend_label,
                     ylim=None):
    """Shared (c)/(d) per-storm TCP bar plot with 90% bootstrap CI.

    `key` is 'mean' (bs.summarize) or 'median' (bs.summarize_median); both dicts
    carry ci_lo / ci_hi. Styled to match 02_fig01.panel_mean_tcp. `ylim` overrides
    the default max(vv)*1.35 upper bound (e.g. (0, 3e4) for a clean 0-3 range).

    Contrast: bars keep absolute TCP, but each is labelled with its % deviation
    from `overall` (the dashed reference) and a '*' where `overall` falls outside
    the phase bootstrap CI — so the modest cross-phase differences are legible.
    """
    vv = [summ[g][key] for g in C.GROUP_KEY]
    lo = [max(0.0, summ[g][key] - summ[g]['ci_lo']) for g in C.GROUP_KEY]
    hi = [summ[g]['ci_hi'] - summ[g][key] for g in C.GROUP_KEY]
    x = np.arange(len(C.GROUP_KEY))
    ax.bar(x, vv, 0.45, yerr=[lo, hi], capsize=4,
           color=[P.PHASE_COLORS[g] for g in C.GROUP_KEY], edgecolor='black', lw=0.5,
           error_kw=dict(alpha=0.85, elinewidth=2))
    ax.axhline(overall, color='gray', ls='--', lw=1.2, label=legend_label)
    for i, g in enumerate(C.GROUP_KEY):
        dev = (vv[i] - overall) / overall * 100.0
        sig = overall < summ[g]['ci_lo'] or overall > summ[g]['ci_hi']
        ax.text(i, vv[i] + hi[i] + 300, f"{dev:+.0f}%{'*' if sig else ''}",
                ha='center', va='bottom', fontsize=P.HOUSE_FS)
    ax.set_xticks(x)
    ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY], fontsize=P.HOUSE_FS)
    ax.tick_params(axis='y', labelsize=P.HOUSE_FS)
    ax.set_ylabel(ylabel, fontsize=P.HOUSE_FS)
    ax.set_title(title, fontsize=P.HOUSE_FS, fontweight='bold')
    ax.legend(fontsize=P.HOUSE_FS, loc='upper left', frameon=False)
    ax.set_ylim(0, ylim[1] if ylim is not None else max(vv) * 1.35)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(P.HOUSE_FS)


def panel_mean_tcp(ax, mean_s, overall_mean):
    """(c) Mean event-total TCP per storm (10^6 m^3) with 90% bootstrap CI.

    Bars keep absolute TCP; each is labelled with its % deviation from the overall
    mean (the dashed reference) + '*' where the phase CI excludes it.
    """
    _per_storm_panel(ax, mean_s, 'mean', overall_mean,
                     '(c) Mean event TCP per storm',
                     'Mean event TCP (10$^6$ m$^3$)', 'overall mean')


def panel_median_tcp(ax, med_s, overall_med):
    """(d) Median event-total TCP per storm (10^6 m^3) with 90% bootstrap CI.

    Same deviation-% labelling as (c); ylim fixed to 0-3 (x10^4).
    """
    _per_storm_panel(ax, med_s, 'median', overall_med,
                     '(d) Median event TCP per storm',
                     'Median event TCP (10$^6$ m$^3$)', 'overall median',
                     ylim=(0, 3e4))   # clean 0-3 (x10^4) range


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)
    ND = mj.active_days_per_group(months=C.SEASON_MONTHS)

    # (a) residence-ratio frequency
    counts = {g: int((df['group_landfall'] == g).sum()) for g in C.GROUP_KEY}
    res = mj.residence_ratio_test(counts, ND)

    # (b) total TCP
    tot = {g: float(df[df['group_landfall'] == g]['tcp_total'].sum()) for g in C.GROUP_KEY}

    # (c) mean + (d) median per storm
    mean_s, med_s = {}, {}
    for g in C.GROUP_KEY:
        v = df[df['group_landfall'] == g]['tcp_total'].values
        mean_s[g] = bs.summarize(v)
        med_s[g] = bs.summarize_median(v)
    overall_mean = df['tcp_total'].mean()
    overall_med = bs.summarize_median(df['tcp_total'].values)['median']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True,
                             gridspec_kw=dict(hspace=0.09, wspace=0.15))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]
    panel_frequency(ax_a, res)
    panel_total_tcp(ax_b, tot, counts)
    panel_mean_tcp(ax_c, mean_s, overall_mean)
    panel_median_tcp(ax_d, med_s, overall_med)
    # shared x-axis: phase labels only on the bottom row
    for ax in (ax_a, ax_b):
        ax.tick_params(labelbottom=False)
    P.save(fig, 'fig5-phase_decomp.png')
    print("mean TCP/storm by phase:", {g: round(mean_s[g]['mean']) for g in C.GROUP_KEY})
    print("median TCP/storm by phase:", {g: round(med_s[g]['median']) for g in C.GROUP_KEY})


if __name__ == "__main__":
    main()
