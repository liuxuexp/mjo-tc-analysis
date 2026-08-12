"""12 — All-WNP vs China-landfall comparison (Figure S4; also feeds Fig 1a).

Compares phase-day-normalized genesis/activity for ALL WNP TCs and for the subset
that later makes landfall in China, using the same period, intensity definition,
and MJO-day denominator.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from lib import mjo as mj
from lib import plot_style as P
from lib.wnp import all_wnp_genesis


def main():
    allwnp = all_wnp_genesis()                       # genesis region + phase@genesis, all WNP
    lf = pd.read_csv(C.data_path("event_table.csv"))
    lf = lf[lf['in_jjaso'] & lf['active_genesis']].rename(columns={'group_genesis': 'group'})
    ND = mj.active_days_per_group(months=C.SEASON_MONTHS)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(4); w = 0.38; bw = 0.30   # bw < w -> fig1-(a) gap within each pair

    # (a) genesis rate all-WNP vs landfall, residence-normalized
    for i, (d, col, lab) in enumerate([(allwnp, '#888888', 'All WNP'),
                                       (lf, P.PHASE_COLORS['5-6'], 'China-landfall')]):
        counts = {g: int((d['group'] == g).sum()) for g in C.GROUP_KEY}
        res = mj.residence_permutation_test(counts, ND)
        vv = [res[g]['rate'] for g in C.GROUP_KEY]
        lo = [max(0.0, res[g]['rate'] - res[g]['ci_lo']) for g in C.GROUP_KEY]
        hi = [max(0.0, res[g]['ci_hi'] - res[g]['rate']) for g in C.GROUP_KEY]
        axs[0].bar(x + (i - 0.5) * w, vv, bw, color=col, edgecolor='black', lw=0.5,
                   label=f'{lab} (n={sum(counts.values())})',
                   yerr=[lo, hi], capsize=3, error_kw=dict(alpha=0.8, elinewidth=1))
    axs[0].set_xticks(x); axs[0].set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY], fontsize=P.HOUSE_FS - 1)
    axs[0].tick_params(axis='y', labelsize=P.HOUSE_FS)
    axs[0].set_ylabel('Genesis events / 1000 phase-days', fontsize=P.HOUSE_FS)
    axs[0].set_title('(a) Phase-normalized genesis rate', fontsize=P.HOUSE_FS, fontweight='bold')
    axs[0].legend(fontsize=P.BARVAL_FS, frameon=False, loc='upper left', bbox_to_anchor=(-0.02, 1.0))

    # (b) fraction of each phase's WNP storms that make landfall in China (conditional)
    fracs, los, his, ns = [], [], [], []
    for g in C.GROUP_KEY:
        n_all = int((allwnp['group'] == g).sum())
        n_lf = int((lf['group'] == g).sum())
        ov = n_lf / max(1, n_all) if n_all else 0
        obs, sig, lo, hi = mj_proportion(n_lf, n_all)
        fracs.append(obs * 100); los.append(max(0, obs - lo) * 100)
        his.append((hi - obs) * 100); ns.append(f'{n_lf}/{n_all}')
    axs[1].bar(x, fracs, 0.55, color=[P.PHASE_COLORS[g] for g in C.GROUP_KEY],
               edgecolor='black', lw=0.5, yerr=[los, his], capsize=3,
               error_kw=dict(alpha=0.85, elinewidth=1))
    overall = len(lf) / max(1, len(allwnp)) * 100
    axs[1].axhline(overall, color='gray', ls='--', lw=1.2, label=f'overall {overall:.1f}%')
    for i, n in enumerate(ns):
        axs[1].text(i, fracs[i] + his[i] + 0.5, n, ha='center', fontsize=P.BARVAL_FS)
    axs[1].set_xticks(x); axs[1].set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY], fontsize=P.HOUSE_FS - 1)
    axs[1].tick_params(axis='y', labelsize=P.HOUSE_FS)
    axs[1].set_ylabel('% of WNP storms landing in China', fontsize=P.HOUSE_FS)
    axs[1].set_title('(b) China-landfall fraction by phase', fontsize=P.HOUSE_FS, fontweight='bold')
    axs[1].legend(fontsize=P.BARVAL_FS, frameon=False, loc='upper left', bbox_to_anchor=(-0.02, 0.95))
    plt.tight_layout()
    P.save(fig, 'figS4-allwnp_vs_landfall.png')


def mj_proportion(succ, tot, n_iter=C.BOOT_NITER):
    """Binomial CI on succ/tot (no overall null -> use rate itself, CI via bootstrap)."""
    if tot <= 0:
        return 0.0, False, 0.0, 0.0
    p = succ / tot
    rng = np.random.default_rng(1)
    draws = rng.binomial(tot, p, size=n_iter) / tot
    return p, False, float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


if __name__ == "__main__":
    main()
