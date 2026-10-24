"""Compare the day-level multinomial null with the year-block inference.

For every comparison in the figure families, recompute the day-level test: each
event is independently assigned to a phase group drawn from the active-day
multinomial (1,000 resamples, significance outside the central 90% band). Each
result is tabulated against the year-block permutation p and BH-adjusted q from
r2_stats_pergrp.csv.

Output: reports/sensitivity_daylevel.csv and reports/sensitivity_daylevel.md
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import config as C
from lib import mjo as mj
from lib import stats_yb as S

RNG_SEED = 20240601          # seed for the day-level resampling
OUT_DIR = Path(__file__).resolve().parent / "reports"


def multinomial_null(counts, ND, n_iter=C.BOOT_NITER, seed=RNG_SEED):
    """Day-level multinomial null.

    Returns {group: (rate, p_emp, sig_a10)} where p_emp is the two-sided
    empirical p from the simulated null rates and sig_a10 is the alpha = 0.10
    outside-the-band criterion.
    """
    rng = np.random.default_rng(seed)
    p = np.array([ND[g] / sum(ND.values()) for g in C.GROUP_KEY])
    nd = np.array([ND[g] for g in C.GROUP_KEY], float)
    ntot = int(sum(counts.values()))
    sim = np.empty((n_iter, 4))
    for b in range(n_iter):
        c = np.bincount(rng.choice(4, size=ntot, p=p), minlength=4).astype(float)
        sim[b] = c / nd * 1000
    out = {}
    for i, g in enumerate(C.GROUP_KEY):
        obs = counts[g] / nd[i] * 1000
        r_up = int((sim[:, i] >= obs).sum()); r_lo = int((sim[:, i] <= obs).sum())
        p_emp = min(1.0, 2 * min((r_up + 1) / (n_iter + 1), (r_lo + 1) / (n_iter + 1)))
        lo, hi = np.percentile(sim[:, i], [5, 95])
        out[g] = (obs, p_emp, bool(obs < lo or obs > hi))
    return out


def binomial_null(n_all, n_lf, n_iter=C.BOOT_NITER, seed=1):
    """Day-level binomial null for proportions."""
    rng = np.random.default_rng(seed)
    phat = n_lf / n_all
    sim = rng.binomial(n_all, phat, size=n_iter) / n_all * 100
    lo, hi = np.percentile(sim, [5, 95])
    r_up = int((sim >= phat * 100).sum()); r_lo = int((sim <= phat * 100).sum())
    p_emp = min(1.0, 2 * min((r_up + 1) / (n_iter + 1), (r_lo + 1) / (n_iter + 1)))
    return phat * 100, p_emp, bool(phat * 100 < lo or phat * 100 > hi)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    st = pd.read_csv(C.data_path("r2_stats_pergrp.csv"))
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)
    ND = mj.active_days_per_group(months=C.SEASON_MONTHS)

    rows = []

    def add(family, sample, stratum, label, counts, metric='rate'):
        new = st[(st['family'] == family)]
        new = new[new['sample'] == sample] if sample else new
        new = new[new['stratum'] == stratum] if stratum else new
        if metric == 'rate':
            old = multinomial_null(counts, ND)
        for i, g in enumerate(C.GROUP_KEY):
            r = new[new['phase_group'] == g].iloc[0]
            if metric == 'rate':
                obs, p_old, sig_old = old[g]
            else:
                obs, p_old, sig_old = counts[g]
            rows.append(dict(
                family=family, sample=sample or '', stratum=stratum or '',
                phase_group=g, n=int(r['n']),
                old_stat=round(float(obs), 2),
                old_p_daylevel=round(float(p_old), 4),
                old_sig_alpha10=bool(sig_old),
                new_stat=round(float(r['est']), 2),
                new_p_yearblock=round(float(r['p_perm']), 4),
                new_q_fdr=round(float(r['q_fdr']), 4)))

    # fig3: 12 sample x region sets
    CAT_FULL = {0: 'All TCs', **{c: C.CATEGORY_FULL[c] for c in C.CATEGORY_ORDER}}
    for cat, sample in [(0, 'All TCs'), (1, 'Weak TC'), (2, 'Moderate TC'),
                        (3, 'Super TC')]:
        sub = df if cat == 0 else df[df['lmi_category'] == cat]
        for region in C.REGION_ORDER:
            counts = {g: int(((sub['group_landfall'] == g) &
                              (sub['landfall_region'] == region)).sum())
                      for g in C.GROUP_KEY}
            add('fig3', sample, region, None, counts)
    # fig1b: per LMI category
    for cat, sample in [(1, 'Weak TC'), (2, 'Moderate TC'), (3, 'Super TC')]:
        sub = df[df['lmi_category'] == cat]
        counts = {g: int((sub['group_landfall'] == g).sum()) for g in C.GROUP_KEY}
        add('fig1b', sample, None, None, counts)
    # fig5a: all TCs
    counts = {g: int((df['group_landfall'] == g).sum()) for g in C.GROUP_KEY}
    add('fig5a', 'All TCs', None, None, counts)
    # fig7c/7d: per genesis region (rebuild the genesis-frame samples)
    from lib.wnp import all_wnp_genesis
    aw = all_wnp_genesis()
    lf_g = df[df['active_genesis']].rename(columns={'group_genesis': 'group'})
    for fam, d, sample in [('fig7c', aw, 'All WNP TCs'),
                           ('fig7d', lf_g, 'China-landfall subset')]:
        for gr in C.GENESIS_ORDER:
            sub = d[d['formation_region'] == gr]
            counts = {g: int((sub['group'] == g).sum()) for g in C.GROUP_KEY}
            add(fam, sample, gr, None, counts)
    # fig8a/8b: genesis rates; fig8c: binomial proportions
    counts_a = {g: int((aw['group'] == g).sum()) for g in C.GROUP_KEY}
    add('fig8a', 'All WNP TCs', None, None, counts_a)
    counts_b = {g: int((lf_g['group'] == g).sum()) for g in C.GROUP_KEY}
    add('fig8b', 'China-landfall subset', None, None, counts_b)
    props = {}
    for g in C.GROUP_KEY:
        n_all = counts_a[g]; n_lf = counts_b[g]
        _, p_old, sig_old = binomial_null(n_all, n_lf)
        props[g] = (100 * n_lf / n_all, p_old, sig_old)
    add('fig8c', 'WNP->China', None, None, props, metric='prop')

    t = pd.DataFrame(rows)
    t.to_csv(OUT_DIR / "sensitivity_daylevel.csv", index=False)

    # ---- summary ----------------------------------------------------------
    lines = ["# Sensitivity: day-level null vs year-block inference",
             "",
             "Day-level = multinomial null, 1,000 resamples, significance",
             "outside the central 90% band (alpha = 0.10). Year-block =",
             "10,000 sequence reassignments with Benjamini-Hochberg FDR",
             "within each family; q < 0.05 significant.",
             ""]
    for fam in t['family'].unique():
        sub = t[t['family'] == fam]
        n_old = int(sub['old_sig_alpha10'].sum())
        n_qp = int((sub['new_p_yearblock'] < 0.05).sum())
        n_q = int((sub['new_q_fdr'] < 0.05).sum())
        m = sub['family'].iloc[0]
        lines.append(f"* **{m}** ({len(sub)} comparisons): "
                     f"old alpha=0.10 significant {n_old}; new p<0.05 {n_qp}; "
                     f"new q<0.05 {n_q}")
    gained = t[(~t['old_sig_alpha10']) & (t['new_q_fdr'] < 0.05)]
    lost = t[t['old_sig_alpha10'] & ~(t['new_q_fdr'] < 0.05)]
    lines += ["",
              f"Comparisons significant under the OLD criterion but not "
              f"surviving FDR: **{len(lost)}**",
              f"Comparisons newly significant at q<0.05 that were NOT "
              f"significant at the old alpha=0.10: **{len(gained)}**", ""]
    if len(gained):
        lines.append("Newly significant (would need explicit justification):")
        for _, r in gained.iterrows():
            lines.append(f"  - {r['family']} {r['sample']} {r['stratum']} "
                         f"{r['phase_group']}: old p={r['old_p_daylevel']}, "
                         f"new q={r['new_q_fdr']}")
    lines += ["", "Full table: sensitivity_daylevel.csv", ""]
    (OUT_DIR / "sensitivity_daylevel.md").write_text("\n".join(lines))
    print(f"saved {OUT_DIR}/sensitivity_daylevel.csv "
          f"({len(t)} comparisons) and .md")
    for fam in t['family'].unique():
        sub = t[t['family'] == fam]
        print(f"  {fam}: old a.10 {int(sub['old_sig_alpha10'].sum())}, "
              f"new p<.05 {int((sub['new_p_yearblock'] < 0.05).sum())}, "
              f"new q<.05 {int((sub['new_q_fdr'] < 0.05).sum())} / {len(sub)}")


if __name__ == "__main__":
    main()
