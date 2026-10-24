"""01 — Table 2, Table 3, Table S1 (Fig-3 statistics), Table S2 (summary).

Reads data03/event_table.csv + the year-block inference output
(r2_inference.py: r2_stats_pergrp.csv / r2_stats_contrast.csv) and writes:
  tables03/Table2_counts_TCP_by_intensity.csv|md
      Storm counts + TCP stats by intensity category, separately for LMI and for
      intensity-at-first-Chinese-landfall. Units: totals 10^12 m^3, per-storm
      10^9 m^3. CIs: 95% year-block bootstrap (10,000 replicates).
  tables03/Table3_LMI_vs_landfall_crosstab.csv|md
      Cross-classification of LMI x intensity-at-landfall with counts and row %.
  tables03/TableS1_fig3_rates_stats.csv|md
      The 48 pre-specified Figure-3 comparisons (4 TC samples x 3 regions x
      4 phase groups): n, phase-days D, normalized rate, 95% null interval
      (2.5-97.5 percentiles of 10,000 year-block permutations), raw two-sided
      permutation p, BH-adjusted q.
  tables03/TableS2_summary_tests.csv|md
      Block A: per-group year-block tests for the other figure families
      (fig1a/1b/5a/7c/7d/8a/8b/8c, tcp_total/mean/median).
      Block B: the 19 pre-specified phases 5-6 vs 1-2 contrasts with rate
      ratio + 95% CI, absolute rate difference + 95% CI, exact p and q.

All samples are restricted to June-October. Phase-stratified families also
require an active MJO on the assigning event date.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import config as C
from lib import stats_yb as S

try:
    from tabulate import tabulate
except Exception:
    tabulate = None


def load_events(jjaso=True, active=False):
    df = pd.read_csv(C.data_path("event_table.csv"))
    if jjaso:
        df = df[df['in_jjaso']]
    if active:
        df = df[df['active_landfall']]
    return df.reset_index(drop=True)


def tcp_rows_for(df, cat_col, cats=None):
    """Build intensity rows for one classification (cat_col in {lmi_category,
    landfall_wind_category}). Defaults to the three TC-strength categories; pass
    cats=[0,1,2,3] for the landfall classification so the TD-at-landfall (cat 0)
    bucket is included and the share sums to 100% over all storms."""
    rows = []
    for cat in (cats if cats is not None else C.CATEGORY_ORDER):
        sub = df[df[cat_col] == cat]
        label = C.CATEGORY_FULL[cat] if cat in C.CATEGORY_FULL else 'TD'
        d = pd.to_datetime(sub['landfall_time'])
        s = S.boot_scalar_stats(sub['tcp_total'].values, d)
        cs = S.boot_scalar_stats(sub['coastal_total'].values, d)
        is_ = S.boot_scalar_stats(sub['inland_total'].values, d)
        rows.append({
            'Intensity category': f"{label} ({'LMI' if cat_col=='lmi_category' else 'landfall'})",
            'No. storms': s['n'],
            'Share of landfalls (%)': round(100 * s['n'] / max(1, len(df)), 1),
            'Total TCP (10^12 m^3)': round(s['mean'] * s['n'] / 1e6, 2),
            'Mean event TCP (10^9 m^3)': f"{s['mean']/1e3:.1f} "
                                         f"({s['mean_lo']/1e3:.1f}-{s['mean_hi']/1e3:.1f})",
            'Median event TCP (10^9 m^3)': f"{s['median']/1e3:.1f} "
                                           f"({s['median_lo']/1e3:.1f}-{s['median_hi']/1e3:.1f})",
            'Coastal mean event TCP (10^9 m^3)': round(cs['mean'] / 1e3, 1),
            'Inland mean event TCP (10^9 m^3)': round(is_['mean'] / 1e3, 1),
        })
    return rows


def table2():
    df = load_events(jjaso=True, active=False)
    rows = tcp_rows_for(df, 'lmi_category') + \
        tcp_rows_for(df, 'landfall_wind_category', cats=[0, 1, 2, 3])
    t = pd.DataFrame(rows)
    t.to_csv(C.table_path("Table2_counts_TCP_by_intensity.csv"), index=False)
    _write_md("Table2_counts_TCP_by_intensity",
              "Table 2. Storm counts and tropical-cyclone precipitation statistics "
              "by intensity category (Jun-Oct, 1960-2024). TCP integrated over "
              "China land cells; coastal = within 200 km of coastline. "
              "Uncertainty ranges are 95% confidence intervals from 10,000 "
              "year-block bootstrap replicates.",
              t)
    return t


def table3():
    df = load_events(jjaso=True, active=False)
    lab = {0: 'TD', 1: 'Weak', 2: 'Moderate', 3: 'Super'}
    ct = pd.crosstab(df['lmi_category'], df['landfall_wind_category'])
    ct = ct.reindex(index=[1, 2, 3], columns=[0, 1, 2, 3], fill_value=0)
    rows = []
    for lmi in [1, 2, 3]:
        rowtot = int(ct.loc[lmi].sum())   # full LMI count (TD-at-landfall included)
        cells = {}
        for lf in [0, 1, 2, 3]:
            n = int(ct.loc[lmi, lf])
            pct = round(100 * n / rowtot, 1) if rowtot else 0.0
            cells[lab[lf]] = f"{n} ({pct}%)"
        rows.append({'LMI category': f"{lab[lmi]} LMI (n={rowtot})", **cells})
    t = pd.DataFrame(rows)
    t.to_csv(C.table_path("Table3_LMI_vs_landfall_crosstab.csv"), index=False)
    _write_md("Table3_LMI_vs_landfall_crosstab",
              "Table 3. Cross-classification of lifetime maximum intensity (LMI) and "
              "intensity at first Chinese landfall. n (row %).",
              t)
    return t


# ---------------------------------------------------------------------------
# Table S1 — the 48 Figure-3 comparisons
# ---------------------------------------------------------------------------
CAT_FULL = {0: 'All TCs', **{c: C.CATEGORY_FULL[c] for c in C.CATEGORY_ORDER}}


def _fmt_pq(x):
    return S.fmt_p(x, nd=3)


def table_s1():
    st = pd.read_csv(C.data_path("r2_stats_pergrp.csv"))
    st = st[st['family'] == 'fig3']
    assert len(st) == 48, f"fig3 family has {len(st)} rows != 48"
    sample_order = [CAT_FULL[c] for c in [0] + C.CATEGORY_ORDER]
    rows = []
    for sample in sample_order:
        for region in C.REGION_ORDER:
            for g in C.GROUP_KEY:
                r = st[(st['sample'] == sample) & (st['stratum'] == region) &
                       (st['phase_group'] == g)].iloc[0]
                rows.append({
                    'TC sample': sample,
                    'Landfall region': region,
                    'MJO phase': f"Phases {g.replace('-', chr(8211))}",
                    'Event count, n': int(r['n']),
                    'MJO-active phase-days, D': int(r['D']),
                    'Normalized rate': f"{r['est']:.2f}",
                    '95% null interval': f"{r['null_lo']:.2f}-{r['null_hi']:.2f}",
                    'Raw p value': _fmt_pq(r['p_perm']),
                    'FDR-adjusted q value': _fmt_pq(r['q_fdr']),
                })
    t = pd.DataFrame(rows)
    # each sample's 3 regions x 4 phases sum to its size: 258+84+112+62 = 516
    assert len(t) == 48 and t['Event count, n'].sum() == 516, \
        (len(t), t['Event count, n'].sum())
    t.to_csv(C.table_path("TableS1_fig3_rates_stats.csv"), index=False)
    note = ("Table S1. Observed phase-normalized regional landfall rates and "
            "statistical results for the comparisons presented in Figure 3.\n\n"
            "Note. The normalized rate is the observed number of landfalls per "
            "MJO-active phase-day divided by the corresponding all-phase rate; "
            "therefore, a value of 1 represents the no-modulation expectation. "
            "The 95% null interval is defined by the 2.5th and 97.5th "
            "percentiles of 10,000 year-block permutations. Raw p values are "
            "two-sided empirical probabilities. The q values were obtained "
            "using the Benjamini-Hochberg false discovery rate correction "
            "across all 48 comparisons shown in Figure 3. Values of q < 0.05 "
            "are interpreted as statistically significant.")
    _write_md("TableS1_fig3_rates_stats", note, t)
    return t


# ---------------------------------------------------------------------------
# Table S2 — per-group tests (block A) + pre-specified contrasts (block B)
# ---------------------------------------------------------------------------
def _interp(p, q):
    if np.isfinite(q) and q < 0.05:
        return 'significant (q < 0.05)'
    if np.isfinite(p) and p < 0.05:
        return 'suggestive (p < 0.05, q >= 0.05)'
    return 'not significant'


def _fmt_est(r):
    """Observed estimate with unit, 2 significant-ish decimals."""
    if r['metric'] in ('genesis_rate',):
        return f"{r['est']:.1f}"
    if r['metric'] == 'conditional_landfall_proportion':
        return f"{r['est']:.1f}%"
    if r['metric'] in ('total_tcp', 'mean_event_tcp', 'median_event_tcp'):
        return f"{r['est']/1e3:.1f}"          # 10^9 m^3
    return f"{r['est']:.2f}"                  # ratios


def _fmt_ci(r):
    if r['metric'] in ('total_tcp', 'mean_event_tcp', 'median_event_tcp'):
        return f"{r['est_lo']/1e3:.1f}-{r['est_hi']/1e3:.1f}"
    if r['metric'] == 'genesis_rate':
        return f"{r['est_lo']:.1f}-{r['est_hi']:.1f}"
    if r['metric'] == 'conditional_landfall_proportion':
        return f"{r['est_lo']:.1f}-{r['est_hi']:.1f}%"
    return f"{r['est_lo']:.2f}-{r['est_hi']:.2f}"


def _fmt_effect(r):
    if r['metric'] in ('total_tcp', 'mean_event_tcp', 'median_event_tcp',
                       'conditional_landfall_proportion'):
        return f"{r['ratio']:.2f}"
    return f"{r['est']:.2f}"


def table_s2():
    st = pd.read_csv(C.data_path("r2_stats_pergrp.csv"))
    ct = pd.read_csv(C.data_path("r2_stats_contrast.csv"))
    fam_order = ['fig1a', 'fig1b', 'fig5a', 'tcp_total', 'tcp_mean', 'tcp_median',
                 'fig7c', 'fig7d', 'fig8a', 'fig8b', 'fig8c']
    figname = {'fig1a': 'Fig. 1a', 'fig1b': 'Fig. 1b', 'fig5a': 'Fig. 5a',
               'tcp_total': 'Figs. 1c/5b', 'tcp_mean': 'Figs. 1d/5c',
               'tcp_median': 'Fig. 5d', 'fig7c': 'Fig. 7c', 'fig7d': 'Fig. 7d',
               'fig8a': 'Fig. 8a', 'fig8b': 'Fig. 8b', 'fig8c': 'Fig. 8c'}
    metric_lab = {'ace_ratio': 'ACE / no-modulation',
                  'landfall_rate_ratio': 'landfall rate / no-modulation',
                  'genesis_rate': 'genesis rate (per 1000 phase-days)',
                  'conditional_landfall_proportion':
                      'conditional landfall proportion (%)',
                  'total_tcp': 'total TCP (10^9 m^3)',
                  'mean_event_tcp': 'mean event TCP (10^9 m^3)',
                  'median_event_tcp': 'median event TCP (10^9 m^3)'}
    rows_a = []
    for fam in fam_order:
        sub = st[st['family'] == fam].sort_values(
            ['sample', 'stratum', 'phase_group'])
        for _, r in sub.iterrows():
            rows_a.append({
                'Figure or analysis': figname[fam],
                'Metric': metric_lab[r['metric']],
                'MJO phase assigned at': r['phase_at'],
                'Phase group or contrast': f"Phases {r['phase_group']}",
                'Sample': f"{r['sample']}"
                          + (f" / {r['stratum']}" if r['stratum'] not in
                             ('China', 'WNP') else ''),
                'n': int(r['n']) if np.isfinite(r['n']) else '',
                'Observed estimate': _fmt_est(r),
                'Effect size (rate ratio)': _fmt_effect(r),
                '95% year-block bootstrap CI': _fmt_ci(r),
                'Exact two-sided permutation p': _fmt_pq(r['p_perm']),
                'FDR-adjusted q': _fmt_pq(r['q_fdr']),
                'Statistical interpretation': _interp(r['p_perm'], r['q_fdr']),
            })
    block_a = pd.DataFrame(rows_a)

    rows_b = []
    for _, r in ct.iterrows():
        if r['metric'].startswith('conditional'):
            v12, v56 = f"{r['value_12']:.1f}%", f"{r['value_56']:.1f}%"
            dunit = 'pp'
        else:
            v12, v56 = f"{r['value_12']:.1f}", f"{r['value_56']:.1f}"
            dunit = 'per 1000 phase-days'
        rr = (f"{r['rate_ratio']:.2f} ({r['rate_ratio_lo']:.2f}-"
              f"{r['rate_ratio_hi']:.2f})" if np.isfinite(r['rate_ratio'])
              else 'undefined (zero rate in phases 1-2)')
        rows_b.append({
            'Figure or analysis': r['analysis'],
            'Metric': r['metric'],
            'MJO phase assigned at': 'genesis' if 'genesis' in r['metric'] or
                                     r['analysis'] in ('fig8a', 'fig8b', 'fig8c')
                                     else 'landfall',
            'Phase group or contrast': 'Phases 5-6 vs 1-2',
            'Sample': f"{r['sample']}"
                      + (f" / {r['stratum']}" if r['stratum'] not in
                         ('China', 'WNP') else ''),
            'Observed estimate': f"{v12} -> {v56}",
            'Effect size (rate ratio)': rr,
            '95% year-block bootstrap CI':
                f"diff {r['diff']:.1f} {dunit} "
                f"({r['diff_lo']:.1f}-{r['diff_hi']:.1f})",
            'Exact two-sided permutation p': _fmt_pq(r['p_perm']),
            'FDR-adjusted q': _fmt_pq(r['q_fdr']),
            'Statistical interpretation': _interp(r['p_perm'], r['q_fdr']),
        })
    block_b = pd.DataFrame(rows_b)

    block_a.insert(0, 'Block', 'A: per-group tests')
    block_b.insert(0, 'Block', 'B: pre-specified contrasts')
    t = pd.concat([block_a, block_b], ignore_index=True)
    t.to_csv(C.table_path("TableS2_summary_tests.csv"), index=False)
    note = ("Table S2. Summary of year-block permutation tests and 95% "
            "year-block bootstrap confidence intervals for the phase-stratified "
            "statistics outside Figure 3.\n\n"
            "Note. Block A lists the per-group tests (each figure family "
            "corrected separately with the Benjamini-Hochberg FDR procedure). "
            "Block B lists the pre-specified phases 5-6 versus phases 1-2 "
            "contrasts: the effect size is the rate ratio with its 95% CI, and "
            "the CI column reports the absolute rate difference (per 1,000 "
            "phase-days, or percentage points pp for proportions) with its 95% "
            "CI; exact p values refer to the two-sided year-block permutation "
            "test of the rate difference. q < 0.05 is interpreted as "
            "statistically significant.")
    _write_md("TableS2_summary_tests", note, t)
    return t


def _write_md(name, title, df):
    with open(C.table_path(name + ".md"), 'w') as f:
        f.write(f"**{title}**\n\n")
        if tabulate is not None:
            f.write(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
        else:
            f.write(df.to_markdown(index=False))
        f.write("\n")


def main():
    # remove a stale sample-size table left by an earlier run
    for stale in ("TableS1_phase_intensity_sample_sizes.csv",
                  "TableS1_phase_intensity_sample_sizes.md"):
        p = Path(C.table_path(stale))
        if p.exists():
            p.unlink()

    print("=== Table 2 ===")
    print(table2().to_string(index=False))
    print("\n=== Table 3 ===")
    print(table3().to_string(index=False))
    print("\n=== Table S1 (first 12 rows) ===")
    print(table_s1().head(12).to_string(index=False))
    print("\n=== Table S2 (block counts) ===")
    t = table_s2()
    print(t['Block'].value_counts().to_string())
    print(f"\nsaved tables to {C.TABLES_DIR}")


if __name__ == "__main__":
    main()
