"""01 — Table 2, Table 3, and Table S1.

Reads data03/event_table.csv and writes:
  tables03/Table2_counts_TCP_by_intensity.csv|md
      Storm counts + TCP stats by intensity category, separately for LMI and for
      intensity-at-first-Chinese-landfall. Columns: n, share %,
      total TCP (10^6 m^3), mean event TCP (90% CI), median, coastal/inland totals.
  tables03/Table3_LMI_vs_landfall_crosstab.csv|md
      Cross-classification of LMI x intensity-at-landfall with counts and row %.
  tables03/TableS1_phase_intensity_sample_sizes.csv|md
      Per phase-group x intensity: n, NDAYS, residence-normalized rate, bootstrap CI.

All figures are restricted to Jun-Oct (analysis season). Phase-stratified Table S1
additionally requires active MJO at landfall (amplitude >= 1).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import config as C
from lib import bootstrap as bs
from lib import mjo as mj

try:
    from tabulate import tabulate
except Exception:
    tabulate = None


def _fmt_ci(lo, hi, unit=''):
    if not np.isfinite(lo):
        return f"—{unit}"
    return f"{lo:.1f}–{hi:.1f}{unit}"


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
    bucket is included and the share sums to 100% over all storms. LMI never has
    cat 0, so its rows stay [1,2,3]."""
    nd = mj.active_days_per_group(months=C.SEASON_MONTHS)  # for reference
    rows = []
    for cat in (cats if cats is not None else C.CATEGORY_ORDER):
        sub = df[df[cat_col] == cat]
        label = C.CATEGORY_FULL[cat] if cat in C.CATEGORY_FULL else 'TD'
        s = bs.summarize(sub['tcp_total'].values)
        # coastal / inland mean event TCP
        c_tot = sub['coastal_total'].values
        i_tot = sub['inland_total'].values
        cs = bs.summarize(c_tot); is_ = bs.summarize(i_tot)
        rows.append({
            'Intensity category': f"{label} ({'LMI' if cat_col=='lmi_category' else 'landfall'})",
            'No. storms': s['n'],
            'Share of landfalls (%)': round(100 * s['n'] / max(1, len(df)), 1),
            'Total TCP (10^6 m^3)': int(round(s['sum'])),   # int -> no sci-notation in .md
            'Mean event TCP (90% CI, 10^6 m^3)': f"{s['mean']:.1f} ({s['ci_lo']:.1f}–{s['ci_hi']:.1f})",
            'Median event TCP (10^6 m^3)': round(s['median'], 1),
            'Coastal mean event TCP (10^6 m^3)': round(cs['mean'], 1),
            'Inland mean event TCP (10^6 m^3)': round(is_['mean'], 1),
        })
    return rows


def table2():
    df = load_events(jjaso=True, active=False)
    rows = tcp_rows_for(df, 'lmi_category') + tcp_rows_for(df, 'landfall_wind_category', cats=[0, 1, 2, 3])
    t = pd.DataFrame(rows)
    t.to_csv(C.table_path("Table2_counts_TCP_by_intensity.csv"), index=False)
    _write_md("Table2_counts_TCP_by_intensity",
              "Table 2. Storm counts and tropical-cyclone precipitation statistics by "
              "intensity category (Jun-Oct, 1960-2024). TCP integrated over China land "
              "cells; coastal = within 200 km of coastline.",
              t)
    return t


def table3():
    df = load_events(jjaso=True, active=False)
    lab = {0: 'TD', 1: 'Weak', 2: 'Moderate', 3: 'Super'}
    ct = pd.crosstab(df['lmi_category'], df['landfall_wind_category'])
    ct = ct.reindex(index=[1, 2, 3], columns=[0, 1, 2, 3], fill_value=0)
    rows = []
    for lmi in [1, 2, 3]:
        rowtot = int(ct.loc[lmi].sum())   # full LMI count (TD-at-landfall now included)
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


def table_s1():
    df = load_events(jjaso=True, active=True)
    nd = mj.active_days_per_group(months=C.SEASON_MONTHS)
    # precompute residence-test results per intensity slice (own phase counts + null)
    res_by_cat = {}
    for cat in [0] + C.CATEGORY_ORDER:
        sub = df if cat == 0 else df[df['lmi_category'] == cat]
        counts = {gg: int((sub['group_landfall'] == gg).sum()) for gg in C.GROUP_KEY}
        res_by_cat[cat] = mj.residence_permutation_test(counts, nd)
    rows = []
    for g in C.GROUP_KEY:
        sub_g = df[df['group_landfall'] == g]
        for cat in [0] + C.CATEGORY_ORDER:   # 0 = all-TC
            if cat == 0:
                sub = sub_g
                lab = 'All TCs'
            else:
                sub = sub_g[sub_g['lmi_category'] == cat]
                lab = C.CATEGORY_FULL[cat]
            s = bs.summarize(sub['tcp_total'].values)
            r = res_by_cat[cat][g]
            rows.append({
                'Phase group': C.group_to_label(g),
                'NDAYS': nd[g],
                'Intensity': lab,
                'n storms': s['n'],
                'rate per 1000 phase-days': round(r['rate'], 2),
                '90% CI': _fmt_ci(r['ci_lo'], r['ci_hi']),
                'sig (a=0.10)': r['stars'],
                'mean event TCP (10^6 m^3)': round(s['mean'], 1) if s['n'] else '—',
            })
    t = pd.DataFrame(rows)
    t.to_csv(C.table_path("TableS1_phase_intensity_sample_sizes.csv"), index=False)
    _write_md("TableS1_phase_intensity_sample_sizes",
              "Table S1. Sample sizes, phase residence days (NDAYS), residence-normalized "
              "landfall rate per 1000 phase-days with 90% permutation CI, significance "
              "(a=0.10), and mean event TCP by phase group x LMI category "
              "(Jun-Oct, active MJO at landfall).",
              t)
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
    print("=== Table 2 ===")
    print(table2().to_string(index=False))
    print("\n=== Table 3 ===")
    print(table3().to_string(index=False))
    print("\n=== Table S1 (head) ===")
    t = table_s1()
    print(t.head(12).to_string(index=False))
    print(f"\nsaved tables to {C.TABLES_DIR}")


if __name__ == "__main__":
    main()
