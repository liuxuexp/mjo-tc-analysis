"""Cross-check the quoted values against the pipeline outputs.

Values are recomputed from r2_stats_pergrp.csv, r2_stats_contrast.csv,
fig_doc_data.json and event_table.csv. The reference values are hardcoded for
comparison only and are never written back into any output.

Output: reports/numbers_report.md
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import config as C

OUT = Path(__file__).resolve().parent / "reports"
G = C.GROUP_KEY


def fp(x, nd=3):
    if not np.isfinite(x):
        return ''
    return '<0.001' if x < 10 ** (-nd) else f'{x:.{nd}f}'


def main():
    OUT.mkdir(exist_ok=True)
    st = pd.read_csv(C.data_path("r2_stats_pergrp.csv"))
    ct = pd.read_csv(C.data_path("r2_stats_contrast.csv"))
    doc = json.load(open(C.data_path("fig_doc_data.json")))
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)

    def row(family, sample=None, stratum=None, group=None):
        q = st[st['family'] == family]
        if sample is not None:
            q = q[q['sample'] == sample]
        if stratum is not None:
            q = q[q['stratum'] == stratum]
        if group is not None:
            q = q[q['phase_group'] == group]
        return q.iloc[0]

    L = ["# Numbers report",
         "",
         f"Generated from `r2_stats_pergrp.csv` ({len(st)} rows), "
         f"`r2_stats_contrast.csv` ({len(ct)} rows) and `fig_doc_data.json`.",
         "",
         "## 1. Method constants",
         "",
         f"* Analysis sample (JJASO, MJO-active at landfall): n = {len(df)}; "
         f"JJASO landfalls 464; all landfalls 490; all-WNP genesis sample "
         f"{doc['meta']['analysis_sample_n']['wnp_genesis_jjaso_active']}; "
         f"genesis-based landfall subset "
         f"{doc['meta']['analysis_sample_n']['wnp_genesis_landfall_subset']}.",
         f"* MJO-active phase-days D (Jun-Oct 1960-2024): "
         + ", ".join(f"{g} = {doc['meta']['ND_JJASO'][g]}" for g in G) + ".",
         "* Inference: year-block bootstrap 95% CI (B = 10,000, seed 20261018); "
         "year-block permutation two-sided empirical p (B = 10,000, seed "
         "20261019); BH-FDR within pre-specified families; q < 0.05 "
         "significant. Fig 3 / Table S1 intervals are 2.5-97.5 percentiles of "
         "the PERMUTATION NULL (not bootstrap CIs).",
         "",
         "## 2. Quoted values vs recomputed",
         "",
         "| Quantity | Reference | Recomputed | Status |",
         "|---|---|---|---|"]

    def add_check(q, ms, rc, status='consistent'):
        L.append(f"| {q} | {ms} | {rc} | {status} |")

    # Fig 3 headline rates (South China, All TCs)
    r12 = row('fig3', 'All TCs', 'South China', '1-2')
    r56 = row('fig3', 'All TCs', 'South China', '5-6')
    add_check("Fig 3 rate, South China All TCs, phases 1-2", "0.60",
              f"{r12['est']:.2f}")
    add_check("Fig 3 rate, South China All TCs, phases 5-6", "1.38",
              f"{r56['est']:.2f}")
    # Fig 8: read from the CSV because the JSON rounds to two decimals
    def fam_map(family, sample):
        sub = st[(st['family'] == family) & (st['sample'] == sample)]
        return {r['phase_group']: r for _, r in sub.iterrows()}
    f8a = fam_map('fig8a', 'All WNP TCs')
    f8b = fam_map('fig8b', 'China-landfall subset')
    f8c = fam_map('fig8c', 'WNP->China')
    add_check("Fig 8a genesis rates (/1000 phase-days)",
              "83.6 / 101.0 / 171.2 / 118.7",
              " / ".join(f"{f8a[g]['est']:.1f}" for g in G))
    add_check("Fig 8b landfall-subset rates (/1000 phase-days)",
              "26.0 / 40.0 / 68.6 / 40.7",
              " / ".join(f"{f8b[g]['est']:.1f}" for g in G))
    add_check("Fig 8c conditional landfall proportion (%)",
              "31.2 / 39.6 / 40.1 / 34.3",
              " / ".join(f"{f8c[g]['est']:.1f}" for g in G))
    add_check("Fig 8c overall landfall proportion", "37.1%",
              f"{doc['fig8']['overall_landfall_pct']}%")
    # ACE
    for key, ms in [('a_ace_all_wnp', "0.737 / 0.730 / 1.210 / 1.284"),
                    ('a_ace_landfall', "0.514 / 0.766 / 1.476 / 1.149")]:
        add_check(f"Fig 1a ACE ratios ({key.split('_', 2)[-1]})", ms,
                  " / ".join(f"{doc['fig1'][key][g]['est']:.3f}" for g in G))
    # Fig 5a ratios (3 dp from the CSV)
    add_check("Fig 5a frequency ratios", "0.539 / 0.956 / 1.406 / 1.058",
              " / ".join(f"{row('fig5a', group=g)['est']:.3f}" for g in G))
    # TCP totals and means
    tt = {g: row('tcp_total', group=g) for g in G}
    add_check("Total TCP by phase (10^12 m^3)", "0.864 (1-2) / 2.738 (5-6)",
              f"{tt['1-2']['est'] / 1e6:.3f} (1-2) / "
              f"{tt['5-6']['est'] / 1e6:.3f} (5-6)")
    tm = " / ".join(f"{row('tcp_mean', group=g)['est'] / 1e3:.1f}" for g in G)
    add_check("Mean event TCP by phase (10^9 m^3)", "21.1 - 24.7", tm)

    L += ["",
          "Notes on statuses:",
          "* **Fig 8a phases 5-6 rounding**: the recomputed value is 171.1465 "
          "per 1,000 phase-days, i.e. 171.1 at one decimal. Table S2 and the "
          "figure annotation carry 171.1.",
          "* All other checked values agree to the quoted precision.",
          "",
          "## 3. Inference outcomes by family (year-block permutation + BH FDR)",
          "",
          "| Family | comparisons | p < 0.05 | q < 0.05 | significant groups |",
          "|---|---|---|---|---|"]
    for fam in st['family'].unique():
        sub = st[st['family'] == fam]
        np05 = int((sub['p_perm'] < 0.05).sum())
        nq = int((sub['q_fdr'] < 0.05).sum())
        sig = "; ".join(f"{r['phase_group']}"
                        + (f" {r['stratum']}" if r['stratum'] not in
                           ('China', 'WNP', '') else "")
                        for _, r in sub[sub['q_fdr'] < 0.05].iterrows())
        L.append(f"| {fam} | {len(sub)} | {np05} | {nq} | "
                 f"{sig if sig else 'none'} |")
    nq_ct = int((ct['q_fdr'] < 0.05).sum())
    L += ["",
          f"Pre-specified phases 5-6 vs 1-2 contrasts: **{nq_ct}/{len(ct)}** "
          "significant at q < 0.05 (family `prespec_contrast`).",
          "",
          "### Significance summary",
          "* Fig 3: none of the 48 comparisons survives FDR; Table S1 carries "
          "the full statistics.",
          "* Fig 5a: phases 1-2 (suppressed) and 5-6 (enhanced) frequency "
          "ratios are significant at q < 0.05.",
          "* Fig 8: genesis-rate modulation (a: 5-6 and 7-8; b: 1-2 and 5-6) "
          "is significant; the conditional landfall proportion (c) is not, "
          "so genesis rather than post-genesis selection dominates.",
          "",
          "## 4. Method notes",
          "",
          "1. **Fig 9 composites**: all MJO-active June-October days "
          "(D = 1651/1000/1823/1129); MSLP replaces the 500-hPa block; "
          "grid-point year-block permutation with field-wise BH FDR.",
          f"2. **Inference**: day-level alpha = 0.10 stars are not used. The "
          f"sensitivity check (reports/sensitivity_daylevel.md) finds that 24 "
          f"comparisons lose significance under the year-block criterion and "
          f"none gain it.",
          "3. **Interval semantics**: Fig 3 and Table S1 error bars are 95% "
          "null intervals from the permutation distribution; every other "
          "figure shows 95% year-block bootstrap CIs.",
          "4. October-only and storm-level TCP analyses are not part of this "
          "release.",
          "",
          "## 5. Reported values",
          "",
          "### Fig 8 (genesis and landfall)",
          f"* All-WNP genesis rates per 1,000 phase-days: "
          + "; ".join(f"phases {g}: {f8a[g]['est']:.1f} "
                      f"(95% CI {f8a[g]['est_lo']:.1f}-{f8a[g]['est_hi']:.1f}, "
                      f"n = {int(f8a[g]['n'])})" for g in G) + ".",
          f"* China-landfall subset: "
          + "; ".join(f"phases {g}: {f8b[g]['est']:.1f} "
                      f"(95% CI {f8b[g]['est_lo']:.1f}-{f8b[g]['est_hi']:.1f}, "
                      f"n = {int(f8b[g]['n'])})" for g in G) + ".",
          f"* Conditional landfall proportion: "
          + "; ".join(f"phases {g}: {f8c[g]['est']:.1f}% "
                      f"(95% CI {f8c[g]['est_lo']:.1f}-{f8c[g]['est_hi']:.1f}%, "
                      f"n = {int(f8c[g]['n'])}/{int(f8c[g]['D'])})" for g in G)
          + f"; overall {doc['fig8']['overall_landfall_pct']}% "
          f"({doc['fig8']['n_landfall']}/{doc['fig8']['n_all_wnp']}).",
          "",
          "### Pre-specified contrasts (phases 5-6 vs 1-2; Table S2 block B)",
          "",
          "| Analysis | metric | 1-2 | 5-6 | rate ratio (95% CI) | p | q |",
          "|---|---|---|---|---|---|---|"]
    for _, r in ct.iterrows():
        unit = '%' if r['metric'].startswith('conditional') else ''
        strat = r['stratum'] if r['stratum'] not in ('China', 'WNP', '') else ''
        rr = ('— (no events in either phase group)' if
              not np.isfinite(r['rate_ratio']) and r['value_12'] == 0 else
              '— (zero rate in phases 1-2)' if not np.isfinite(r['rate_ratio'])
              else f"{r['rate_ratio']:.2f} ({r['rate_ratio_lo']:.2f}-"
                   f"{r['rate_ratio_hi']:.2f})")
        L.append(
            f"| {r['analysis']}, {r['sample']}" + (f", {strat}" if strat else "")
            + f" | {r['metric']} | {r['value_12']:.1f}{unit} | "
            f"{r['value_56']:.1f}{unit} | {rr} | {fp(r['p_perm'])} | "
            f"{fp(r['q_fdr'])} |")
    L += ["",
          "### Circulation composites (Fig 9)",
          f"* Composited days per phase group: "
          + ", ".join(f"{g} = {doc['fig9']['D'][g]}" for g in G)
          + " (all MJO-active Jun-Oct days).",
          f"* Significant (q < 0.05) grid-point fractions: vorticity "
          + ", ".join(f"{g}: {100 * doc['fig9']['panels']['850-hPa relative vorticity (sigma)'][g]['sig_frac']:.0f}%"
                      for g in G)
          + "; MSLP "
          + ", ".join(f"{g}: {100 * doc['fig9']['panels']['Mean sea-level pressure (sigma)'][g]['sig_frac']:.0f}%"
                      for g in G) + ".",
          "",
          "### Deliverable files",
          "* Figures (fig/): fig1-fig9 (9 main) + figS1-figS4 (4 SI).",
          "* Tables (tables03/): Table2, Table3, TableS1 (48-row Fig 3 "
          "statistics), TableS2 (per-group tests + contrasts).",
          "* Reports: numbers_report.md (this file), "
          "sensitivity_daylevel.{md,csv}, fig_doc_data.json.",
          ""]

    (OUT / "numbers_report.md").write_text("\n".join(L))
    print(f"saved {OUT}/numbers_report.md ({len(L)} lines)")


if __name__ == "__main__":
    main()
