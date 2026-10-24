"""Write data/data03/fig_doc_data.json, the plotting values for Figures 1-9 and S1-S4.

Inference values (point estimates, 95% CIs, permutation p, BH q) are read from
r2_stats_pergrp.csv and r2_stats_contrast.csv. Map ranges, simple counts and
the coastal/inland sensitivity are computed in this script.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import xarray as xr

import config as C
from lib import stats_yb as S
from lib import tcp as tcp_lib

R = 3          # default rounding for ratios
out = {}

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
df_all = pd.read_csv(C.data_path("event_table.csv"))              # 490 landfall storms
df_jjaso = df_all[df_all['in_jjaso']].copy()                      # 464
df = df_all[df_all['in_jjaso'] & df_all['active_landfall']].reset_index(drop=True)  # 258

st = pd.read_csv(C.data_path("r2_stats_pergrp.csv"))
ct = pd.read_csv(C.data_path("r2_stats_contrast.csv"))
blocks = S.load_blocks()
ND = {g: int(blocks['D_total'][i]) for i, g in enumerate(C.GROUP_KEY)}


def fam(family, sample=None, stratum=None):
    """{group: rounded stats} for one inference family (optionally one sample
    or stratum). Includes the year-block CI, permutation p and BH q."""
    q = st[st['family'] == family]
    if sample is not None:
        q = q[q['sample'] == sample]
    if stratum is not None:
        q = q[q['stratum'] == stratum]
    d = {}
    for _, r in q.iterrows():
        d[r['phase_group']] = dict(
            est=round(float(r['est']), 2 if r['metric'] != 'ace_ratio' else 3),
            ci=[round(float(r['est_lo']), 2), round(float(r['est_hi']), 2)],
            n=int(r['n']) if np.isfinite(r['n']) else None,
            D=int(r['D']) if np.isfinite(r['D']) else None,
            p=round(float(r['p_perm']), 4), q=round(float(r['q_fdr']), 4))
    assert len(d) == 4, (family, sample, stratum, len(d))
    return d


# ===========================================================================
# Fig 1  (a ACE, b landfall by LMI, c total TCP, d mean TCP)
# ===========================================================================
fig1 = {
    'a_ace_all_wnp': fam('fig1a', 'All WNP TCs'),
    'a_ace_landfall': fam('fig1a', 'China-landfall subset'),
    'b_landfall_by_lmi': {c: fam('fig1b', f'{c} TC') for c in
                          ('Weak', 'Moderate', 'Super')},
    'c_total_tcp': fam('tcp_total'),
    'd_mean_tcp': fam('tcp_mean'),
}
out['fig1'] = fig1

# ===========================================================================
# Fig 2  (landfall-point counts per phase x intensity)
# ===========================================================================
fig2 = {}
for g in C.GROUP_KEY:
    sub = df[df['group_landfall'] == g]
    fig2[g] = {C.CATEGORY_LABEL[c]: int((sub['lmi_category'] == c).sum())
               for c in C.CATEGORY_ORDER}
    fig2[g]['total'] = int(len(sub))
out['fig2'] = fig2

# ===========================================================================
# Fig 3  (regional landfall rate ratio; 48 pre-specified comparisons)
# ===========================================================================
fig3 = {'panels': {}}
panel_sets = {'a_all': ('All TCs', df)}
for cat, key in [(1, 'b_weak'), (2, 'c_moderate'), (3, 'd_major')]:
    panel_sets[key] = (C.CATEGORY_FULL[cat], df[df['lmi_category'] == cat])
for key, (sample, sub) in panel_sets.items():
    panel = {'n': int(len(sub)), 'regions': {}}
    for region in C.REGION_ORDER:
        d = fam('fig3', sample=sample, stratum=region)
        panel['regions'][region] = d
    fig3['panels'][key] = panel
out['fig3'] = fig3

# ===========================================================================
# Fig 4  (TCP by intensity: maps + storm-level + coastal/inland)
# ===========================================================================
fig4 = {'stormlevel': {}, 'coastal_inland': {}, 'map_p99_mm': {}}
ftot = {}
for cat in C.CATEGORY_ORDER:
    lab = C.CATEGORY_LABEL[cat]
    codes = [f"{c:04d}" for c in df[df['lmi_category'] == cat]['chinese_code']]
    s, cnt, lat, lon = tcp_lib.accumulate_event_fields(codes, land_only=True)
    mean = np.where(cnt > 0, s / np.where(cnt == 0, 1, cnt), np.nan)
    ftot[lab] = (s, mean)
    fig4['map_p99_mm']['total_' + lab] = round(float(np.nanpercentile(s, 99)), 1)
    fig4['map_p99_mm']['mean_' + lab] = round(float(np.nanpercentile(np.abs(mean), 99)), 1)
    sub = df[df['lmi_category'] == cat]
    sm = S.boot_scalar_stats(sub['tcp_total'].values,
                             pd.to_datetime(sub['landfall_time']))
    fig4['stormlevel'][lab] = dict(
        n=sm['n'], mean=round(sm['mean'] / 1e3, 1), median=round(sm['median'] / 1e3, 1),
        mean_ci=[round(sm['mean_lo'] / 1e3, 1), round(sm['mean_hi'] / 1e3, 1)],
        median_ci=[round(sm['median_lo'] / 1e3, 1), round(sm['median_hi'] / 1e3, 1)],
        units='10^9 m^3')
    fig4['coastal_inland'][lab] = dict(
        coastal=round(float(sub['coastal_total'].mean() / 1e3), 1),
        inland=round(float(sub['inland_total'].mean() / 1e3), 1),
        units='10^9 m^3')
fig4['vmax_total_mm'] = round(max(float(np.nanpercentile(ftot[c][0], 99))
                                  for c in C.CATEGORY_LABEL.values()), 1)
fig4['vmax_mean_mm'] = round(max(float(np.nanpercentile(np.abs(ftot[c][1]), 99))
                                 for c in C.CATEGORY_LABEL.values()), 1)
out['fig4'] = fig4

# ===========================================================================
# Fig 5  (phase decomposition)
# ===========================================================================
overall_mean = float(df['tcp_total'].mean())
overall_med = float(df['tcp_total'].median())
fig5 = {'a_frequency': fam('fig5a'),
        'b_total_tcp': fam('tcp_total'),
        'c_mean': fam('tcp_mean'),
        'd_median': fam('tcp_median'),
        'overall_mean': round(overall_mean / 1e3, 1),
        'overall_median': round(overall_med / 1e3, 1)}
for g in C.GROUP_KEY:
    fig5['c_mean'][g]['pct_dev'] = round(
        (fig5['c_mean'][g]['est'] * 1e6 - overall_mean) / overall_mean * 100, 0)
    fig5['d_median'][g]['pct_dev'] = round(
        (fig5['d_median'][g]['est'] * 1e6 - overall_med) / overall_med * 100, 0)
out['fig5'] = fig5

# ===========================================================================
# Fig 6  (genesis locations + pre-landfall tracks by landfall region)
# ===========================================================================
fig6 = {}
for region in C.REGION_ORDER:
    sub = df[df['landfall_region'] == region]
    glat, glon = sub['genesis_lat'].values, sub['genesis_lon'].values
    stl = S.boot_scalar_stats(sub['pre_landfall_track_km'].values,
                              pd.to_datetime(sub['landfall_time']))
    fig6[region] = dict(
        n=int(len(sub)),
        genesis_lat_mean=round(float(glat.mean()), 1),
        genesis_lat_median=round(float(np.median(glat)), 1),
        genesis_lon_mean=round(float(glon.mean()), 1),
        genesis_lon_median=round(float(np.median(glon)), 1),
        track_len_mean_km=round(stl['mean'], 0),
        track_len_mean_ci=[round(stl['mean_lo'], 0), round(stl['mean_hi'], 0)],
        track_len_median_km=round(stl['median'], 0))
out['fig6'] = fig6

# ===========================================================================
# Fig 7  (genesis-region composition + MJO-phase genesis rates)
# ===========================================================================
def _ab(sub):
    """Panel (a)/(b): landfall destination shares by genesis region."""
    d = {}
    for lr in ('South China', 'East/North China'):
        d[lr] = {}
        for gr in C.GENESIS_ORDER:
            gsub = sub[sub['formation_region'] == gr]
            tot = len(gsub)
            succ = int((gsub['landfall_region2'] == lr).sum())
            d[lr][gr] = dict(n=succ, tot=tot,
                             pct=round(100 * succ / tot, 1) if tot else 0.0)
    return d


fig7 = {'a_weak': _ab(df[df['lmi_category'] == 1]),
        'b_mod_maj': _ab(df[df['lmi_category'].isin([2, 3])])}
for gr in C.GENESIS_ORDER:
    fig7['c_all_wnp'] = fig7.get('c_all_wnp', {})
    fig7['c_all_wnp'][gr] = fam('fig7c', 'All WNP TCs', stratum=gr)
    fig7['d_landfall'] = fig7.get('d_landfall', {})
    fig7['d_landfall'][gr] = fam('fig7d', 'China-landfall subset', stratum=gr)
out['fig7'] = fig7

# ===========================================================================
# Fig 8  (all-WNP genesis vs China landfall)
# ===========================================================================
n_wnp = int(sum(fam('fig8a', 'All WNP TCs')[g]['n'] for g in C.GROUP_KEY))
n_lf = int(sum(fam('fig8b', 'China-landfall subset')[g]['n'] for g in C.GROUP_KEY))
fig8 = {
    'a_all_wnp_genesis_rate': fam('fig8a', 'All WNP TCs'),
    'b_landfall_subset_genesis_rate': fam('fig8b', 'China-landfall subset'),
    'c_conditional_landfall_proportion': fam('fig8c', 'WNP->China'),
    'n_all_wnp': n_wnp, 'n_landfall': n_lf,
    'overall_landfall_pct': round(100 * n_lf / n_wnp, 1),
}
out['fig8'] = fig8

# ===========================================================================
# Fig 9  (JJASO all-active-day circulation composites, year-block FDR)
# ===========================================================================
ds9 = xr.open_dataset(C.data_path("composite_jjaso.nc"))
fig9 = {
    'domain': [30, 180, 0, 60], 'season': 'Jun-Oct',
    'day_selection': 'ALL MJO-active days per phase group (no TC-track-day conditioning)',
    'sig_method': ('per-grid-point year-block permutation (B=10000, seed=20261019) '
                   '+ Benjamini-Hochberg FDR per field x phase panel; q < 0.05'),
    'D': {g: int(ds9['n_days'].values[i]) for i, g in enumerate(C.GROUP_KEY)},
    'panels': {}}
for var, label in [('vort_850', '850-hPa relative vorticity (sigma)'),
                   ('slp', 'Mean sea-level pressure (sigma)'),
                   ('uwnd_850', '850-hPa zonal wind anomaly (m/s)'),
                   ('vwnd_850', '850-hPa meridional wind anomaly (m/s)')]:
    fig9['panels'][label] = {}
    for i, g in enumerate(C.GROUP_KEY):
        a = ds9[f'{var}_anom'].values[i]
        q = ds9[f'{var}_q'].values[i]
        fig9['panels'][label][g] = dict(
            min=round(float(np.nanmin(a)), 2), max=round(float(np.nanmax(a)), 2),
            mean=round(float(np.nanmean(a)), 3),
            sig_frac=round(float((q < 0.05).mean()), 3))
u = ds9['uwnd_850_anom'].values; v = ds9['vwnd_850_anom'].values
sp = np.hypot(u, v)
fig9['wind'] = {g: dict(max_speed=round(float(sp[i].max()), 1),
                        p95_speed=round(float(np.nanpercentile(sp[i], 95)), 1))
                for i, g in enumerate(C.GROUP_KEY)}
ds9.close()
out['fig9'] = fig9

# ===========================================================================
# Supplementary figures (S1-S4)
# ===========================================================================
# ---- S1 : monthly distribution (all 490 storms) ----
figS1 = {'n_total': int(len(df_all)), 'monthly': {}}
_jjaso = 0
for m in range(1, 13):
    by_cat = {C.CATEGORY_LABEL[c]: int(((df_all['month'] == m) &
              (df_all['lmi_category'] == c)).sum()) for c in C.CATEGORY_ORDER}
    tot = sum(by_cat.values())
    figS1['monthly'][m] = dict(by_cat=by_cat, total=tot)
    if 6 <= m <= 10:
        _jjaso += tot
figS1['jjaso_n'] = _jjaso
figS1['jjaso_share'] = round(100 * _jjaso / max(1, len(df_all)), 1)
out['figS1'] = figS1

# ---- S2 : LMI vs intensity-at-landfall ----
_LF = {0: 'TD', 1: 'Weak', 2: 'Moderate', 3: 'Super'}
ct2 = pd.crosstab(df_jjaso['lmi_category'], df_jjaso['landfall_wind_category']).reindex(
    index=[1, 2, 3], columns=[0, 1, 2, 3], fill_value=0).values
_lmi_v = df_jjaso['lmi_category'].to_numpy()
_lf_v = df_jjaso['landfall_wind_category'].to_numpy()
figS2 = {'n': int(len(df_jjaso)),
         'contingency': {C.CATEGORY_LABEL[r + 1]: {_LF[c]: int(ct2[r, c])
                         for c in range(4)} for r in range(3)},
         'diag_nochange': int(np.sum(_lf_v == _lmi_v)),
         'weaken': int(np.sum(_lf_v < _lmi_v)),
         'intensify': int(np.sum(_lf_v > _lmi_v)),
         'regional_by_landfall_intensity': {}, 'tcp_by_landfall_intensity': {}}
for cat in C.CATEGORY_ORDER:
    lab = C.CATEGORY_LABEL[cat]
    sub = df_jjaso[df_jjaso['landfall_wind_category'] == cat]
    figS2['regional_by_landfall_intensity'][lab] = {
        r: round(100 * int((sub['landfall_region'] == r).sum()) / max(1, len(sub)), 1)
        for r in C.REGION_ORDER}
    sm = S.boot_scalar_stats(sub['tcp_total'].values,
                             pd.to_datetime(sub['landfall_time'])) if len(sub) else None
    figS2['tcp_by_landfall_intensity'][lab] = dict(
        n=int(len(sub)),
        median=round(sm['median'] / 1e3, 1) if sm else None,
        median_ci=[round(sm['median_lo'] / 1e3, 1), round(sm['median_hi'] / 1e3, 1)]
        if sm else None, units='10^9 m^3')
out['figS2'] = figS2

# ---- S3 : IBTrACS robustness (cached ibtracs_matched.csv) ----
S3 = {}
_ib = Path(C.data_path("ibtracs_matched.csv"))
if _ib.exists():
    m = pd.read_csv(_ib)
    ctib = pd.crosstab(m['lmi_category'], m['ib_category']).reindex(
        index=[1, 2, 3], columns=[0, 1, 2, 3], fill_value=0).values
    ma = m[m['active_landfall'] == True]
    S3 = {'n_in_jjaso': int(len(df_jjaso)), 'n_matched': int(len(m)),
          'match_rate_pct': round(100 * len(m) / max(1, len(df_jjaso)), 1),
          'median_match_km': round(float(np.median(m['match_km'])), 0),
          'wind_corr_r': round(float(np.corrcoef(m['cma_lmi'], m['ib_wind_ms'])[0, 1]), 2),
          'n_active_matched': int(len(ma)),
          'cma_ib_contingency': {
              C.CATEGORY_LABEL[r + 1]: {('TD' if c == 0 else C.CATEGORY_LABEL[c]):
                                        int(ctib[r, c]) for c in range(4)}
              for r in range(3)},
          'major_share_pct': {}}
    for col, lab in [('lmi_category', 'CMA'), ('ib_category', 'IBTrACS')]:
        S3['major_share_pct'][lab] = {
            g: round(100 * int((ma[ma['group_landfall'] == g][col] == 3).sum())
               / max(1, int((ma['group_landfall'] == g).sum())), 1)
            for g in C.GROUP_KEY}
out['figS3'] = S3

# ---- S4 : coastal/inland TCP sensitivity to the distance threshold ----
df_s4 = df.copy()
figS4 = {'thresholds_km': [100, 200, 300], 'n_storms': 0, 'stats': {}}
_s4_dates = pd.to_datetime(df_s4['landfall_time'])
for k, (code, cat) in enumerate(zip([f"{c:04d}" for c in df_s4['chinese_code']],
                                    df_s4['lmi_category'].values)):
    if cat not in C.CATEGORY_ORDER:
        continue
    try:
        res = tcp_lib.coastal_inland_by_threshold(code, [100.0, 200.0, 300.0])
    except Exception:
        continue
    figS4['n_storms'] += 1
    for thr in (100.0, 200.0, 300.0):
        coast, inl = res[thr]
        figS4['stats'].setdefault(int(thr), {}).setdefault('coastal', {}).setdefault(cat, []).append(
            (coast, _s4_dates.iloc[k]))
        figS4['stats'][int(thr)]['inland'] = figS4['stats'][int(thr)].get('inland', {})
        figS4['stats'][int(thr)]['inland'].setdefault(cat, []).append((inl, _s4_dates.iloc[k]))
stats_s4 = {}
for thr in (100, 200, 300):
    stats_s4[thr] = {}
    for zone in ('coastal', 'inland'):
        stats_s4[thr][zone] = {}
        for cat in C.CATEGORY_ORDER:
            pairs = figS4['stats'][thr].get(zone, {}).get(cat, [])
            if pairs:
                s = S.boot_scalar_stats(np.array([p[0] for p in pairs], float),
                                        pd.Series([p[1] for p in pairs]))
                stats_s4[thr][zone][C.CATEGORY_LABEL[cat]] = dict(
                    n=s['n'], mean=round(s['mean'] / 1e3, 1),
                    mean_ci=[round(s['mean_lo'] / 1e3, 1), round(s['mean_hi'] / 1e3, 1)],
                    units='10^9 m^3')
figS4['stats'] = stats_s4
out['figS4'] = figS4

# ===========================================================================
# Pre-specified contrasts (all 19) + meta
# ===========================================================================
out['contrasts_5_6_vs_1_2'] = [
    dict(analysis=r['analysis'], metric=r['metric'], sample=r['sample'],
         stratum=r['stratum'], value_12=round(float(r['value_12']), 2),
         value_56=round(float(r['value_56']), 2),
         rate_ratio=round(float(r['rate_ratio']), 2),
         rate_ratio_ci=[round(float(r['rate_ratio_lo']), 2),
                        round(float(r['rate_ratio_hi']), 2)],
         diff=round(float(r['diff']), 2),
         diff_ci=[round(float(r['diff_lo']), 2), round(float(r['diff_hi']), 2)],
         p=round(float(r['p_perm']), 4), q=round(float(r['q_fdr']), 4))
    for _, r in ct.iterrows()]

out['meta'] = {
    'analysis_sample_n': {'landfall_jjaso_active': int(len(df)),
                          'landfall_jjaso': int(len(df_jjaso)),
                          'landfall_all': int(len(df_all)),
                          'wnp_genesis_jjaso_active': n_wnp,
                          'wnp_genesis_landfall_subset': n_lf},
    'ND_JJASO': ND,
    'inference': ('year-block bootstrap 95% CI (B=10000, seed=20261018); '
                  'year-block permutation two-sided empirical p (B=10000, '
                  'seed=20261019); Benjamini-Hochberg FDR within pre-specified '
                  'families; q < 0.05 significant'),
    'fig3_family_size': 48,
    'fig3_q_significant': int((st[st['family'] == 'fig3']['q_fdr'] < 0.05).sum()),
}

with open(C.data_path("fig_doc_data.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print("saved", C.data_path("fig_doc_data.json"))
print("n analysis sample:", len(df), "| ND:", ND)
print("fig8:", n_wnp, n_lf, f"{100 * n_lf / n_wnp:.1f}%")
print("fig3 q<0.05:", out['meta']['fig3_q_significant'], "/48")
print("fig9 D:", fig9['D'])
