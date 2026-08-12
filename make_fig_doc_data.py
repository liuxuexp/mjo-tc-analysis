"""Build the per-figure data tables (Figs 1-9 + SI S1-S10) for the figure-reference PDFs.

Recomputes every summary statistic that underlies the main-text and supplementary
figures, using the SAME formulas as the figure scripts, so the
numbers in the PDF match the figures. Dumps to data/fig_doc_data.json.

CI policy: each bootstrap call is seeded with a fresh default_rng(20240601), so
every CI is independently reproducible (point estimates are exact & deterministic;
CIs are within bootstrap noise of the figure, which uses the shared module RNG).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import xarray as xr

import config as C
from lib import mjo as mj
from lib import bootstrap as bs
from lib import tcp as tcp_lib

RNG_SEED = 20240601


def _sum(values):
    return bs.summarize(values, rng=np.random.default_rng(RNG_SEED))


def _med(values):
    return bs.summarize_median(values, rng=np.random.default_rng(RNG_SEED))


def R(g):  # round helper for a group-keyed dict of floats
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()}


out = {}
ND = mj.active_days_per_group(months=C.SEASON_MONTHS)
ND_jj = mj.active_days_per_group(months=C.MAIN_COMP_MONTHS)
out['meta'] = {
    'season': 'Jun-Oct (JJASO), 1960-2024',
    'active_mjo': 'amplitude >= 1',
    'ND_JJASO': ND,
    'ND_JunSep': ND_jj,
    'alpha': C.ALPHA,
    'tcrit_90': C.TCRIT_90,
    'boot_niter': C.BOOT_NITER,
}

df = pd.read_csv(C.data_path("event_table.csv"))
df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)
out['meta']['n_landfall_jjaso_active'] = int(len(df))
out['meta']['lmi_counts'] = {C.CATEGORY_LABEL[c]: int((df['lmi_category'] == c).sum())
                             for c in C.CATEGORY_ORDER}
out['meta']['region_counts'] = {r: int((df['landfall_region'] == r).sum())
                                for r in C.REGION_ORDER}
out['meta']['phase_counts'] = {g: int((df['group_landfall'] == g).sum())
                               for g in C.GROUP_KEY}

ND_arr = np.array([ND[g] for g in C.GROUP_KEY], dtype=float)


# ===========================================================================
# Fig 1
# ===========================================================================
def _storm_ace_by_phase(tracks):
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
    return tracks.groupby(['chinese_code', 'group'], observed=True)['ace'].sum().reset_index()


def _ace_ratio(tracks):
    g = _storm_ace_by_phase(tracks)
    storms = g['chinese_code'].unique()
    ace_g = {gg: float(g[g['group'] == gg]['ace'].sum()) for gg in C.GROUP_KEY}
    tot = sum(ace_g.values()); totd = sum(ND.values())
    null = tot / totd
    ratio = {gg: (ace_g[gg] / ND[gg]) / null if null > 0 else 0.0 for gg in C.GROUP_KEY}
    rng = np.random.default_rng(RNG_SEED)
    storm_idx = {s: i for i, s in enumerate(storms)}
    mat = np.zeros((len(storms), 4))
    for _, r in g.iterrows():
        mat[storm_idx[r['chinese_code']], C.GROUP_KEY.index(r['group'])] += r['ace']
    boot = np.zeros((C.BOOT_NITER, 4))
    for b in range(C.BOOT_NITER):
        idx = rng.integers(0, len(storms), len(storms))
        s = mat[idx].sum(axis=0); bt = s.sum(); bn = bt / totd if totd else 1.0
        boot[b] = np.where(ND_arr > 0, (s / ND_arr) / bn, 0.0)
    lo = np.percentile(boot, 5, axis=0); hi = np.percentile(boot, 95, axis=0)
    n = {gg: int(g[g['group'] == gg]['chinese_code'].nunique()) for gg in C.GROUP_KEY}
    return {gg: dict(ratio=round(ratio[gg], 3), ci_lo=round(float(lo[i]), 3),
                     ci_hi=round(float(hi[i]), 3), n=n[gg], ace=round(ace_g[gg], 1))
            for i, gg in enumerate(C.GROUP_KEY)}


alltr = pd.read_csv(C.TYPHOON_OUT / "all_typhoons_tracks.csv")
lftr = pd.read_csv(C.TYPHOON_OUT / "landfall_typhoons_tracks.csv")
fig1 = {
    'a_ace_all_wnp': _ace_ratio(alltr),
    'a_ace_landfall': _ace_ratio(lftr),
    'b_landfall_by_lmi': {},
    'c_total_tcp': {},
    'd_mean_tcp': {},
}
for cat in C.CATEGORY_ORDER:
    sub = df[df['lmi_category'] == cat]
    counts = {g: int((sub['group_landfall'] == g).sum()) for g in C.GROUP_KEY}
    res = mj.residence_ratio_test(counts, ND, rng=np.random.default_rng(RNG_SEED))
    fig1['b_landfall_by_lmi'][C.CATEGORY_LABEL[cat]] = {
        g: dict(ratio=round(res[g]['ratio'], 3), n=counts[g], stars=res[g]['stars']) for g in C.GROUP_KEY}
for g in C.GROUP_KEY:
    v = df[df['group_landfall'] == g]['tcp_total'].values
    fig1['c_total_tcp'][g] = dict(total=round(float(v.sum()), 0), n=int(len(v)))
    s = _sum(v)
    fig1['d_mean_tcp'][g] = dict(mean=round(s['mean'], 0), ci_lo=round(s['ci_lo'], 0),
                                 ci_hi=round(s['ci_hi'], 0), n=s['n'])
out['fig1'] = fig1


# ===========================================================================
# Fig 2  (landfall-point counts per phase x intensity -- matches event_table)
# ===========================================================================
fig2 = {}
for g in C.GROUP_KEY:
    sub = df[df['group_landfall'] == g]
    fig2[g] = {C.CATEGORY_LABEL[c]: int((sub['lmi_category'] == c).sum()) for c in C.CATEGORY_ORDER}
    fig2[g]['total'] = int(len(sub))
out['fig2'] = fig2


# ===========================================================================
# Fig 3  (regional landfall rate ratio)
# ===========================================================================
fig3 = {'panels': {}}
panel_sets = {'a_all': df}
for cat in C.CATEGORY_ORDER:
    panel_sets[{1: 'b_weak', 2: 'c_moderate', 3: 'd_major'}[cat]] = df[df['lmi_category'] == cat]
for key, sub in panel_sets.items():
    panel = {'n': int(len(sub)), 'regions': {}}
    for region in C.REGION_ORDER:
        obs = {g: int(((sub['group_landfall'] == g) & (sub['landfall_region'] == region)).sum())
               for g in C.GROUP_KEY}
        res = mj.residence_ratio_test(obs, ND, rng=np.random.default_rng(RNG_SEED))
        panel['regions'][region] = {
            g: dict(ratio=round(res[g]['ratio'], 3), n=obs[g],
                    ci_hi=round(res[g]['ci_hi'], 3), stars=res[g]['stars']) for g in C.GROUP_KEY}
    fig3['panels'][key] = panel
out['fig3'] = fig3


# ===========================================================================
# Fig 4  (TCP by intensity: maps + storm-level + coastal/inland)
# ===========================================================================
fig4 = {'total_tcp': {}, 'mean_tcp': {}, 'stormlevel': {}, 'coastal_inland': {}}
ftot, fmean = {}, {}
for cat in C.CATEGORY_ORDER:
    codes = [f"{c:04d}" for c in df[df['lmi_category'] == cat]['chinese_code']]
    s, cnt, lat, lon = tcp_lib.accumulate_event_fields(codes, land_only=True)
    mean = np.where(cnt > 0, s / np.where(cnt == 0, 1, cnt), np.nan)
    ftot[C.CATEGORY_LABEL[cat]] = (s, mean)
    fmean[C.CATEGORY_LABEL[cat]] = mean
    v = df[df['lmi_category'] == cat]['tcp_total'].values
    sm = _sum(v)
    fig4['stormlevel'][C.CATEGORY_LABEL[cat]] = dict(
        n=int(len(v)), mean=round(sm['mean'], 0), median=round(sm['median'], 0),
        ci_lo=round(sm['ci_lo'], 0), ci_hi=round(sm['ci_hi'], 0))
    fig4['coastal_inland'][C.CATEGORY_LABEL[cat]] = dict(
        coastal=round(bs.summarize(df[df['lmi_category'] == c]['coastal_total'],
                    rng=np.random.default_rng(RNG_SEED))['mean'], 0) if False else
        round(float(df[df['lmi_category'] == cat]['coastal_total'].mean()), 0),
        inland=round(float(df[df['lmi_category'] == cat]['inland_total'].mean()), 0))
vmax_tot = max(float(np.nanpercentile(ftot[c][0], 99)) for c in C.CATEGORY_LABEL.values())
vmax_mean = max(float(np.nanpercentile(np.abs(ftot[c][1]), 99)) for c in C.CATEGORY_LABEL.values())
fig4['vmax_total_mm'] = round(vmax_tot, 1)
fig4['vmax_mean_mm'] = round(vmax_mean, 1)
for cat in C.CATEGORY_ORDER:
    lab = C.CATEGORY_LABEL[cat]
    fig4['total_tcp'][lab] = round(float(ftot[lab][0].sum()) / 1e6, 2)  # summed grid total (10^6 m^3 not meaningful; report per-cell max)
# report per-panel grid 99th pct instead
fig4['total_tcp'] = {C.CATEGORY_LABEL[c]: round(float(np.nanpercentile(ftot[C.CATEGORY_LABEL[c]][0], 99)), 1)
                     for c in C.CATEGORY_ORDER}
fig4['mean_tcp'] = {C.CATEGORY_LABEL[c]: round(float(np.nanpercentile(np.abs(ftot[C.CATEGORY_LABEL[c]][1]), 99)), 1)
                    for c in C.CATEGORY_ORDER}
out['fig4'] = fig4


# ===========================================================================
# Fig 5  (phase decomposition)
# ===========================================================================
fig5 = {'a_frequency': {}, 'b_total_tcp': {}, 'c_mean': {}, 'd_median': {}}
counts5 = {g: int((df['group_landfall'] == g).sum()) for g in C.GROUP_KEY}
res5 = mj.residence_ratio_test(counts5, ND, rng=np.random.default_rng(RNG_SEED))
for g in C.GROUP_KEY:
    fig5['a_frequency'][g] = dict(ratio=round(res5[g]['ratio'], 3), n=counts5[g], stars=res5[g]['stars'])
    v = df[df['group_landfall'] == g]['tcp_total'].values
    fig5['b_total_tcp'][g] = dict(total=round(float(v.sum()), 0), n=len(v))
    sm = _sum(v); md = _med(v)
    fig5['c_mean'][g] = dict(mean=round(sm['mean'], 0), ci_lo=round(sm['ci_lo'], 0), ci_hi=round(sm['ci_hi'], 0))
    fig5['d_median'][g] = dict(median=round(md['median'], 0), ci_lo=round(md['ci_lo'], 0), ci_hi=round(md['ci_hi'], 0))
overall_mean = float(df['tcp_total'].mean())
overall_med = _med(df['tcp_total'].values)['median']
fig5['overall_mean'] = round(overall_mean, 0)
fig5['overall_median'] = round(overall_med, 0)
for g in C.GROUP_KEY:
    cm = fig5['c_mean'][g]; dm = fig5['d_median'][g]
    cm['pct_dev_mean'] = round((cm['mean'] - overall_mean) / overall_mean * 100, 0)
    dm['pct_dev_median'] = round((dm['median'] - overall_med) / overall_med * 100, 0)
out['fig5'] = fig5


# ===========================================================================
# Fig 6  (storm-level TCP by LMI x phase)
# ===========================================================================
fig6 = {'a_event_tcp_by_lmi_phase': {}, 'b_precip_area_by_lmi_phase': {},
        'cf_mean_map_vmax_mm': {}}
for cat in C.CATEGORY_ORDER:
    fig6['a_event_tcp_by_lmi_phase'][C.CATEGORY_LABEL[cat]] = {}
    fig6['b_precip_area_by_lmi_phase'][C.CATEGORY_LABEL[cat]] = {}
    for g in C.GROUP_KEY:
        v = df[(df['lmi_category'] == cat) & (df['group_landfall'] == g)]['tcp_total'].values
        ar = df[(df['lmi_category'] == cat) & (df['group_landfall'] == g)]['affected_area'].values
        med = _med(v) if len(v) else dict(median=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0)
        fig6['a_event_tcp_by_lmi_phase'][C.CATEGORY_LABEL[cat]][g] = dict(
            n=int(len(v)), median=(round(med['median'], 0) if med['n'] else None))
        fig6['b_precip_area_by_lmi_phase'][C.CATEGORY_LABEL[cat]][g] = (
            round(float(ar.mean()) / 1e3, 1) if len(ar) else None)
# (c-f) map vmax per phase
fields6 = {}
for g in C.GROUP_KEY:
    codes = [f"{c:04d}" for c in df[df['group_landfall'] == g]['chinese_code']]
    m, lat, lon = tcp_lib.mean_per_storm_field(codes, land_only=True)
    fields6[g] = m
vmax6 = max(float(np.nanpercentile(np.abs(fields6[g]), 99)) for g in C.GROUP_KEY)
for g in C.GROUP_KEY:
    fig6['cf_mean_map_vmax_mm'][g] = dict(
        n_storms=int((df['group_landfall'] == g).sum()),
        p99_mm=round(float(np.nanpercentile(np.abs(fields6[g]), 99)), 1))
fig6['shared_vmax_mm'] = round(vmax6, 1)
out['fig6'] = fig6


# ===========================================================================
# Fig 7  (genesis + track by landfall region)
# ===========================================================================
fig7 = {}
for region in C.REGION_ORDER:
    sub = df[df['landfall_region'] == region]
    glat = sub['genesis_lat'].values; glon = sub['genesis_lon'].values
    tl = sub['pre_landfall_track_km'].values
    sl = _sum(glat); so = _sum(glon); st = _sum(tl)
    fig7[region] = dict(
        n=int(len(sub)),
        genesis_lat_mean=round(sl['mean'], 1), genesis_lat_median=round(float(np.median(glat)), 1),
        genesis_lon_mean=round(so['mean'], 1), genesis_lon_median=round(float(np.median(glon)), 1),
        track_len_mean_km=round(st['mean'], 0), track_len_ci= [round(st['ci_lo'], 0), round(st['ci_hi'], 0)],
        track_len_median_km=round(float(np.median(tl)), 0))
out['fig7'] = fig7


# ===========================================================================
# Fig 8  (genesis-MJO)
# ===========================================================================
def _genesis_all_wnp():
    tr = pd.read_csv(C.TYPHOON_OUT / "all_typhoons_tracks.csv")
    tr['TIME'] = pd.to_datetime(tr['TIME'])
    tr = tr.sort_values(['chinese_code', 'TIME'])
    rows = []
    d2g = mj.date_to_phase_map(months=C.SEASON_MONTHS)
    for code, g in tr.groupby('chinese_code'):
        g = g.sort_values('TIME'); r = g.iloc[0]
        gdate = pd.Timestamp(r['TIME']).normalize()
        rows.append(dict(chinese_code=int(code), genesis_lat=float(r['LAT']),
                         genesis_lon=float(r['LONG']), genesis_date=gdate,
                         wind_category=int(g['wind_category'].iloc[0]) if 'wind_category' in g else 0,
                         group=d2g.get(gdate)))
    d = pd.DataFrame(rows)
    d['formation_region'] = d.apply(
        lambda r: C.formation_region_from_genesis(r['genesis_lat'], r['genesis_lon']), axis=1)
    d = d.dropna(subset=['group'])
    return d

allwnp = _genesis_all_wnp()
weak = df[df['lmi_category'] == 1]
modmaj = df[df['lmi_category'].isin([2, 3])]
landfall_g = df[df['active_genesis']].rename(columns={'group_genesis': 'group'})


def _ab(sub):
    out = {}
    for lr in ('South China', 'East/North China'):
        out[lr] = {}
        for gr in C.GENESIS_ORDER:
            gsub = sub[sub['formation_region'] == gr]
            tot = len(gsub); succ = int((gsub['landfall_region2'] == lr).sum())
            out[lr][gr] = dict(n=succ, tot=tot, pct=round(100 * succ / tot, 1) if tot else 0.0)
    return out


def _cd(d):
    out = {}
    for gr in C.GENESIS_ORDER:
        sub = d[d['formation_region'] == gr]
        counts = {gg: int((sub['group'] == gg).sum()) for gg in C.GROUP_KEY}
        res = mj.residence_permutation_test(counts, ND, rng=np.random.default_rng(RNG_SEED))
        out[gr] = {gg: dict(rate=round(res[gg]['rate'], 2), n=counts[gg],
                            ci=[round(res[gg]['ci_lo'], 2), round(res[gg]['ci_hi'], 2)],
                            stars=res[gg]['stars']) for gg in C.GROUP_KEY}
    return out


fig8 = {'a_weak': _ab(weak), 'b_mod_maj': _ab(modmaj),
        'c_all_wnp': _cd(allwnp), 'd_landfall': _cd(landfall_g)}
out['fig8'] = fig8


# ===========================================================================
# Fig 9  (circulation composites — data ranges + sig fraction)
# ===========================================================================
ds = xr.open_dataset(C.data_path("composite_jjas.nc"))
EXT = [30, 180, 0, 60]
lat = ds['lat'].values; lon = ds['lon'].values
lonsel = (lon >= EXT[0]) & (lon <= EXT[1]); latsel = (lat <= EXT[3]) & (lat >= EXT[2])


def _rang(var, g):
    a = ds[f"{var}_{g}_anom"].values[np.ix_(latsel, lonsel)]
    s = ds[f"{var}_{g}_sig"].values[np.ix_(latsel, lonsel)]
    return dict(min=round(float(np.nanmin(a)), 2), max=round(float(np.nanmax(a)), 2),
                mean=round(float(np.nanmean(a)), 3),
                sig_frac=round(float(np.mean(s >= 1)), 3))


fig9 = {'domain': EXT, 'levels': [-0.8, 0.8, 9], 'cmap': 'RdBu_r',
        'sig_method': 'one-sample t = composite*sqrt(n), |t|>1.645 (a=0.10)',
        'ND_JunSep': ND_jj, 'n_storms_jjaso_active': int(len(df)),
        'panels': {}}
for var, lab in [('vort_850', '850-hPa relative vorticity'),
                 ('hgt_500', '500-hPa geopotential height')]:
    fig9['panels'][lab] = {g: _rang(var, g) for g in ['1-2', '5-6']}
out['fig9'] = fig9


# ===========================================================================
# Supplementary figures (S1-S10)
#   Same recomputation policy as the main figures: point estimates exact &
#   deterministic; bootstrap CIs re-seeded per call. S5-S8 read the same
#   calendar-month composites the figure scripts plot; S3/S10 reuse cached
#   outputs (ibtracs_matched.csv / recompute from event_table).
# ===========================================================================
def _rang_nc(nc_name, var, g):
    """min/max/mean/sig_frac of a composite-anomaly field, domain-restricted."""
    d = xr.open_dataset(C.data_path(f"composite_{nc_name}.nc"))
    a = d[f"{var}_{g}_anom"].values[np.ix_(latsel, lonsel)]
    s = d[f"{var}_{g}_sig"].values[np.ix_(latsel, lonsel)]
    d.close()
    return dict(min=round(float(np.nanmin(a)), 2), max=round(float(np.nanmax(a)), 2),
                mean=round(float(np.nanmean(a)), 3),
                sig_frac=round(float(np.mean(s >= 1)), 3))


def _circ_stats(nc_name, var_label_pairs, groups):
    """Per-variable, per-phase composite field stats used by Fig 9 & S5-S8."""
    d = {}
    for var, lab in var_label_pairs:
        d[lab] = {g: _rang_nc(nc_name, var, g) for g in groups}
    return d


df_all = pd.read_csv(C.data_path("event_table.csv"))            # all 490 landfall storms
df_jjaso = df_all[df_all['in_jjaso']].copy()                    # in_jjaso (464)


# ---- Fig S1 : monthly distribution (all 490 storms, Jan-Dec) ----
months = list(range(1, 13))
figS1 = {'n_total': int(len(df_all)), 'monthly': {}, 'jjaso_share': None}
_jjaso_tot = 0
for m in months:
    by_cat = {C.CATEGORY_LABEL[c]: int(((df_all['month'] == m) &
             (df_all['lmi_category'] == c)).sum()) for c in C.CATEGORY_ORDER}
    tot = sum(by_cat.values())
    figS1['monthly'][m] = dict(by_cat=by_cat, total=tot)
    if 6 <= m <= 10:
        _jjaso_tot += tot
figS1['jjaso_n'] = _jjaso_tot
figS1['jjaso_share'] = round(100 * _jjaso_tot / max(1, len(df_all)), 1)
out['figS1'] = figS1


# ---- Fig S2 : LMI vs intensity-at-landfall (in_jjaso sample, 464) ----
_LF_LAB = {0: 'TD', 1: 'Weak', 2: 'Moderate', 3: 'Super'}
ct = pd.crosstab(df_jjaso['lmi_category'], df_jjaso['landfall_wind_category']).reindex(
    index=[1, 2, 3], columns=[0, 1, 2, 3], fill_value=0).values   # 3 (LMI) x 4 (landfall, incl TD-at-landfall)
# diag / weaken / intensify from the per-storm LMI-vs-landfall comparison directly:
# the 3x4 matrix has no clean trace for the off-main diagonal, and TD-at-landfall
# (landfall=0 < LMI) must count as weakening.
_lmi_v = df_jjaso['lmi_category'].to_numpy()
_lf_v = df_jjaso['landfall_wind_category'].to_numpy()
figS2 = {'n': int(len(df_jjaso)),
         'contingency': {C.CATEGORY_LABEL[r + 1]:
                         {_LF_LAB[c]: int(ct[r, c]) for c in range(4)}
                         for r in range(3)},
         'diag_nochange': int(np.sum(_lf_v == _lmi_v)),
         'weaken':        int(np.sum(_lf_v < _lmi_v)),
         'intensify':     int(np.sum(_lf_v > _lmi_v))}
figS2['regional_by_landfall_intensity'] = {}
figS2['tcp_by_landfall_intensity'] = {}
for cat in C.CATEGORY_ORDER:
    lab = C.CATEGORY_LABEL[cat]
    sub = df_jjaso[df_jjaso['landfall_wind_category'] == cat]
    figS2['regional_by_landfall_intensity'][lab] = {
        r: round(100 * int((sub['landfall_region'] == r).sum()) / max(1, len(sub)), 1)
        for r in C.REGION_ORDER}
    v = sub['tcp_total'].values
    sm = _sum(v) if len(v) else dict(mean=np.nan, median=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0)
    figS2['tcp_by_landfall_intensity'][lab] = dict(
        n=int(len(v)), mean=round(float(sm['mean']), 0), median=round(float(sm['median']), 0),
        ci_lo=round(float(sm['ci_lo']), 0), ci_hi=round(float(sm['ci_hi']), 0))
out['figS2'] = figS2


# ---- Fig S3 : IBTrACS robustness (from cached ibtracs_matched.csv) ----
S3 = {}
_ib_path = Path(C.data_path("ibtracs_matched.csv"))
if _ib_path.exists():
    m = pd.read_csv(_ib_path)
    ctib = pd.crosstab(m['lmi_category'], m['ib_category']).reindex(
        index=[1, 2, 3], columns=[0, 1, 2, 3], fill_value=0).values
    ma = m[m['active_landfall'] == True]
    S3 = {
        'n_in_jjaso': int(len(df_jjaso)), 'n_matched': int(len(m)),
        'match_rate_pct': round(100 * len(m) / max(1, len(df_jjaso)), 1),
        'median_match_km': round(float(np.median(m['match_km'])), 0),
        'wind_corr_r': round(float(np.corrcoef(m['cma_lmi'], m['ib_wind_ms'])[0, 1]), 2),
        'n_active_matched': int(len(ma)),
        'cma_ib_contingency': {
            C.CATEGORY_LABEL[r + 1]: {('TD' if c == 0 else C.CATEGORY_LABEL[c]): int(ctib[r, c])
                                      for c in range(4)} for r in range(3)},
        'major_share_pct': {},
    }
    for col, lab in [('lmi_category', 'CMA'), ('ib_category', 'IBTrACS')]:
        S3['major_share_pct'][lab] = {
            g: round(100 * int((ma[ma['group_landfall'] == g][col] == 3).sum())
               / max(1, int((ma['group_landfall'] == g).sum())), 1) for g in C.GROUP_KEY}
out['figS3'] = S3


# ---- Fig S4 : all-WNP vs China-landfall genesis ----
from lib.wnp import all_wnp_genesis
allwnp = all_wnp_genesis()
lf_s4 = df_all[df_all['in_jjaso'] & df_all['active_genesis']].rename(
    columns={'group_genesis': 'group'})
figS4 = {'a_genesis_rate': {}, 'b_landfall_fraction': {}}
for key, d in [('all_wnp', allwnp), ('landfall', lf_s4)]:
    counts = {g: int((d['group'] == g).sum()) for g in C.GROUP_KEY}
    res = mj.residence_permutation_test(counts, ND, rng=np.random.default_rng(RNG_SEED))
    figS4['a_genesis_rate'][key] = {
        'n': int(sum(counts.values())),
        'groups': {g: dict(rate=round(res[g]['rate'], 2),
                           ci=[round(res[g]['ci_lo'], 2), round(res[g]['ci_hi'], 2)],
                           n=counts[g], stars=res[g]['stars']) for g in C.GROUP_KEY}}
overall_frac = len(lf_s4) / max(1, len(allwnp))
figS4['b_landfall_fraction']['overall_pct'] = round(100 * overall_frac, 1)
figS4['b_landfall_fraction']['groups'] = {}
for g in C.GROUP_KEY:
    n_all = int((allwnp['group'] == g).sum())
    n_lf = int((lf_s4['group'] == g).sum())
    p = n_lf / n_all if n_all else 0.0
    rng = np.random.default_rng(1)                      # matches 12_all_wnp_compare.mj_proportion
    draws = rng.binomial(n_all, p, size=C.BOOT_NITER) / n_all if n_all else np.zeros(C.BOOT_NITER)
    figS4['b_landfall_fraction']['groups'][g] = dict(
        pct=round(100 * p, 1), n=f"{n_lf}/{n_all}",
        ci=[round(100 * float(np.percentile(draws, 5)), 1),
            round(100 * float(np.percentile(draws, 95)), 1)])
out['figS4'] = figS4


# ---- Fig S5 : Jun-Sep 200-hPa zonal wind + MSLP (4 phases) ----
out['figS5'] = {'nc': 'jjas', 'domain': EXT, 'levels': [-0.8, 0.8, 9],
                'panels': _circ_stats('jjas', [('uwnd_200', '200-hPa zonal wind'),
                                               ('slp', 'Mean sea-level pressure')], C.GROUP_KEY)}

# ---- Fig S6/S7/S8 : calendar-month circulation (vort_850 + hgt_500, 4 phases) ----
_month_pairs = {'figS6': ('jun_jul', 'June + July'), 'figS7': ('aug_sep', 'August + September'),
                'figS8': ('oct', 'October')}
_vl = [('vort_850', '850-hPa relative vorticity'),
       ('hgt_500', '500-hPa geopotential height')]
for key, (nc_name, period) in _month_pairs.items():
    out[key] = {'nc': nc_name, 'period': period, 'domain': EXT, 'levels': [-0.8, 0.8, 9],
                'panels': _circ_stats(nc_name, _vl, C.GROUP_KEY)}


# ---- Fig S9 : coastal/inland TCP sensitivity to distance threshold ----
df_s9 = df_all[df_all['in_jjaso'] & df_all['active_landfall']].reset_index(drop=True)
figS9 = {'thresholds': [100, 200, 300], 'n_storms': 0, 'stats': {}}
_s9_codes = [f"{c:04d}" for c in df_s9['chinese_code']]
_s9_cats = df_s9['lmi_category'].values
_s9_per = {thr: {'coastal': {c: [] for c in C.CATEGORY_ORDER},
                 'inland': {c: [] for c in C.CATEGORY_ORDER}} for thr in [100.0, 200.0, 300.0]}
for code, cat in zip(_s9_codes, _s9_cats):
    if cat not in C.CATEGORY_ORDER:
        continue
    try:
        res = tcp_lib.coastal_inland_by_threshold(code, [100.0, 200.0, 300.0])
    except Exception:
        continue
    figS9['n_storms'] += 1
    for thr in [100.0, 200.0, 300.0]:
        coast, inl = res[thr]
        _s9_per[thr]['coastal'][cat].append(coast)
        _s9_per[thr]['inland'][cat].append(inl)
for thr in [100.0, 200.0, 300.0]:
    figS9['stats'][int(thr)] = {}
    for zone in ('coastal', 'inland'):
        figS9['stats'][int(thr)][zone] = {}
        for cat in C.CATEGORY_ORDER:
            arr = np.array(_s9_per[thr][zone][cat], dtype=float)
            s = bs.summarize(arr, rng=np.random.default_rng(RNG_SEED)) if arr.size else \
                dict(mean=0.0, ci_lo=0.0, ci_hi=0.0, n=0)
            figS9['stats'][int(thr)][zone][C.CATEGORY_LABEL[cat]] = dict(
                n=int(arr.size), mean=round(float(s['mean']), 0),
                ci_lo=round(float(s['ci_lo']), 0), ci_hi=round(float(s['ci_hi']), 0))
out['figS9'] = figS9


# ---- Fig S10 : October decision (Jun-Sep vs October event TCP) ----
df_s10 = df_all[df_all['active_landfall']].copy()
jj10 = df_s10[df_s10['month'].between(6, 9)]
oct10 = df_s10[df_s10['month'] == 10]
s_jj10 = bs.summarize(jj10['tcp_total'].values, rng=np.random.default_rng(RNG_SEED))
s_oct10 = bs.summarize(oct10['tcp_total'].values, rng=np.random.default_rng(RNG_SEED))
pd10 = bs.pct_diff_ci(oct10['tcp_total'].values, jj10['tcp_total'].values,
                      rng=np.random.default_rng(RNG_SEED))
out['figS10'] = {
    'jun_sep': dict(n=int(s_jj10['n']), mean=round(float(s_jj10['mean']), 0),
                    ci=[round(float(s_jj10['ci_lo']), 0), round(float(s_jj10['ci_hi']), 0)]),
    'october': dict(n=int(s_oct10['n']), mean=round(float(s_oct10['mean']), 0),
                    ci=[round(float(s_oct10['ci_lo']), 0), round(float(s_oct10['ci_hi']), 0)]),
    'oct_vs_jj_pct': round(float(pd10['pct']), 1),
    'oct_vs_jj_ci': [round(float(pd10['ci_lo']), 1), round(float(pd10['ci_hi']), 1)],
    'robust_oct_increase': bool(pd10['pct'] > 0 and pd10['ci_lo'] > 0),
}


# ===========================================================================
with open(C.data_path("fig_doc_data.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print("saved", C.data_path("fig_doc_data.json"))
print("n landfall (JJASO active):", len(df))
print("ND JJASO:", ND)
print("Fig1 ACE all-WNP ratios:", {g: out['fig1']['a_ace_all_wnp'][g]['ratio'] for g in C.GROUP_KEY})
print("Fig1 ACE landfall ratios:", {g: out['fig1']['a_ace_landfall'][g]['ratio'] for g in C.GROUP_KEY})
print("Fig5 freq ratios:", {g: out['fig5']['a_frequency'][g]['ratio'] for g in C.GROUP_KEY})
print("Fig4 vmax total/mean mm:", fig4['vmax_total_mm'], fig4['vmax_mean_mm'])
print("figS1 Jun-Oct share:", figS1['jjaso_share'], "% (", figS1['jjaso_n'], "/", figS1['n_total'], ")")
print("figS2 contingency diag/weaken/intensify:", figS2['diag_nochange'], figS2['weaken'], figS2['intensify'])
print("figS3 match rate / r:", out['figS3'].get('match_rate_pct'), out['figS3'].get('wind_corr_r'))
print("figS10 Oct-vs-JunSep pct:", out['figS10']['oct_vs_jj_pct'])
