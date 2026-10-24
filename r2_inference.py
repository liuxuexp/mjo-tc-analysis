"""Single-pass year-block inference for all reported statistics.

Computes the per-group tests and confidence intervals used by Figures 1, 3, 5,
7c/d and 8 and by Tables S1/S2:

  * Significance: 10,000 year-block permutations of the 65 complete JJASO MJO
    sequences (TC records stay in their original years), two-sided empirical p,
    then Benjamini-Hochberg q within each family.
  * Confidence intervals: 10,000-replicate year-block bootstrap (65 seasons
    resampled with replacement); 95% CI = 2.5/97.5 percentiles.
  * The Figure 3 / Table S1 family also stores the 2.5-97.5 percentile interval
    of the permutation null, which is what the Figure 3 error bars show.

Test families (q computed within each):
  fig3       48 = 4 TC samples x 3 regions x 4 phase groups   -> Table S1
  fig1a       8 = 2 ACE samples x 4 groups
  fig1b      12 = 3 LMI categories x 4 groups
  fig5a       4 = all-TC landfall frequency
  fig7c/d    16 + 16 = 4 genesis regions x 4 groups, two samples
  fig8a/b     4 + 4 = WNP genesis rates, two samples
  fig8c       4 = conditional China-landfall proportions
  tcp_total/mean/median  4 each (Fig 1c/d, Fig 5b/c/d)
  prespec_contrast  19 pre-specified phases 5-6 vs 1-2 contrasts (Table S2)

Outputs:
  data03/r2_stats_pergrp.csv    one row per (family, analysis, stratum, group)
  data03/r2_stats_contrast.csv  pre-specified 5-6 vs 1-2 contrasts
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import config as C
from lib import stats_yb as S

B = S.B_DEFAULT
PCT = (2.5, 97.5)

t0 = time.time()
blk = S.load_blocks()
G, D, Dt = blk['G'], blk['D'], blk['D_total']
PERMS = S.make_perms(B, n_years=G.shape[0], seed=S.SEED_PERM)
MULTI = S.make_multisets(B, n_years=G.shape[0], seed=S.SEED_BOOT)
GK = C.GROUP_KEY
GI = {g: i for i, g in enumerate(GK)}          # '1-2' -> 0 ...
GN = {g: GK.index(g) for g in GK}
DTf = Dt.astype(float)

rows = []           # per-group rows (assembled into families, q applied later)
contrast_specs = []  # (analysis, sample, stratum, metric, keep-core dict)


def _pct(a, axis=0):
    return np.nanpercentile(a, PCT, axis=axis)


# ---------------------------------------------------------------------------
# Core machinery for one count-based event set
# ---------------------------------------------------------------------------
def count_core(dates):
    """Year-block permutation null + bootstrap for a set of dated events.

    Returns dict with observed counts/rates/ratios, the permutation null of
    counts/rates/ratios [B,4] and the bootstrap distributions [B,4].
    """
    yi, di = S.day_index(dates)
    obs = G[yi, di]
    lab = S.labels_under(G, PERMS, yi, di)
    null_cnt = S.counts_from_labels(lab)
    null_rate = null_cnt / DTf[None, :] * 1000.0
    null_ratio = S.rate_ratio_from_labels(lab, Dt)
    obs_cnt = S.counts_from_labels(obs[None, :])[0]
    obs_rate = obs_cnt / DTf * 1000.0
    obs_ratio = S.rate_ratio_from_labels(obs[None, :], Dt)[0]
    boot_cnt = S.boot_group_counts(obs, yi, MULTI)
    boot_D = np.maximum(S.boot_D(MULTI, D).astype(float), 1e-12)
    boot_rate = boot_cnt / boot_D * 1000.0
    boot_base = np.maximum(boot_cnt.sum(axis=1) / boot_D.sum(axis=1), 1e-12)
    boot_ratio = (boot_cnt / boot_D) / boot_base[:, None]
    return dict(obs_cnt=obs_cnt, obs_rate=obs_rate, obs_ratio=obs_ratio,
                null_cnt=null_cnt, null_rate=null_rate, null_ratio=null_ratio,
                boot_cnt=boot_cnt, boot_rate=boot_rate, boot_ratio=boot_ratio)


def emit_count(family, analysis, metric, sample, stratum, core, phase_at,
               est_kind, unit):
    """Push 4 per-group rows for one count analysis.

    est_kind: 'ratio'  -> headline estimate = rate ratio (Fig 3, Fig 1b, ...)
              'rate'   -> headline estimate = events per 1000 phase-days (Fig 8)
    """
    if est_kind == 'ratio':
        est, (lo, hi) = core['obs_ratio'], _pct(core['boot_ratio'])
    else:
        est, (lo, hi) = core['obs_rate'], _pct(core['boot_rate'])
    nlo, nhi = _pct(core['null_ratio'])
    for i, g in enumerate(GK):
        rows.append(dict(
            family=family, analysis=analysis, metric=metric, sample=sample,
            stratum=stratum, phase_group=g, phase_at=phase_at,
            n=int(core['obs_cnt'][i]), D=int(Dt[i]),
            est=float(est[i]), est_lo=float(lo[i]), est_hi=float(hi[i]),
            est_unit=unit, ratio=float(core['obs_ratio'][i]),
            null_lo=float(nlo[i]), null_hi=float(nhi[i]),
            p_perm=S.emp_p_two_sided(core['null_ratio'][:, i],
                                     float(core['obs_ratio'][i]))))


def add_contrast(analysis, sample, stratum, metric, core=None, prop=None):
    """Register a pre-specified phases 5-6 vs 1-2 contrast (count metric via
    `core` from count_core; proportion metric via `prop` dict from prop_core).
    """
    if prop is not None:
        contrast_specs.append(dict(analysis=analysis, sample=sample,
                                   stratum=stratum, metric=metric,
                                   prop=prop))
    else:
        contrast_specs.append(dict(analysis=analysis, sample=sample,
                                   stratum=stratum, metric=metric,
                                   core=core))


# ---------------------------------------------------------------------------
# Load event samples
# ---------------------------------------------------------------------------
ev = pd.read_csv(C.data_path("event_table.csv"))
ev = ev[ev['in_jjaso'] & ev['active_landfall']].reset_index(drop=True)
ev_gen = pd.read_csv(C.data_path("event_table.csv"))
ev_gen = ev_gen[ev_gen['in_jjaso'] & ev_gen['active_genesis']].reset_index(drop=True)

CAT_FULL = {0: 'All TCs', **{c: C.CATEGORY_FULL[c] for c in C.CATEGORY_ORDER}}
SAMPLES = [(0, ev)] + [(c, ev[ev['lmi_category'] == c]) for c in C.CATEGORY_ORDER]
EXPECTED_N = {0: 258, 1: 84, 2: 112, 3: 62}
for c, sub in SAMPLES:
    assert len(sub) == EXPECTED_N[c], f"sample {c}: n={len(sub)} != {EXPECTED_N[c]}"

print(f"[{time.time()-t0:6.1f}s] samples: "
      f"landfall-based All/Weak/Mod/Super = {[len(s) for _, s in SAMPLES]}, "
      f"genesis-based landfall subset n={len(ev_gen)}")

# ---------------------------------------------------------------------------
# Family fig3 — 48 regional tests (Table S1); revised Fig 3 error bars
# ---------------------------------------------------------------------------
for c, sub_base in SAMPLES:
    for region in C.REGION_ORDER:
        sub = sub_base[sub_base['landfall_region'] == region]
        core = count_core(pd.to_datetime(sub['landfall_time']))
        emit_count('fig3', 'fig3', 'landfall_rate_ratio', CAT_FULL[c], region,
                   core, 'landfall', 'ratio', 'ratio to no-modulation')
        add_contrast('fig3', CAT_FULL[c], region,
                     'landfall rate (per 1000 phase-days)', core=core)
if True:
    s_all = int(((ev['landfall_region'] == 'South China') &
                 (ev['group_landfall'] == '1-2')).sum())
    assert s_all == 31, f"South China 1-2 n={s_all} != 31"
print(f"[{time.time()-t0:6.1f}s] fig3: 48 rows")

# ---------------------------------------------------------------------------
# Families fig1b (LMI landfall counts) and fig5a (all-TC frequency)
# ---------------------------------------------------------------------------
for c in C.CATEGORY_ORDER:
    sub = ev[ev['lmi_category'] == c]
    core = count_core(pd.to_datetime(sub['landfall_time']))
    emit_count('fig1b', 'fig1b', 'landfall_rate_ratio', CAT_FULL[c], 'China',
               core, 'landfall', 'ratio', 'ratio to no-modulation')
    add_contrast('fig1b', CAT_FULL[c], 'China',
                 'landfall rate (per 1000 phase-days)', core=core)
core = count_core(pd.to_datetime(ev['landfall_time']))
emit_count('fig5a', 'fig5a', 'landfall_rate_ratio', 'All TCs', 'China',
           core, 'landfall', 'ratio', 'ratio to no-modulation')
add_contrast('fig5a', 'All TCs', 'China',
             'landfall rate (per 1000 phase-days)', core=core)

# ---------------------------------------------------------------------------
# Genesis samples: all WNP (685, Jun-Oct genesis) + China-landfall subset
# (254, genesis may fall in late May while landfall is in Jun-Oct; the May
# preamble of the label grid carries those events)
# ---------------------------------------------------------------------------
from lib.wnp import all_wnp_genesis

aw = all_wnp_genesis()                       # n = 685, phase@genesis
assert len(aw) == 685, f"all-WNP genesis n={len(aw)} != 685"
assert len(ev_gen) == 254, f"landfall-subset genesis n={len(ev_gen)} != 254"

core = count_core(pd.to_datetime(aw['genesis_date']))
emit_count('fig8a', 'fig8a', 'genesis_rate', 'All WNP TCs', 'WNP',
           core, 'genesis', 'rate', 'events per 1000 phase-days')
add_contrast('fig8a', 'All WNP TCs', 'WNP',
             'genesis rate (per 1000 phase-days)', core=core)
core = count_core(pd.to_datetime(ev_gen['genesis_time']))
# observed labels must reproduce event_table's group_genesis column
_gn = {g: i for i, g in enumerate(GK)}
assert np.array_equal(
    G[S.day_index(pd.to_datetime(ev_gen['genesis_time']))],
    ev_gen['group_genesis'].map(_gn).values), "genesis labels mismatch"
emit_count('fig8b', 'fig8b', 'genesis_rate', 'China-landfall subset', 'WNP',
           core, 'genesis', 'rate', 'events per 1000 phase-days')
add_contrast('fig8b', 'China-landfall subset', 'WNP',
             'genesis rate (per 1000 phase-days)', core=core)
print(f"[{time.time()-t0:6.1f}s] genesis rates done (685 / 254)")

# ---------------------------------------------------------------------------
# Family fig7c/d — genesis rates by genesis region
# ---------------------------------------------------------------------------
for fam, dat, col, smp in [('fig7c', aw, 'genesis_date', 'All WNP TCs'),
                           ('fig7d', ev_gen, 'genesis_time',
                            'China-landfall subset')]:
    for gr in C.GENESIS_ORDER:
        sub = dat[dat['formation_region'] == gr]
        core = count_core(pd.to_datetime(sub[col]))
        emit_count(fam, fam, 'genesis_rate', smp, gr, core, 'genesis',
                   'rate', 'events per 1000 phase-days')

# ---------------------------------------------------------------------------
# Family fig8c — conditional China-landfall proportion by genesis phase.
# Numerator sample = the 254-storm landfall subset; denominator sample = the
# 685-storm all-WNP set (overall 254/685 = 37.1%).
# Both event sets are relabeled by the SAME permuted sequences / multisets.
# ---------------------------------------------------------------------------
yi_a, di_a = S.day_index(pd.to_datetime(aw['genesis_date']))
obs_all = G[yi_a, di_a]
yi_l, di_l = S.day_index(pd.to_datetime(ev_gen['genesis_time']))
obs_land = G[yi_l, di_l]
lab_a = S.labels_under(G, PERMS, yi_a, di_a)
lab_l = S.labels_under(G, PERMS, yi_l, di_l)
null_n_all = S.counts_from_labels(lab_a).astype(float)
null_n_land = S.counts_from_labels(lab_l).astype(float)
null_prop = np.divide(null_n_land, null_n_all,
                      out=np.zeros_like(null_n_land), where=null_n_all > 0)
obs_n_all = S.counts_from_labels(obs_all[None, :])[0].astype(float)
obs_n_land = S.counts_from_labels(obs_land[None, :])[0].astype(float)
assert obs_n_land.sum() == 254 and obs_n_all.sum() == 685
obs_prop = obs_n_land / np.maximum(obs_n_all, 1.0)
overall_prop = 254.0 / 685.0
ya_cnt = np.zeros((G.shape[0], 4))
yl_cnt = np.zeros((G.shape[0], 4))
for g in range(4):
    ya_cnt[:, g] = np.bincount(yi_a[obs_all == g], minlength=G.shape[0])
    yl_cnt[:, g] = np.bincount(yi_l[obs_land == g], minlength=G.shape[0])
boot_n_all = MULTI @ ya_cnt
boot_n_land = MULTI @ yl_cnt
boot_prop = np.divide(boot_n_land, boot_n_all,
                      out=np.full_like(boot_n_land, np.nan),
                      where=boot_n_all > 0)
plo, phi = _pct(boot_prop)
for i, g in enumerate(GK):
    rows.append(dict(
        family='fig8c', analysis='fig8c',
        metric='conditional_landfall_proportion', sample='WNP->China',
        stratum='WNP', phase_group=g, phase_at='genesis',
        n=int(obs_n_land[i]), D=int(obs_n_all[i]),
        est=float(obs_prop[i] * 100.0), est_lo=float(plo[i] * 100.0),
        est_hi=float(phi[i] * 100.0), est_unit='%',
        ratio=float(obs_prop[i] / overall_prop),
        null_lo=np.nan, null_hi=np.nan,
        p_perm=S.emp_p_two_sided(null_prop[:, i], float(obs_prop[i]))))
add_contrast('fig8c', 'WNP->China', 'WNP', 'conditional landfall proportion (%)',
             prop=dict(obs=obs_prop, boot=boot_prop, null=null_prop))
print(f"[{time.time()-t0:6.1f}s] fig7c/d + fig8c done "
      f"(overall proportion {overall_prop*100:.1f}%)")

# ---------------------------------------------------------------------------
# Family fig1a — ACE ratios (track-point weighted, E-matrix permutation)
# ---------------------------------------------------------------------------
def ace_daily_matrix(tracks):
    """E[y, d] = ACE of 6-hourly track points on that calendar day."""
    tr = tracks.copy()
    tr['TIME'] = pd.to_datetime(tr['TIME'])
    tr = tr[(tr['TIME'].dt.year.between(C.YEAR_START, C.YEAR_END)) &
            (tr['TIME'].dt.month.between(6, 10))]
    tr = tr.dropna(subset=['WND'])
    tr = tr[tr['WND'] > 0]
    yi, di = S.day_index(tr['TIME'].dt.normalize())
    ace = (tr['WND'] * C.MS_TO_KT) ** 2 / C.ACE_UNIT
    E = np.zeros((G.shape[0], S.NDAY_TOTAL))
    np.add.at(E, (yi, di), ace.values)
    return E


def ace_analysis(tracks, sample, family_key='fig1a'):
    E = ace_daily_matrix(tracks)
    n_y = G.shape[0]
    # storm counts per group (for Fig 1a bar annotations)
    tr = tracks.copy()
    tr['TIME'] = pd.to_datetime(tr['TIME'])
    tr = tr[(tr['TIME'].dt.year.between(C.YEAR_START, C.YEAR_END)) &
            (tr['TIME'].dt.month.between(6, 10))]
    tr = tr.dropna(subset=['WND'])
    tr = tr[tr['WND'] > 0]
    yi_tr, di_tr = S.day_index(tr['TIME'].dt.normalize())
    tr['g'] = G[yi_tr, di_tr]
    nstorm = tr[tr['g'] >= 0].groupby('g')['chinese_code'].nunique()
    nstorm = np.array([int(nstorm.get(g, 0)) for g in range(4)])
    W = np.stack([E @ (G == g).T for g in range(4)], axis=2)   # [y, s, g]
    V = np.stack([np.einsum('yd,yd->y', E, G == g) for g in range(4)],
                 axis=1)                                        # [y, g]
    obs_ace = V.sum(axis=0)
    # observed ratio directly (labels = own sequence by construction)
    obs_ratio = (obs_ace / DTf) / max(obs_ace.sum() / Dt.sum(), 1e-12)
    # permutation null: gather W[y, perm(y), g] and sum over y
    yidx = np.arange(n_y)[None, :]
    null_ace = np.stack([W[yidx, PERMS, g].sum(axis=1) for g in range(4)],
                        axis=1)                                  # [B, 4]
    base = np.maximum(null_ace.sum(axis=1) / Dt.sum(), 1e-12)
    null_ratio = (null_ace / DTf[None, :]) / base[:, None]
    # year-block bootstrap
    boot_ace = MULTI @ V
    boot_D = np.maximum(S.boot_D(MULTI, D).astype(float), 1e-12)
    boot_base = np.maximum(boot_ace.sum(axis=1) / boot_D.sum(axis=1), 1e-12)
    boot_ratio = (boot_ace / boot_D) / boot_base[:, None]
    lo, hi = _pct(boot_ratio)
    nlo, nhi = _pct(null_ratio)
    for i, g in enumerate(GK):
        rows.append(dict(
            family=family_key, analysis='fig1a', metric='ace_ratio',
            sample=sample, stratum='WNP', phase_group=g, phase_at='track point',
            n=int(nstorm[i]), D=int(Dt[i]),
            est=float(obs_ratio[i]), est_lo=float(lo[i]), est_hi=float(hi[i]),
            est_unit='ratio to no-modulation', ratio=float(obs_ratio[i]),
            null_lo=float(nlo[i]), null_hi=float(nhi[i]),
            p_perm=S.emp_p_two_sided(null_ratio[:, i], float(obs_ratio[i]))))
    print(f"    ACE ratio {sample}: "
          f"{ {g: round(obs_ratio[i], 2) for i, g in enumerate(GK)} }")


ace_analysis(pd.read_csv(C.TYPHOON_OUT / "all_typhoons_tracks.csv"),
             'All WNP TCs')
ace_analysis(pd.read_csv(C.TYPHOON_OUT / "landfall_typhoons_tracks.csv"),
             'China-landfall subset')
print(f"[{time.time()-t0:6.1f}s] fig1a ACE done")

# ---------------------------------------------------------------------------
# TCP families — total / mean / median event TCP per phase group
# ---------------------------------------------------------------------------
tcp = ev['tcp_total'].values.astype(float)
dates_lf = pd.to_datetime(ev['landfall_time'])
yi_t, di_t = S.day_index(dates_lf)
g_obs = ev['group_landfall'].map(GN).values.astype(np.int8)
lab_t = S.labels_under(G, PERMS, yi_t, di_t)

# --- tcp_total (Fig 1c / Fig 5b): estimate = group total; test = weighted ratio
boot_tot = S.boot_weighted_sums(tcp, g_obs, yi_t, MULTI)
null_tot = S.weighted_sums_from_labels(lab_t, tcp)
obs_tot = S.weighted_sums_from_labels(g_obs[None, :], tcp)[0]
obs_wratio = S.weighted_ratio_from_labels(g_obs[None, :], tcp, Dt)[0]
null_wratio = S.weighted_ratio_from_labels(lab_t, tcp, Dt)
boot_wratio_base = np.maximum(boot_tot.sum(axis=1) / Dt.sum(), 1e-12)
boot_wratio = (boot_tot / DTf[None, :]) / boot_wratio_base[:, None]
tlo, thi = _pct(boot_tot)
wnlo, wnhi = _pct(null_wratio)
for i, g in enumerate(GK):
    rows.append(dict(
        family='tcp_total', analysis='fig1c', metric='total_tcp',
        sample='All TCs', stratum='China', phase_group=g, phase_at='landfall',
        n=int((g_obs == i).sum()), D=int(Dt[i]),
        est=float(obs_tot[i]), est_lo=float(tlo[i]), est_hi=float(thi[i]),
        est_unit='10^6 m^3', ratio=float(obs_wratio[i]),
        null_lo=float(wnlo[i]), null_hi=float(wnhi[i]),
        p_perm=S.emp_p_two_sided(null_wratio[:, i], float(obs_wratio[i]))))

# --- tcp_mean (Fig 1d / Fig 5c) and tcp_median (Fig 5d)
vs = S.boot_value_stats(tcp, g_obs, yi_t, MULTI)
null_mean = np.divide(null_tot, null_cnt_t := S.counts_from_labels(lab_t),
                      out=np.full_like(null_tot, np.nan), where=null_cnt_t > 0)
null_median = S.perm_group_median(lab_t, tcp)
obs_mean = np.array([tcp[g_obs == i].mean() if (g_obs == i).any() else np.nan
                     for i in range(4)])
obs_median = np.array([np.median(tcp[g_obs == i]) if (g_obs == i).any()
                       else np.nan for i in range(4)])
mlo, mhi = _pct(vs['mean'])
medlo, medhi = _pct(vs['median'])
for i, g in enumerate(GK):
    rows.append(dict(
        family='tcp_mean', analysis='fig1d', metric='mean_event_tcp',
        sample='All TCs', stratum='China', phase_group=g, phase_at='landfall',
        n=int((g_obs == i).sum()), D=int(Dt[i]),
        est=float(obs_mean[i]), est_lo=float(mlo[i]), est_hi=float(mhi[i]),
        est_unit='10^6 m^3', ratio=np.nan,
        null_lo=_pct(null_mean)[0][i], null_hi=_pct(null_mean)[1][i],
        p_perm=S.emp_p_two_sided(null_mean[:, i], float(obs_mean[i]))))
    rows.append(dict(
        family='tcp_median', analysis='fig5d', metric='median_event_tcp',
        sample='All TCs', stratum='China', phase_group=g, phase_at='landfall',
        n=int((g_obs == i).sum()), D=int(Dt[i]),
        est=float(obs_median[i]), est_lo=float(medlo[i]), est_hi=float(medhi[i]),
        est_unit='10^6 m^3', ratio=np.nan,
        null_lo=_pct(null_median)[0][i], null_hi=_pct(null_median)[1][i],
        p_perm=S.emp_p_two_sided(null_median[:, i], float(obs_median[i]))))
print(f"[{time.time()-t0:6.1f}s] TCP total/mean/median families done")

# ---------------------------------------------------------------------------
# Assemble per-group table, apply BH within each family
# ---------------------------------------------------------------------------
pergrp = pd.DataFrame(rows)
pergrp['q_fdr'] = np.nan
for fam, sub in pergrp.groupby('family', sort=False):
    q = S.bh_fdr(sub['p_perm'].values)
    pergrp.loc[sub.index, 'q_fdr'] = q
pergrp.to_csv(C.data_path("r2_stats_pergrp.csv"), index=False)
print(f"\nper-group rows: {len(pergrp)}  "
      f"(families: {pergrp.groupby('family').size().to_dict()})")

# ---------------------------------------------------------------------------
# Pre-specified contrasts: phases 5-6 vs 1-2 (statistic = absolute difference;
# two-sided permutation p; bootstrap CIs for both difference and rate ratio)
# ---------------------------------------------------------------------------
crows = []
for spec in contrast_specs:
    if 'core' in spec and spec['core'] is not None:
        core = spec['core']
        obs12, obs56 = core['obs_rate'][0], core['obs_rate'][2]
        null_diff = core['null_rate'][:, 2] - core['null_rate'][:, 0]
        boot_diff = core['boot_rate'][:, 2] - core['boot_rate'][:, 0]
        boot_rr = np.divide(core['boot_rate'][:, 2], core['boot_rate'][:, 0],
                            out=np.full(B, np.nan),
                            where=core['boot_rate'][:, 0] > 0)
        scale = 1.0
    else:
        p = spec['prop']
        obs12, obs56 = p['obs'][0] * 100, p['obs'][2] * 100
        null_diff = (p['null'][:, 2] - p['null'][:, 0]) * 100
        boot_diff = (p['boot'][:, 2] - p['boot'][:, 0]) * 100
        boot_rr = np.divide(p['boot'][:, 2], p['boot'][:, 0],
                            out=np.full(B, np.nan), where=p['boot'][:, 0] > 0)
        scale = 100.0
    obs_diff = obs56 - obs12
    obs_rr = obs56 / obs12 if obs12 > 0 else np.nan
    dlo, dhi = np.nanpercentile(boot_diff, PCT)
    with np.errstate(invalid='ignore'):
        rr_fin = boot_rr[np.isfinite(boot_rr)]
    if rr_fin.size:
        rlo, rhi = np.nanpercentile(rr_fin, PCT)
    else:
        rlo = rhi = np.nan
    crows.append(dict(
        analysis=spec['analysis'], sample=spec['sample'],
        stratum=spec['stratum'], metric=spec['metric'],
        phases='5-6 vs 1-2',
        value_12=float(obs12), value_56=float(obs56),
        diff=float(obs_diff), diff_lo=float(dlo), diff_hi=float(dhi),
        rate_ratio=float(obs_rr), rate_ratio_lo=float(rlo),
        rate_ratio_hi=float(rhi),
        p_perm=S.emp_p_two_sided(null_diff, float(obs_diff))))
contr = pd.DataFrame(crows)
contr['q_fdr'] = S.bh_fdr(contr['p_perm'].values)
contr.to_csv(C.data_path("r2_stats_contrast.csv"), index=False)
print(f"contrast rows: {len(contr)} (family prespec_contrast)")

# ---------------------------------------------------------------------------
# Console summary of what survives FDR
# ---------------------------------------------------------------------------
print("\n=== q < 0.05 per family ===")
for fam, sub in pergrp.groupby('family'):
    sig = sub[sub['q_fdr'] < 0.05]
    print(f"  {fam:<11s} {len(sig):2d}/{len(sub):2d} significant"
          + (f"  {list(sig['phase_group'])}" if len(sig) else ""))
sig_c = contr[contr['q_fdr'] < 0.05]
print(f"  prespec_contrast {len(sig_c)}/{len(contr)} significant")
print(f"\nDONE in {time.time()-t0:.1f}s")
