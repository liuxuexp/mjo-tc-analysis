"""Year-block bootstrap and permutation inference.

  * Confidence intervals: nonparametric year-block bootstrap over the 65
    complete June-October seasons (1960-2024), sampled with replacement;
    10,000 replicates, 95% CI from the 2.5th and 97.5th percentiles.
  * Significance: complete annual JJASO MJO sequences are reassigned among
    years while TC and precipitation records stay in their original years.
    Two-sided empirical p = min(1, 2*min[(r_up+1)/(B+1), (r_lo+1)/(B+1)]).
  * Multiple comparisons: Benjamini-Hochberg FDR within pre-specified families,
    q < 0.05.

Every JJASO season spans the same 153 calendar dates, so sequence permutation
leaves the per-group phase-day totals unchanged; only event labels move.
Events on days inactive in the assigned sequence carry label -1.

lib/mjo keeps the day-level method for the sensitivity check only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

NGROUP = len(C.GROUP_KEY)          # 4 phase groups
PRE_DAYS = 31                      # May 1-31 label-only preamble
NDAY = 153                         # Jun 1 .. Oct 31 inclusive (D_g window)
NDAY_TOTAL = PRE_DAYS + NDAY       # May 1 .. Oct 31 event-labeling grid
B_DEFAULT = 10_000
SEED_BOOT = 20261018
SEED_PERM = 20261019


# ---------------------------------------------------------------------------
# Season blocks
# ---------------------------------------------------------------------------
def load_blocks(year_start=C.YEAR_START, year_end=C.YEAR_END):
    """Return dict with the 65 annual MJO sequences on the May 1-Oct 31 grid.

    G        int8 [n_years, 184]: phase-group index 0..3, or -1 for inactive
             days, day 0 = May 1. May days are LABEL-ONLY preamble: they can
             carry event labels (a few storms form in late May and land in
             June) but never count toward the phase-day totals D_g.
    years    int  [n_years]
    D        int  [n_years, 4]: per-year Jun-Oct phase-day counts
    D_total  int  [4]: season-total phase-day counts per group
    """
    years = np.arange(year_start, year_end + 1)
    mjo = pd.read_csv(C.MJO_CSV)
    mjo['date'] = pd.to_datetime(mjo['date'])
    mjo = mjo[(mjo['date'] >= pd.Timestamp(f'{year_start}-01-01')) &
              (mjo['date'] <= pd.Timestamp(f'{year_end}-12-31'))]
    mjo = mjo[mjo['amplitude'] >= 1.0].copy()
    mjo['month'] = mjo['date'].dt.month
    mjo = mjo[mjo['month'].between(5, 10)]
    gmap = {}
    for g, (p1, p2) in C.GROUP_PHASES.items():
        i = C.GROUP_KEY.index(g)
        gmap[p1] = i
        gmap[p2] = i
    G = np.full((len(years), NDAY_TOTAL), -1, dtype=np.int8)
    yidx = mjo['date'].dt.year.values - year_start
    may1_by_year = np.array([np.datetime64(f'{y}-05-01', 'D') for y in years])
    didx = (mjo['date'].values.astype('datetime64[D]') -
            may1_by_year[yidx]).astype(int)
    keep = (didx >= 0) & (didx < NDAY_TOTAL)
    yidx, didx = yidx[keep], didx[keep]
    ph = mjo['phase'].astype(int).values[keep]
    G[yidx, didx] = np.array([gmap.get(p, -1) for p in ph], dtype=np.int8)
    Gjja = G[:, PRE_DAYS:]                      # Jun-Oct window only
    D = np.stack([(Gjja == g).sum(axis=1) for g in range(NGROUP)], axis=1)
    return dict(G=G, years=years, D=D, D_total=D.sum(axis=0))


def day_index(dates):
    """Calendar dates -> (year_idx, day_idx) into the May 1-Oct 31 grid."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize()
    yi = (dates.dt.year - C.YEAR_START).values
    may1 = pd.to_datetime({'year': dates.dt.year.values, 'month': 5, 'day': 1})
    di = (dates.values.astype('datetime64[D]') -
          may1.values.astype('datetime64[D]')).astype(int)
    ok = (yi >= 0) & (yi <= C.YEAR_END - C.YEAR_START) & \
         (di >= 0) & (di < NDAY_TOTAL)
    if not ok.all():
        bad = dates[~ok]
        raise ValueError(f"{len(bad)} event dates outside May 1-Oct 31, "
                         f"e.g. {list(bad.head())}")
    return yi, di


# ---------------------------------------------------------------------------
# Resampling plans
# ---------------------------------------------------------------------------
def make_perms(B=B_DEFAULT, n_years=None, seed=SEED_PERM):
    """B random permutations of the n_years annual sequences -> int32 [B, n]."""
    n = n_years if n_years is not None else C.YEAR_END - C.YEAR_START + 1
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(n) for _ in range(B)])
    return perms.astype(np.int32)


def make_multisets(B=B_DEFAULT, n_years=None, seed=SEED_BOOT):
    """B year-block bootstrap draws -> multiplicity matrix int32 [B, n_years]."""
    n = n_years if n_years is not None else C.YEAR_END - C.YEAR_START + 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(B, n))
    multi = np.zeros((B, n), dtype=np.int32)
    rows = np.repeat(np.arange(B), n)
    np.add.at(multi, (rows, draws.ravel()), 1)
    return multi


def labels_under(G, perms, y_idx, d_idx):
    """Group labels of events under each permutation -> int8 [B, n_events].

    Row b: label of each event when year y_i's sequence is G[perms[b, y_i]].
    -1 = the day is MJO-inactive in the assigned sequence.
    """
    return G[perms[:, y_idx], d_idx]


# ---------------------------------------------------------------------------
# Statistics recomputed from labels (permutation null)
# ---------------------------------------------------------------------------
def rate_ratio_from_labels(labels, D_total, per=1000):
    """Phase-normalized rate RATIOS from label rows -> float [B, 4].

    ratio_g = (n_g / D_g) / (n_eff / sum(D)); n_eff counts labels >= 0.
    """
    lab = labels  # [B, n]
    B = lab.shape[0]
    counts = np.stack([ (lab == g).sum(axis=1) for g in range(NGROUP)], axis=1)
    n_eff = (lab >= 0).sum(axis=1)
    Dg = np.maximum(D_total.astype(float), 1e-12)
    rates = counts / Dg[None, :] * per
    base = n_eff / max(D_total.sum(), 1) * per
    base = np.maximum(base, 1e-12)
    return rates / base[:, None]


def weighted_sums_from_labels(labels, w):
    """Per-group weighted sums (TCP, ACE) from label rows -> float [B, 4]."""
    w = np.asarray(w, dtype=float)
    out = np.empty((labels.shape[0], NGROUP))
    for g in range(NGROUP):
        out[:, g] = (labels == g) @ w
    return out


def weighted_ratio_from_labels(labels, w, D_total):
    """Phase-normalized weighted RATIO (e.g. total-TCP ratio) from label rows.

    ratio_g = (W_g / D_g) / (W_eff / sum(D)); W_eff sums labels >= 0.
    """
    sums = weighted_sums_from_labels(labels, w)
    eff = sums.sum(axis=1)
    Dg = np.maximum(np.asarray(D_total, dtype=float), 1e-12)
    rates = sums / Dg[None, :]
    base = np.maximum(eff / max(D_total.sum(), 1), 1e-12)
    return rates / base[:, None]


def perm_group_median(labels, values):
    """Per-group MEDIAN of `values` under each label row -> float [B, 4].

    Exact: for each replicate the labeled subset is fully determined, so the
    median is its middle element (average of two middles for even k). Uses the
    sort-once / cumulative-count trick over the [B, n] label matrix.
    """
    val = np.asarray(values, dtype=float)
    order = np.argsort(val, kind='stable')
    v_sorted = val[order]
    out = np.full((labels.shape[0], NGROUP), np.nan)
    for g in range(NGROUP):
        mem = (labels[:, order] == g)                  # [B, n] membership
        cnt = mem.sum(axis=1)
        cum = np.cumsum(mem, axis=1)
        k1 = (cnt + 1) // 2                            # lower middle rank
        k2 = (cnt + 2) // 2                            # upper middle rank
        has = cnt > 0
        lo = np.full(labels.shape[0], v_sorted[0])
        hi = lo.copy()
        lo[has] = v_sorted[np.argmax(cum[has] >= k1[has, None], axis=1)]
        hi[has] = v_sorted[np.argmax(cum[has] >= k2[has, None], axis=1)]
        med = 0.5 * (lo + hi)
        out[:, g] = np.where(has, med, np.nan)
    return out


def counts_from_labels(labels):
    """Per-group event counts (labels >= 0 only) -> int [B, 4]."""
    return np.stack([(labels == g).sum(axis=1) for g in range(NGROUP)], axis=1)


# ---------------------------------------------------------------------------
# Statistics recomputed from year multisets (bootstrap CIs)
# ---------------------------------------------------------------------------
def boot_group_counts(g_obs, y_idx, multisets):
    """Per-group event counts in each bootstrap multiset -> int [B, 4]."""
    n_years = multisets.shape[1]
    cnt_y = np.zeros((n_years, NGROUP), dtype=np.int64)
    for g in range(NGROUP):
        cnt_y[:, g] = np.bincount(y_idx[g_obs == g], minlength=n_years)
    return multisets @ cnt_y


def boot_weighted_sums(w, g_obs, y_idx, multisets):
    """Per-group weighted sums (e.g. total TCP, ACE) in each multiset."""
    n_years = multisets.shape[1]
    s_y = np.zeros((n_years, NGROUP))
    for g in range(NGROUP):
        s_y[:, g] = np.bincount(y_idx[g_obs == g], weights=w[g_obs == g],
                                minlength=n_years)
    return multisets @ s_y


def boot_value_stats(values, g_obs, y_idx, multisets):
    """Per-group sum / mean / median of a per-event value (e.g. event TCP).

    Median uses the year-multiset expansion (values repeated by year
    multiplicity), which is exact for these sample sizes.
    Returns dict of arrays [B, 4]: n, sum, mean, median.
    """
    n_years = multisets.shape[1]
    B = multisets.shape[0]
    val = np.asarray(values, dtype=float)
    sum_y = np.zeros((n_years, NGROUP))
    cnt_y = np.zeros((n_years, NGROUP), dtype=np.int64)
    keep = ~np.isnan(val)
    val, g_o, y_o = val[keep], g_obs[keep], y_idx[keep]
    for g in range(NGROUP):
        m = g_o == g
        if m.sum() == 0:
            continue
        cnt_y[:, g] = np.bincount(y_o[m], minlength=n_years)
        sum_y[:, g] = np.bincount(y_o[m], weights=val[m], minlength=n_years)
    sums = multisets @ sum_y
    cnts = multisets @ cnt_y
    means = np.divide(sums, cnts, out=np.full_like(sums, np.nan),
                      where=cnts > 0)
    # median of the expanded multiset: years sorted by per-year median is NOT
    # valid in general -> expand exactly (samples are small)
    medians = np.full((B, NGROUP), np.nan)
    for g in range(NGROUP):
        m = g_o == g
        if m.sum() == 0:
            continue
        vg, yg = val[m], y_o[m]
        # pre-sort once; per replicate rebuild the multiset via np.repeat
        for b in range(B):
            rep = multisets[b][yg]
            if rep.sum() == 0:
                continue
            vexp = np.repeat(vg, rep)
            medians[b, g] = np.median(vexp)
    return dict(n=cnts, sum=sums, mean=means, median=medians)


def boot_D(multisets, D):
    """Phase-day totals per group in each bootstrap multiset -> [B, 4]."""
    return multisets @ D


def boot_scalar_stats(values, dates, B=B_DEFAULT, seed=SEED_BOOT):
    """Year-block bootstrap of an UNGROUPED scalar sample (no phase groups).

    The resampling unit is the calendar year of each observation (its date's
    year). Returns dict(n, mean, median, mean_lo, mean_hi, median_lo,
    median_hi) with 95% CIs = 2.5/97.5 percentiles. NaN values (and their
    dates) are dropped first.
    """
    v = np.asarray(values, dtype=float)
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    keep = ~np.isnan(v)
    v, d = v[keep], d[keep].reset_index(drop=True)
    yi, _ = day_index(d)
    n_y = C.YEAR_END - C.YEAR_START + 1
    multi = make_multisets(B, n_years=n_y, seed=seed)
    cnt_y = np.bincount(yi, minlength=n_y)
    sum_y = np.bincount(yi, weights=v, minlength=n_y)
    cnt = multi @ cnt_y
    s = multi @ sum_y
    means = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    order = np.argsort(v, kind='stable')
    v_s = v[order]
    yi_s = yi[order]
    medians = np.empty(B)
    for b in range(B):
        rep = multi[b][yi_s]
        k = rep.sum()
        if k == 0:
            medians[b] = np.nan
            continue
        cum = np.cumsum(rep)
        k1 = (k + 1) // 2
        k2 = (k + 2) // 2
        lo = v_s[np.searchsorted(cum, k1, side='left')]
        hi = v_s[np.searchsorted(cum, k2, side='left')]
        medians[b] = 0.5 * (lo + hi)
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                mean_lo=float(np.nanpercentile(means, 2.5)),
                mean_hi=float(np.nanpercentile(means, 97.5)),
                median_lo=float(np.nanpercentile(medians, 2.5)),
                median_hi=float(np.nanpercentile(medians, 97.5)))


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def emp_p_two_sided(null, obs):
    """Two-sided empirical p with finite-simulation correction.

    p = min(1, 2*min[(r_up+1)/(B+1), (r_lo+1)/(B+1)]).
    """
    null = np.asarray(null, dtype=float)
    B = null.size
    r_up = int((null >= obs).sum())
    r_lo = int((null <= obs).sum())
    return float(min(1.0, 2 * min((r_up + 1) / (B + 1), (r_lo + 1) / (B + 1))))


def bh_fdr(pvals):
    """Benjamini-Hochberg step-up FDR adjusted q values (Benjamini & Hochberg
    1995). NaN p values pass through as NaN (excluded from the family).
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    m = np.isfinite(p)
    pv = p[m]
    n = pv.size
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.minimum(ranked, 1.0)
    qm = np.empty(n)
    qm[order] = ranked
    q[m] = qm
    return q


def fmt_p(x, nd=3):
    """Format p/q values: '<0.001' convention, else fixed decimals."""
    if not np.isfinite(x):
        return ''
    return '<0.001' if x < 10 ** (-nd) else f'{x:.{nd}f}'
