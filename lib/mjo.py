"""MJO loading, phase-day (residence-time) accounting, and significance.

Centralises the residence-time-corrected machinery: counts of active MJO days per
phase group, phase-day normalization, and a permutation null that respects phase
residence time (replacing the old hardcoded asterisks / flat 25% line).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_mjo(year_start=C.YEAR_START, year_end=C.YEAR_END, amp_min=1.0,
             months=None) -> pd.DataFrame:
    """Active-MJO day table with integer phase and group key.

    Returns columns: date(datetime,00:00), phase(int 1-8), amplitude, group('1-2'..).
    months=None -> full year; pass SEASON_MONTHS for Jun-Oct.
    """
    mjo = pd.read_csv(C.MJO_CSV)
    mjo['date'] = pd.to_datetime(mjo['date']).dt.normalize()
    mjo = mjo[(mjo['date'] >= pd.Timestamp(f'{year_start}-01-01')) &
              (mjo['date'] <= pd.Timestamp(f'{year_end}-12-31'))]
    mjo = mjo[mjo['amplitude'] >= amp_min].copy()
    if months is not None:
        mjo = mjo[mjo['date'].dt.month.isin(months)].copy()
    mjo['phase'] = mjo['phase'].astype(int)
    mjo['group'] = mjo['phase'].apply(C.phase_to_group)
    mjo = mjo.dropna(subset=['group']).reset_index(drop=True)
    return mjo[['date', 'phase', 'amplitude', 'group']]


def date_to_phase_map(year_start=C.YEAR_START, year_end=C.YEAR_END,
                      months=None, amp_min=1.0):
    """date(Timestamp or date) -> group key '1-2'.. for active days (else absent)."""
    mjo = load_mjo(year_start, year_end, amp_min, months)
    return dict(zip(mjo['date'], mjo['group']))


def date_to_phase_int(year_start=C.YEAR_START, year_end=C.YEAR_END,
                      months=None, amp_min=1.0):
    mjo = load_mjo(year_start, year_end, amp_min, months)
    return dict(zip(mjo['date'], mjo['phase']))


# ---------------------------------------------------------------------------
# Residence-time accounting
# ---------------------------------------------------------------------------
def active_days_per_group(year_start=C.YEAR_START, year_end=C.YEAR_END,
                          months=None, amp_min=1.0) -> dict:
    """NDAYS[group] = count of active-MJO days with phase in group, in season."""
    mjo = load_mjo(year_start, year_end, amp_min, months)
    s = mjo['group'].value_counts()
    return {g: int(s.get(g, 0)) for g in C.GROUP_KEY}


def phase_day_fractions(ndays: dict) -> dict:
    """p_g = NDAYS[g] / sum(NDAYS) -- the residence-time multinomial weights."""
    tot = float(sum(ndays.values()))
    if tot <= 0:
        return {g: 0.0 for g in C.GROUP_KEY}
    return {g: ndays[g] / tot for g in C.GROUP_KEY}


def no_modulation_rate(counts: dict, ndays: dict, per: int = 1000) -> dict:
    """Uniform residence-time expectation of the phase-day-normalized rate.

    rate_g = count_g / NDAYS_g ;  under no modulation every phase has the SAME rate
    = total_events / total_active_days.  Scaled by `per` (e.g. per 1000 phase-days)
    for plotting.  Returned per group (same value for all groups).
    """
    tot_days = float(sum(ndays.values()))
    tot_events = float(sum(counts.values()))
    base = (tot_events / tot_days) * per if tot_days > 0 else 0.0
    return {g: base for g in C.GROUP_KEY}


def normalize_counts(counts: dict, ndays: dict, per: int = 1000) -> dict:
    """Phase-day-normalized rate: events per `per` active-phase-days."""
    return {g: (counts[g] / ndays[g] * per) if ndays[g] > 0 else 0.0
            for g in C.GROUP_KEY}


# ---------------------------------------------------------------------------
# Significance: residence-time permutation null (replaces hardcoded stars)
# ---------------------------------------------------------------------------
def residence_permutation_test(observed_counts: dict, ndays: dict,
                               n_iter=C.BOOT_NITER, alpha=C.ALPHA,
                               per: int = 1000, rng=None):
    """Two-sided test of phase-day-normalized rate against a residence-time null.

    Null: each of N=sum(counts) events is independently assigned to a phase group
    drawn from the active-day multinomial p_g = NDAYS_g / sum(NDAYS). For each
    resample we compute the normalized rate per group; the (1-alpha) CI is the
    [alpha/2, 1-alpha/2] percentile band. Significant if observed rate falls
    outside the band (two-sided).

    Returns dict per group: {n, ndays, rate, ci_lo, ci_hi, sig(bool), stars}.
    """
    if rng is None:
        rng = np.random.default_rng(20240601)
    p = np.array([phase_day_fractions(ndays)[g] for g in C.GROUP_KEY], dtype=float)
    p = p / p.sum()
    ntot = int(sum(observed_counts.values()))
    nd = np.array([ndays[g] for g in C.GROUP_KEY], dtype=float)
    # resampled rates
    sim = np.empty((n_iter, len(C.GROUP_KEY)), dtype=float)
    for b in range(n_iter):
        c = np.bincount(rng.choice(len(C.GROUP_KEY), size=ntot, p=p),
                        minlength=len(C.GROUP_KEY)).astype(float)
        sim[b] = np.where(nd > 0, c / nd * per, 0.0)
    lo = np.percentile(sim, 100 * alpha / 2, axis=0)
    hi = np.percentile(sim, 100 * (1 - alpha / 2), axis=0)
    out = {}
    for i, g in enumerate(C.GROUP_KEY):
        rate = observed_counts[g] / nd[i] * per if nd[i] > 0 else 0.0
        sig = bool(rate < lo[i] or rate > hi[i])
        out[g] = dict(n=int(observed_counts[g]), ndays=int(nd[i]), rate=float(rate),
                      ci_lo=float(lo[i]), ci_hi=float(hi[i]), sig=sig,
                      stars='*' if sig else '')
    return out


def stars_for(observed_counts: dict, ndays: dict, **kw) -> dict:
    """Convenience: just the significance stars per group."""
    r = residence_permutation_test(observed_counts, ndays, **kw)
    return {g: r[g]['stars'] for g in C.GROUP_KEY}


def residence_ratio_test(observed_counts: dict, ndays: dict,
                        n_iter=C.BOOT_NITER, alpha=C.ALPHA, per: int = 1000, rng=None):
    """Express the phase-day-normalized rate as a RATIO to the no-modulation rate
    (1.0 = expectation under residence-time-only). Plots as 'multiples of no
    modulation' (e.g. 0.60 / 0.55 / 1.92 / 0.92).

    Returns dict per group {n, ndays, ratio, ci_lo, ci_hi, sig, stars, rate, null}.
    """
    if rng is None:
        rng = np.random.default_rng(20240601)
    base = residence_permutation_test(observed_counts, ndays, n_iter, alpha, per, rng)
    null_rate = no_modulation_rate(observed_counts, ndays, per=per)[C.GROUP_KEY[0]]
    out = {}
    for g in C.GROUP_KEY:
        b = base[g]
        r = b['rate'] / null_rate if null_rate > 0 else 0.0
        out[g] = dict(n=b['n'], ndays=b['ndays'], rate=b['rate'], null=null_rate,
                      ratio=float(r),
                      ci_lo=float(b['ci_lo'] / null_rate) if null_rate > 0 else 0.0,
                      ci_hi=float(b['ci_hi'] / null_rate) if null_rate > 0 else 0.0,
                      sig=b['sig'], stars=b['stars'])
    return out
