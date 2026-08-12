"""Storm-level bootstrap confidence intervals and summary statistics.

Two flavours:
  * storm-level bootstrap of a scalar per-storm metric (mean / median event TCP,
    track length, ...), reporting the (1-alpha) percentile CI and sample size.
  * a proportion CI against an overall-rate null for regional/genesis shares
    (alpha=0.10).

All routines use a fixed RNG so results are reproducible across runs.
"""
from __future__ import annotations
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

RNG = np.random.default_rng(20240601)


def summarize(values, n_iter=C.BOOT_NITER, alpha=C.ALPHA, rng=RNG):
    """Storm-level summary of a scalar array.

    Returns dict(n, mean, median, std, ci_lo, ci_hi) with the (1-alpha) bootstrap
    CI on the *mean* (resampled storm sets with replacement). Median is reported
    alongside (no CI unless requested). NaNs dropped.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, sum=0.0)
    mean = float(v.mean())
    median = float(np.median(v))
    # bootstrap on the mean
    idx = rng.integers(0, n, size=(n_iter, n))
    boot_means = v[idx].mean(axis=1)
    return dict(n=int(n), mean=mean, median=median, std=float(v.std(ddof=1)) if n > 1 else 0.0,
                ci_lo=float(np.percentile(boot_means, 100 * alpha / 2)),
                ci_hi=float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
                sum=float(v.sum()))


def summarize_median(values, n_iter=C.BOOT_NITER, alpha=C.ALPHA, rng=RNG):
    """Bootstrap CI on the *median* (for Fig 5d median event-TCP panel)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return dict(n=0, median=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    idx = rng.integers(0, n, size=(n_iter, n))
    boot_med = np.median(v[idx], axis=1)
    return dict(n=int(n), median=float(np.median(v)),
                ci_lo=float(np.percentile(boot_med, 100 * alpha / 2)),
                ci_hi=float(np.percentile(boot_med, 100 * (1 - alpha / 2))))


def proportion_ci(group_success, group_total, overall_prop,
                 n_iter=C.BOOT_NITER, alpha=C.ALPHA, rng=RNG):
    """Binomial CI under an overall-rate null.

    Returns (observed_prop, significant, ci_lo, ci_hi). Significant if observed
    falls outside the [alpha/2, 1-alpha/2] percentile band of Binom(total, overall).
    """
    if group_total <= 0 or overall_prop <= 0 or overall_prop >= 1:
        return 0.0, False, 0.0, 0.0
    obs = group_success / group_total
    draws = rng.binomial(group_total, overall_prop, size=n_iter) / group_total
    lo = np.percentile(draws, 100 * alpha / 2)
    hi = np.percentile(draws, 100 * (1 - alpha / 2))
    return float(obs), bool(obs < lo or obs > hi), float(lo), float(hi)


def pct_diff_ci(values_a, values_b, n_iter=C.BOOT_NITER, rng=RNG):
    """Bootstrap (a-b)/b*100 for two storm samples (percent difference)."""
    a = np.asarray(values_a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(values_b, dtype=float); b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return dict(pct=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    ma, mb = a.mean(), b.mean()
    ra = a[rng.integers(0, a.size, (n_iter, a.size))].mean(axis=1)
    rb = b[rng.integers(0, b.size, (n_iter, b.size))].mean(axis=1)
    pct = (ra - rb) / rb * 100
    return dict(pct=float((ma - mb) / mb * 100),
                ci_lo=float(np.percentile(pct, 5)),
                ci_hi=float(np.percentile(pct, 95)))
