"""Unit checks for lib/stats_yb.py (run: python tests_stats_yb.py).

Covers:
  1. D_g invariance under sequence permutation
  2. finite-simulation two-sided empirical p
  3. Benjamini-Hochberg q values on a worked example
  4. year-block bootstrap count and value statistics on hand-checkable cases
plus cross-checks on the event table.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import config as C
from lib import stats_yb as S
from lib import mjo as mj

ok = 0
fail = 0


def check(name, cond, detail=''):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


print("== 1. D_g invariance under permutation ==")
blk = S.load_blocks()
G, D, Dt = blk['G'], blk['D'], blk['D_total']
perms = S.make_perms(B=200, n_years=G.shape[0], seed=1)
# every Jun-Oct (year, day) as a pseudo-event -> labels under each permutation
y_all = np.repeat(np.arange(G.shape[0]), S.NDAY)
d_all = np.tile(np.arange(S.PRE_DAYS, S.NDAY_TOTAL), G.shape[0])
lab = S.labels_under(G, perms, y_all, d_all)
cnt = np.stack([(lab == g).sum(axis=1) for g in range(4)], axis=1)
check("phase-day totals invariant (all 200 perms)",
      np.array_equal(cnt, Dt[None, :].repeat(200, axis=0)),
      f"max dev={np.abs(cnt - Dt).max()}")
# cross-check D_total against the phase-day totals in the event table
ND = mj.active_days_per_group(months=C.SEASON_MONTHS)
Dt_ref = np.array([ND[g] for g in C.GROUP_KEY])
check(f"D_total matches lib/mjo accounting {Dt.tolist()}",
      np.array_equal(Dt, Dt_ref), f"yb={Dt.tolist()} ref={Dt_ref.tolist()}")
check("G shape 65x184 (May 1-Oct 31)", G.shape == (65, S.NDAY_TOTAL), str(G.shape))
check("May preamble all inactive-vs-active but excluded from D",
      not np.array_equal(D.sum(axis=1),
                         np.stack([(G == g).sum(1) for g in range(4)], 1).sum(1))
      or (G[:, :S.PRE_DAYS] == -1).all())

print("== 2. two-sided empirical p ==")
rng = np.random.default_rng(7)
null = rng.normal(0, 1, S.B_DEFAULT)
p_extreme = S.emp_p_two_sided(null, 10.0)
check("extreme obs -> p = 2/(B+1)", abs(p_extreme - 2 / (S.B_DEFAULT + 1)) < 1e-12,
      f"{p_extreme}")
p_mid = S.emp_p_two_sided(null, 0.0)
check("median obs -> p ~ 1 (>= 0.9)", p_mid >= 0.9, f"{p_mid}")
p_small = S.emp_p_two_sided(np.array([1.0, 2.0, 3.0]), 0.5)
check("tiny B: obs below all null -> p = 2*min(1/4, 4/4) = 0.5",
      p_small == 0.5, f"{p_small}")

print("== 3. Benjamini-Hochberg ==")
p = np.array([0.01, 0.04, 0.03, 0.005])
q = S.bh_fdr(p)
check("BH worked example", np.allclose(q, [0.02, 0.04, 0.04, 0.02]),
      f"q={q.tolist()}")
p15 = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205,
                0.212, 0.216, 0.222, 0.251, 0.269, 0.275, 0.34])
q15 = S.bh_fdr(p15)
# classic worked example (BH 1995-style): expected step-up q in sorted order
q15_expected = np.array([0.015, 0.06, 0.126, 0.126, 0.126, 0.15, 0.1586,
                         0.2946, 0.2946, 0.2946, 0.2946, 0.2946, 0.2946,
                         0.2946, 0.34])
check("BH classic 15-p example",
      np.allclose(q15, q15_expected, atol=1e-4), f"q={q15.tolist()}")
check("BH monotone in p", np.all(np.diff(q15[np.argsort(p15)]) >= -1e-12))
qn = S.bh_fdr(np.append(p15, np.nan))
check("NaN passthrough", np.isnan(qn[-1]) and np.allclose(qn[:-1], q15))

print("== 4. bootstrap engines on synthetic data ==")
# 3 years, 4 groups; events: year0->(g0,g0), year1->(g1), year2->(g3)
Gsyn = np.array([[0, 0, 1, -1], [1, 1, 2, 3], [2, -1, -1, 3]], dtype=np.int8)
ys = np.array([0, 0, 1, 2])
ds = np.array([0, 1, 0, 3])
g_obs = Gsyn[ys, ds]                      # [0, 0, 1, 3]
check("synthetic labels", g_obs.tolist() == [0, 0, 1, 3], g_obs.tolist())
multi = np.array([[2, 1, 1], [1, 0, 3]], dtype=np.int32)  # 2 replicates
cnt = S.boot_group_counts(g_obs, ys, multi)
check("boot counts rep0", cnt[0].tolist() == [4, 1, 0, 1], cnt[0].tolist())
check("boot counts rep1", cnt[1].tolist() == [2, 0, 0, 3], cnt[1].tolist())
vals = np.array([1.0, 3.0, 5.0, 2.0])
st = S.boot_value_stats(vals, g_obs, ys, multi)
check("boot sums rep0", st['sum'][0].tolist() == [8.0, 5.0, 0.0, 2.0],
      st['sum'][0].tolist())
check("boot means rep0 (g2 empty -> 0 by cnt>0 guard then nan-safe)",
      np.allclose(st['mean'][0][[0, 1, 3]], [2.0, 5.0, 2.0]),
      st['mean'][0].tolist())
check("boot median rep1 g3 = 2.0 (3x value 2)",
      st['median'][1][3] == 2.0, f"{st['median'][1]}")
ws = S.boot_weighted_sums(vals, g_obs, ys, multi)
check("boot weighted sums == sums", np.allclose(ws, st['sum']))

print("== 4b. weighted sums / ratio / perm median ==")
lab2 = np.array([[0, 0, 1, 3], [1, 1, 1, 1]], dtype=np.int8)
ws2 = np.array([1.0, 3.0, 5.0, 2.0])
ws_sum = S.weighted_sums_from_labels(lab2, ws2)
check("weighted sums from labels",
      ws_sum[0].tolist() == [4.0, 5.0, 0.0, 2.0] and ws_sum[1].tolist() == [0.0, 11.0, 0.0, 0.0],
      ws_sum.tolist())
Dt2 = np.array([2, 1, 1, 2])
wr = S.weighted_ratio_from_labels(lab2, ws2, Dt2)
# rep0: rates [4/2, 5/1, 0, 2/2] = [2,5,0,1]; eff=11, base=11/6 -> ratios
exp0 = np.array([2, 5, 0, 1]) / (11 / 6)
check("weighted ratio rep0", np.allclose(wr[0], exp0), wr[0].tolist())
med2 = S.perm_group_median(lab2, ws2)
check("perm median rep0 (g0: {1,3}->2.0, g1: {5}, g3: {2})",
      med2[0][0] == 2.0 and med2[0][1] == 5.0 and np.isnan(med2[0][2])
      and med2[0][3] == 2.0, med2[0].tolist())
check("perm median rep1 g1 even count {1,3,5,2} -> 2.5",
      med2[1][1] == 2.5 and np.isnan(med2[1][0]), med2[1].tolist())
# brute-force cross-check on random labels
rng2 = np.random.default_rng(11)
lr = rng2.integers(0, 4, size=(7, 12)).astype(np.int8)
vr = rng2.normal(size=12)
mr = S.perm_group_median(lr, vr)
brute = np.full((7, 4), np.nan)
for b in range(7):
    for g in range(4):
        sel = lr[b] == g
        if sel.sum():
            brute[b, g] = np.median(vr[sel])
check("perm median == brute force", np.allclose(mr, brute, equal_nan=True))

print("== 5. real-data smoke: event labeling matches event_table ==")
df = pd.read_csv(C.data_path("event_table.csv"))
df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)
yi, di = S.day_index(pd.to_datetime(df['landfall_time']))
check("landfall dates inside JJASO window", True)
obs_lab = G[yi, di]
gnum = df['group_landfall'].map({g: i for i, g in enumerate(C.GROUP_KEY)}).values
check("labels reproduce group_landfall column",
      np.array_equal(obs_lab, gnum),
      f"mismatches={int((obs_lab != gnum).sum())}")
perms1 = S.make_perms(B=50, seed=3)
lab1 = S.labels_under(G, perms1, yi, di)
rr = S.rate_ratio_from_labels(lab1, Dt)
check("permuted ratio rows finite & positive",
      np.isfinite(rr).all() and (rr >= 0).all())
identity = S.rate_ratio_from_labels(obs_lab[None, :], Dt)
# observed ratio for group with all events concentrated
check("identity labeling gives finite ratios", np.isfinite(identity).all(),
      identity.round(2).tolist())

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
