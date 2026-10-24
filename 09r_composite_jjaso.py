"""Build JJASO circulation composites over all MJO-active days.

Each phase group is composited from every active day in June-October after
removing the calendar-month climatology. Grid-point significance uses a
year-block permutation test followed by BH FDR within each panel (q < 0.05).

Fields: vort_850, uwnd_850, vwnd_850, slp on the 30-180E, 0-60N grid.
Output: data/data03/composite_jjaso.nc
"""
from __future__ import annotations
import sys, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import xarray as xr
import pandas as pd

import config as C
from lib import dynamics as dyn
from lib import stats_yb as S

B = S.B_DEFAULT
LAT_RANGE = (0.0, 60.0)          # analysis domain (matches the Fig 9 extent)
LON_RANGE = (30.0, 180.0)

# fields -> (kind): 'sigma' fields contribute one standardized tensor,
# 'wind' fields contribute a standardized tensor (for the q test) AND an
# unstandardized m/s anomaly tensor (for the vectors).
SIGMA_FIELDS = ['vort_850', 'slp']
WIND_FIELDS = ['uwnd_850', 'vwnd_850']


# ---------------------------------------------------------------------------
# Daily series helpers
# ---------------------------------------------------------------------------
def load_series(var, level, months=C.SEASON_MONTHS):
    """Jun-Oct daily NCEP series on the analysis domain, lat ascending."""
    da = dyn.load_ncar_level(var, level, range(C.YEAR_START, C.YEAR_END + 1),
                             months=months)
    da = da.sortby('time').sortby('lat')
    da = da.sel(lat=slice(*LAT_RANGE), lon=slice(*LON_RANGE))
    return da


def check_calendar(t):
    """Assert t is exactly Jun 1 .. Oct 31 for each of the 65 years."""
    n_y = C.YEAR_END - C.YEAR_START + 1
    assert len(t) == n_y * S.NDAY, (len(t), n_y * S.NDAY)
    expect = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f'{y}-06-01', f'{y}-10-31', freq='D') for y in
        range(C.YEAR_START, C.YEAR_END + 1)]))
    assert (t == expect).all(), 'NCEP calendar != Jun-Oct grid'
    return n_y


def z_and_anom(da):
    """Return (z, anom): calendar-month standardized series and the raw
    calendar-month anomaly (physical units)."""
    cmean, cstd, mmap = dyn.monthly_climatology(da)
    arr = da.values.astype(np.float64)
    month_of = pd.DatetimeIndex(da.time.values).month.values
    clim_row = np.array([mmap[m] for m in month_of])
    anom = arr - cmean[clim_row]
    z = (arr - cmean[clim_row]) / cstd[clim_row]
    z = np.where(np.isfinite(z), z, 0.0)
    anom = np.where(np.isfinite(anom), anom, 0.0)
    return z, anom


# ---------------------------------------------------------------------------
# Year-block permutation of grid-point composites
# ---------------------------------------------------------------------------
def build_C(z2d, Gjja, n_y):
    """C[y,s,g,p] = sum over Jun-Oct days d of z2d[y,d,p] * 1[Gjja[s,d]==g].

    z2d: float32 [n_y, NDAY, P]. Returns float32 [n_y, n_y, 4, P].
    """
    n, nday, P = z2d.shape
    assert (n, nday) == (n_y, S.NDAY)
    C_t = np.empty((n, n, S.NGROUP, P), dtype=np.float32)
    Zd = np.ascontiguousarray(z2d.transpose(1, 0, 2).reshape(nday, n * P))
    MA = (Gjja[:, None, :] == np.arange(S.NGROUP)[None, :, None]).astype(np.float32)
    for s in range(n):
        out = Zd.T @ MA[s].T                    # [n*P, 4] BLAS gemm (contract days)
        C_t[:, s] = out.reshape(n, P, S.NGROUP).transpose(0, 2, 1)
    return C_t


def perm_field_means(C_t, perms, D_total):
    """Observed + permutation-null group-mean composites.

    Returns (obs [4,P], null [B,4,P]). Row b of the null assigns year y the
    sequence of year perms[b,y]; group means divide by the (permutation-
    invariant) phase-day totals D_g.
    """
    n = C_t.shape[0]
    B = perms.shape[0]
    P = C_t.shape[-1]
    yy = np.arange(n)
    obs = np.empty((S.NGROUP, P), dtype=np.float64)
    null = np.empty((B, S.NGROUP, P), dtype=np.float32)
    batch = max(1, int(2.5e7 // (n * P)))       # keep gather batches <= ~100 MB
    for g in range(S.NGROUP):
        Cg = C_t[:, :, g, :]
        obs[g] = Cg[yy, yy].sum(axis=0) / D_total[g]
        for i0 in range(0, B, batch):
            pb = perms[i0:i0 + batch]
            gat = Cg[yy[None, :], pb]           # [nb, n, P]
            null[i0:i0 + batch, g] = gat.sum(axis=1) / D_total[g]
    return obs, null


def grid_pvals(obs, null):
    """Two-sided empirical p at each grid point -> [4,P]."""
    B = null.shape[0]
    p = np.empty(obs.shape, dtype=np.float64)
    for g in range(S.NGROUP):
        r_up = (null[:, g, :] >= obs[g][None, :]).sum(axis=0)
        r_lo = (null[:, g, :] <= obs[g][None, :]).sum(axis=0)
        p[g] = np.minimum(1.0, 2 * np.minimum((r_up + 1) / (B + 1),
                                              (r_lo + 1) / (B + 1)))
    return p


def run_field(z2d, Gjja, D_total, perms, n_y, check=False):
    """Full pipeline for one tensor: C build -> null -> p -> BH q per panel.

    check=True additionally verifies the tensor-diagonal observed composite
    against the direct mean over the group's active days (group 0).
    """
    z2d = z2d.astype(np.float32)
    C_t = build_C(z2d, Gjja, n_y)
    obs, null = perm_field_means(C_t, perms, D_total)
    if check:
        direct = (z2d * (Gjja == 0)[:, :, None]).sum(axis=(0, 1)) / D_total[0]
        assert np.allclose(obs[0], direct, atol=2e-3), \
            np.abs(obs[0] - direct).max()
        print("    tensor-diagonal check vs direct composite: OK")
    del C_t; gc.collect()
    p = grid_pvals(obs, null)
    del null; gc.collect()
    q = np.empty_like(p)
    for g in range(S.NGROUP):
        q[g] = S.bh_fdr(p[g])                   # FDR across the field's grid points
    return obs, q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    blocks = S.load_blocks()
    Gjja = blocks['G'][:, S.PRE_DAYS:]
    D_total = blocks['D_total']
    n_y = len(blocks['years'])
    perms = S.make_perms(B, n_years=n_y)
    print(f"years={n_y}, D_total={D_total.tolist()}, B={B}")

    lat = lon = None
    store = {}

    def register(name, obs, q, units):
        store[name] = (obs.astype(np.float32), q.astype(np.float32))
        for gi, g in enumerate(C.GROUP_KEY):
            nq = int((q[gi] < 0.05).sum())
            print(f"    {name} {g}: max|anom|={np.abs(obs[gi]).max():.2f} "
                  f"{units}, q<0.05 at {nq}/{q.shape[1]} points "
                  f"({100 * nq / q.shape[1]:.1f}%)")

    # -- sigma fields ------------------------------------------------------
    for base, var, level in [('slp', 'slp', None)]:
        print(f"processing {base} ({var}@{level}) ...")
        da = load_series(var, level)
        check_calendar(pd.DatetimeIndex(da.time.values))
        lat, lon = da['lat'].values, da['lon'].values
        z, _ = z_and_anom(da)
        del da; gc.collect()
        obs, q = run_field(z.reshape(n_y, S.NDAY, -1), Gjja, D_total, perms,
                           n_y, check=True)
        register(base, obs, q, 'sigma')

    print("processing vort_850 (curl u,v@850) ...")
    u = load_series('uwnd', 850); v = load_series('vwnd', 850)
    u, v = xr.align(u, v, join='inner')
    tidx = pd.DatetimeIndex(u.time.values)
    check_calendar(tidx)
    lat, lon = u['lat'].values, u['lon'].values
    ut, vt = u.values, v.values
    zv = np.empty(ut.shape, dtype=np.float64)
    for t in range(ut.shape[0]):
        zv[t] = dyn.relative_vorticity(ut[t], vt[t], lat, lon)
    del u, v, ut, vt; gc.collect()
    z_vort, _ = z_and_anom(xr.DataArray(
        zv, coords=[('time', tidx), ('lat', lat), ('lon', lon)], name='vort'))
    del zv; gc.collect()
    obs, q = run_field(z_vort.reshape(n_y, S.NDAY, -1), Gjja, D_total, perms, n_y)
    register('vort_850', obs, q, 'sigma')

    # -- wind fields: z (for q) + m/s anomaly (for vectors) -----------------
    for base, var in [('uwnd_850', 'uwnd'), ('vwnd_850', 'vwnd')]:
        print(f"processing {base} ({var}@850) ...")
        da = load_series(var, 850)
        check_calendar(pd.DatetimeIndex(da.time.values))
        z, anom = z_and_anom(da)
        del da; gc.collect()
        _, q = run_field(z.reshape(n_y, S.NDAY, -1), Gjja, D_total, perms, n_y)
        del z; gc.collect()
        obs_a, _ = run_field(anom.reshape(n_y, S.NDAY, -1), Gjja, D_total, perms, n_y)
        register(base + '_anom', obs_a, q, 'm/s')   # vectors + component q

    # -- assemble netCDF ----------------------------------------------------
    ds = xr.Dataset(coords={
        'group': ('group', C.GROUP_KEY),
        'lat': ('lat', lat), 'lon': ('lon', lon)})
    # nc variable prefix -> (store key, units); wind q values come from the
    # standardized-series test stored with the anomaly entry.
    fields = {
        'slp': ('slp', 'sigma (calendar-month standardized)'),
        'vort_850': ('vort_850', 'sigma (calendar-month standardized)'),
        'uwnd_850': ('uwnd_850_anom', 'm s-1'),
        'vwnd_850': ('vwnd_850_anom', 'm s-1'),
    }
    for name, (key, unit) in fields.items():
        obs, q = store[key]
        shp = (S.NGROUP, len(lat), len(lon))
        ds[name + '_anom'] = (('group', 'lat', 'lon'), obs.reshape(shp))
        ds[name + '_anom'].attrs['units'] = unit
        ds[name + '_q'] = (('group', 'lat', 'lon'), q.reshape(shp))
        ds[name + '_q'].attrs['units'] = '1'
        desc = ('JJASO composite mean over all MJO-active days of the phase '
                'group (n_days = D_g, see n_days). q: BH-FDR-adjusted '
                'two-sided year-block permutation q per grid point, B=10000'
                + ('; wind components tested on the standardized series'
                   if name.endswith('wnd_850') else '')
                + '.')
        ds[name + '_anom'].attrs['description'] = desc
    ds['n_days'] = (('group',), D_total.astype(np.int32))
    ds.attrs.update({
        'description': ('JJASO all-MJO-active-day composites, 1960-2024; '
                        'per-grid-point year-block permutation (B=10000, '
                        'seed=20261019) + BH FDR per field x phase-group panel'),
        'domain': f'lat {LAT_RANGE[0]}-{LAT_RANGE[1]}N, lon {LON_RANGE[0]}-'
                  f'{LON_RANGE[1]}E',
        'season': 'Jun-Oct',
    })
    out = C.data_path('composite_jjaso.nc')
    ds.to_netcdf(out)
    print(f"saved {out} ({len(ds.data_vars)} vars)")

    # remove monthly composites left by an earlier run
    for stale in ('composite_jjas.nc', 'composite_jun_jul.nc',
                  'composite_aug_sep.nc', 'composite_oct.nc'):
        p = Path(C.data_path(stale))
        if p.exists():
            p.unlink()
            print(f"removed stale {stale}")


if __name__ == "__main__":
    main()
