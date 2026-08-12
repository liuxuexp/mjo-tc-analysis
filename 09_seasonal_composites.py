"""09 — Calendar-month circulation composites.

For each field, standardise the full Jun-Oct daily series by its per-calendar-month
climatology (z = (X - Xbar_month) / sigma_month, 1960-2024), then composite
different day-sets from the same standardised series:
  * Jun-Sep   -> data03/composite_jjas.nc   (Fig 9; October excluded)
  * Jun-Jul   -> data03/composite_jun_jul.nc (Fig S6)
  * Aug-Sep   -> data03/composite_aug_sep.nc (Fig S7)
  * Oct       -> data03/composite_oct.nc     (Fig S8)

Day selection: active-MJO days (amplitude>=1) that intersect the set of
landfall-TC track days. Significance: one-sample t = composite*sqrt(n), |t|>1.645.

Fields: vort_850 (curl of u,v@850), uwnd_850, vwnd_850, hgt_500, uwnd_500,
vwnd_500, uwnd_200, slp.
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
from lib import mjo as mj

FIELDS = [           # (base_name, ncar_var, level)
    ('uwnd_850', 'uwnd', 850), ('vwnd_850', 'vwnd', 850),
    ('hgt_500', 'hgt', 500), ('uwnd_500', 'uwnd', 500), ('vwnd_500', 'vwnd', 500),
    ('uwnd_200', 'uwnd', 200), ('slp', 'slp', None),
]
MONTHSETS = {
    'jjas':     [6, 7, 8, 9],     # Fig 9 (Oct excluded)
    'jun_jul':  [6, 7],           # S6
    'aug_sep':  [8, 9],           # S7
    'oct':      [10],             # S8
}


def landfall_tc_dates():
    tr = pd.read_csv(C.TYPHOON_OUT / "landfall_typhoons_tracks.csv")
    tr['TIME'] = pd.to_datetime(tr['TIME']).dt.normalize()
    return set(tr['TIME'].dt.date.unique())


def dates_by_group(months, tc_dates):
    mjo = mj.load_mjo(months=months)              # active MJO days in the month set
    out = {g: [] for g in C.GROUP_KEY}
    for _, r in mjo.iterrows():
        if r['date'].date() in tc_dates:
            out[r['group']].append(pd.Timestamp(r['date']))
    return out


def process_field(base, var, level, tc_dates, std_cache):
    """Return dict monthset_name -> xarray.Dataset of {base}_{group}_{anom|sig}."""
    # full Jun-Oct series (needed so the calendar-month climatology uses all 5 months)
    da = dyn.load_ncar_level(var, level, range(C.YEAR_START, C.YEAR_END + 1), months=C.SEASON_MONTHS)
    da = da.sortby('time')
    z = dyn.standardize_by_month(da)
    del da; gc.collect()
    out = {}
    for name, months in MONTHSETS.items():
        dBG = dates_by_group(months, tc_dates)
        res, lat, lon = dyn.composite_standardized(z, dBG)
        ds = xr.Dataset(coords={'lat': ('lat', lat), 'lon': ('lon', lon)})
        for g, d in res.items():
            ds[f"{base}_{g}_anom"] = (('lat', 'lon'), d['anom'].astype(np.float32))
            ds[f"{base}_{g}_sig"] = (('lat', 'lon'), d['sig'].astype(np.float32))
        ds.attrs['description'] = f"{base} calendar-month standardised anomaly composite, months {months}"
        out[name] = (ds, {g: res[g]['n'] for g in C.GROUP_KEY})
    return out


def main():
    tc_dates = landfall_tc_dates()
    print(f"landfall-TC track dates: {len(tc_dates)}")
    # accumulators per monthset
    accu = {name: [] for name in MONTHSETS}
    for base, var, level in FIELDS:
        print(f"  processing {base} ({var}@{level}) ...")
        res = process_field(base, var, level, tc_dates, None)
        for name, (ds, ns) in res.items():
            accu[name].append((ds, base, ns))
            print(f"    {name}: n per group = {ns}")
        gc.collect()

    # 850-hPa vorticity (separately, needs u+v)
    print("  processing vort_850 (curl u,v@850) ...")
    u = dyn.load_ncar_level('uwnd', 850, range(C.YEAR_START, C.YEAR_END + 1), months=C.SEASON_MONTHS).sortby('time')
    v = dyn.load_ncar_level('vwnd', 850, range(C.YEAR_START, C.YEAR_END + 1), months=C.SEASON_MONTHS).sortby('time')
    u, v = xr.align(u, v, join='inner')
    lat = u['lat'].values; lon = u['lon'].values
    zv = np.empty(u.shape, dtype=np.float64)
    ut, vt = u.values, v.values
    for t in range(ut.shape[0]):
        zv[t] = dyn.relative_vorticity(ut[t], vt[t], lat, lon)
    z_vort = xr.DataArray(zv, coords=[u['time'], ('lat', lat), ('lon', lon)],
                          dims=['time', 'lat', 'lon'], name='vort')
    z_vort = dyn.standardize_by_month(z_vort)
    del u, v, ut, vt, zv; gc.collect()
    for name, months in MONTHSETS.items():
        dBG = dates_by_group(months, tc_dates)
        res, lat, lon = dyn.composite_standardized(z_vort, dBG)
        ds = xr.Dataset(coords={'lat': ('lat', lat), 'lon': ('lon', lon)})
        for g, d in res.items():
            ds[f"vort_850_{g}_anom"] = (('lat', 'lon'), d['anom'].astype(np.float32))
            ds[f"vort_850_{g}_sig"] = (('lat', 'lon'), d['sig'].astype(np.float32))
        ds.attrs['description'] = f"vort_850 calendar-month standardised anomaly, months {months}"
        accu[name].append((ds, 'vort_850', {g: res[g]['n'] for g in C.GROUP_KEY}))
        print(f"    vort {name}: n = {[res[g]['n'] for g in C.GROUP_KEY]}")

    # merge + save per monthset
    for name in MONTHSETS:
        merged = xr.merge([d for d, _, _ in accu[name]], compat='override')
        out = C.DATA_DIR / f"composite_{name}.nc"
        merged.to_netcdf(out)
        print(f"saved {out}  ({len(merged.data_vars)} vars)")


if __name__ == "__main__":
    main()
