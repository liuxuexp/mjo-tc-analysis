"""Large-scale circulation diagnostics for Fig 9 and supplementary figures.

Provides:
  * 850-hPa relative vorticity (curl of u,v) with cos(lat)-weighted finite
    differences.
  * circulation anomalies relative to EACH calendar month's climatology
    (not a pooled seasonal climatology, which would introduce bias).

Standardised anomaly per day d in month m:
    z(d) = (X(d) - Xbar_m) / sigma_m ,  Xbar_m,sigma_m over all days of month m
Composite = mean of z over active-MJO-phase days in the group;
significance: one-sample t = composite * sqrt(n), |t| > 1.645 (two-tailed 90%).
"""
from __future__ import annotations
import numpy as np
import xarray as xr
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


# ---------------------------------------------------------------------------
# Relative vorticity
# ---------------------------------------------------------------------------
def relative_vorticity(u, v, lat, lon, radius_km=C.EARTH_R_KM):
    """850-hPa (or any level) relative vorticity zeta = dv/dx - du/dy (s^-1).

    u, v : (nlat, nlon) arrays. lat/lon in degrees (any monotonic order; the
    actual coordinate values are passed to np.gradient so N->S grids are handled
    correctly -- no manual sign flip).
    """
    lat_r = np.deg2rad(np.asarray(lat, dtype=float))
    lon_r = np.deg2rad(np.asarray(lon, dtype=float))
    R = radius_km * 1000.0
    coslat = np.cos(lat_r)[:, None]
    # guard against cos=0 at poles
    coslat = np.where(np.abs(coslat) < 1e-6, 1e-6, coslat)
    dv_dlon = np.gradient(v, lon_r, axis=1)            # per radian of longitude
    du_dlat = np.gradient(u, lat_r, axis=0)            # per radian of latitude
    dv_dx = dv_dlon / (R * coslat)
    du_dy = du_dlat / R
    return dv_dx - du_dy


# ---------------------------------------------------------------------------
# NCEP/NCAR loading
# ---------------------------------------------------------------------------
def _files_for(var, years):
    return [str(C.NCAR_DIR / f"{var}.{y}.nc") for y in years
            if (C.NCAR_DIR / f"{var}.{y}.nc").exists()]


def load_ncar_level(var, level, years, months=None):
    """Load a NCEP daily variable at one pressure level -> DataArray(time,lat,lon).

    months: optional list to pre-filter (saves memory).
    """
    files = _files_for(var, years)
    if not files:
        raise FileNotFoundError(f"no NCEP files for {var}")

    def pre(ds):
        if level is not None:
            ds = ds.sel(level=level, method='nearest').drop_vars('level', errors='ignore')
        return ds.drop_vars(['time_bnds'], errors='ignore')

    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time',
                           preprocess=pre, chunks={'time': 120})
    da = ds[var if var in ds else list(ds.data_vars)[0]]
    da = da.sortby('time')
    if months is not None:
        da = da.sel(time=da.time.dt.month.isin(months))
    return da


# ---------------------------------------------------------------------------
# Calendar-month climatology + composites
# ---------------------------------------------------------------------------
def monthly_climatology(da):
    """Return (clim_mean, clim_std) each with a leading 'month' dimension.

    clim_mean[m-1] = mean of da over all days whose month==m (across all years)
    clim_std[m-1]  = std (ddof=1). Computed from the months present in `da`.
    """
    months_present = sorted(set(np.unique(da.time.dt.month.values.tolist())))
    mmap = {m: i for i, m in enumerate(months_present)}
    shape = (len(months_present),) + da.shape[1:]
    cmean = np.full(shape, np.nan, dtype=np.float64)
    cstd = np.full(shape, np.nan, dtype=np.float64)
    arr = da.values
    marr = da.time.dt.month.values
    for m in months_present:
        sel = marr == m
        if sel.sum() > 1:
            cmean[mmap[m]] = np.nanmean(arr[sel], axis=0)
            cstd[mmap[m]] = np.nanstd(arr[sel], axis=0, ddof=1)
        elif sel.sum() == 1:
            cmean[mmap[m]] = arr[sel][0]
    cstd = np.where(cstd == 0, np.nan, cstd)
    return cmean, cstd, mmap


def seasonal_composites(da, dates_by_group, alpha=C.ALPHA):
    """Calendar-month standardised-anomaly composites per phase group.

    da             : DataArray(time,lat,lon) for the analysis months (e.g. Jun-Sep)
    dates_by_group : {group_key: list[pd.Timestamp]} active-MJO-phase days to composite
    Returns {group: dict(anom, sig, n)} where anom/sig are (nlat,nlon) numpy arrays.
    """
    z = standardize_by_month(da)
    return composite_standardized(z, dates_by_group, alpha)


def standardize_by_month(da):
    """Return DataArray of z-scores: z(t)=(X(t)-Xbar_month)/sigma_month.

    The climatology is computed per calendar month from `da` itself (all years).
    Doing this once lets multiple day-sets (JJAS, Jun-Jul, Oct, ...) be composited
    cheaply from the same standardized series.
    """
    cmean, cstd, mmap = monthly_climatology(da)
    arr = da.values.astype(np.float64)
    month_of = pd.DatetimeIndex(da.time.values).month.values
    clim_row = np.array([mmap[m] for m in month_of])
    z = (arr - cmean[clim_row]) / cstd[clim_row]
    z = np.where(np.isfinite(z), z, 0.0)
    return xr.DataArray(z, coords=[da['time'], da['lat'], da['lon']],
                        dims=['time', 'lat', 'lon'], name=str(da.name or 'z'))


def composite_standardized(z, dates_by_group, alpha=C.ALPHA):
    """Composite a pre-standardised series over active-MJO-phase days per group."""
    tindex = pd.DatetimeIndex(z.time.values)
    arr = z.values
    out = {}
    for g, dates in dates_by_group.items():
        idx = tindex.get_indexer([pd.Timestamp(d) for d in dates])
        idx = idx[idx >= 0]
        n = idx.size
        if n == 0:
            out[g] = dict(anom=np.full(arr.shape[1:], np.nan),
                          sig=np.zeros(arr.shape[1:], np.float32), n=0)
            continue
        comp = arr[idx].mean(axis=0)
        t = comp * np.sqrt(n)
        sig = (np.abs(t) > C.TCRIT_90).astype(np.float32)
        out[g] = dict(anom=comp, sig=sig, n=n)
    return out, z['lat'].values, z['lon'].values


# ---------------------------------------------------------------------------
# High-level: build composite dataset for a field (incl. vorticity)
# ---------------------------------------------------------------------------
def composite_dataset(field_da, dates_by_group, varname, alpha=C.ALPHA):
    """Return an xarray.Dataset with {varname}_{group}_{anom,sig} vars.

    field_da already restricted to the analysis months (caller chooses Jun-Sep vs Oct).
    """
    res, lat, lon = seasonal_composites(field_da, dates_by_group, alpha)
    ds = xr.Dataset(coords={'lat': ('lat', lat), 'lon': ('lon', lon)})
    for g, d in res.items():
        ds[f"{varname}_{g}_anom"] = (('lat', 'lon'), d['anom'].astype(np.float32))
        ds[f"{varname}_{g}_sig"] = (('lat', 'lon'), d['sig'].astype(np.float32))
    return ds


def vorticity_dataarray(years, months, level=850):
    """Build a daily 850-hPa relative-vorticity DataArray (s^-1) over years/months."""
    u = load_ncar_level('uwnd', level, years, months)
    v = load_ncar_level('vwnd', level, years, months)
    # align (same grid/time)
    u, v = xr.align(u, v, join='inner')
    lat = u['lat'].values
    lon = u['lon'].values
    zv = np.empty(u.shape, dtype=np.float64)
    ut = u.values
    vt = v.values
    for t in range(ut.shape[0]):
        zv[t] = relative_vorticity(ut[t], vt[t], lat, lon)
    da = xr.DataArray(zv, coords=[u['time'], ('lat', lat), ('lon', lon)],
                      dims=['time', 'lat', 'lon'], name='vort')
    return da
