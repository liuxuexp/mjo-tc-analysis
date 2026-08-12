"""Per-storm tropical-cyclone precipitation (TCP) over China.

The per-storm pre-processed files store daily precipitation (mm/day) on the
CHM_PRE_V2 0.1 deg grid within a 500-km mask of each storm's track (ocean + land,
NaN outside the mask). This module provides event-total TCP per storm over China
and a coastal/inland split. It builds:

  * a China land mask at the CHM grid (china_country.shp, EPSG:4326)
  * a grid-cell distance-to-coastline field (km) for the coastal/inland split
  * per-storm event-total TCP as both an area-integrated volume (10^6 m^3) and
    an area-weighted mean depth (mm) over the affected (wet) area, plus the
    affected area (km^2) and coastal/inland sub-totals.

Two clearly distinct quantities:
  TCP_total  = sum_cell depth_mm * cell_area_km2 * 1e-3          [10^6 m^3]
  TCP_depth  = sum_cell(depth_mm*area) / sum_cell(area)          [mm, over affected area]
"""
from __future__ import annotations
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.vectorized import contains
from scipy.ndimage import distance_transform_edt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


# ---------------------------------------------------------------------------
# Grid + masks (built lazily, cached on the module)
# ---------------------------------------------------------------------------
_LAT = np.arange(C.GRID_LAT0, C.GRID_LAT1 + 1e-6, C.GRID_D)
_LON = np.arange(C.GRID_LON0, C.GRID_LON1 + 1e-6, C.GRID_D)
_LON2D, _LAT2D = np.meshgrid(_LON, _LAT)
_LAND = None      # bool (nlat, nlon)
_COAST = None     # float km (nlat, nlon)


def grid_vectors():
    return _LAT.copy(), _LON.copy()


def _cell_area_km2():
    """Per-cell area (km^2): (111.32*dlon*cos lat)*(111.32*dlat)."""
    dx = 111.32 * np.cos(np.deg2rad(_LAT2D)) * C.GRID_D
    dy = 111.32 * np.ones_like(_LAT2D) * C.GRID_D
    return dx * dy


def china_land_mask():
    """Boolean China-land mask on the CHM grid (True inside china_country.shp)."""
    global _LAND
    if _LAND is None:
        geom = gpd.read_file(str(C.COUNTRY_SHP)).geometry.union_all()
        _LAND = contains(geom, _LON2D, _LAT2D)
    return _LAND


def coast_distance_km():
    """Distance (km) from each land cell to the nearest coastline.

    Uses distance_transform_edt on the land mask with km spacing at a fixed
    mid-latitude (30 N) cosine correction; documented approximation (errors
    <~12% at the 20-40 N coast band, well inside the 200-km split uncertainty).
    Ocean / outside-land cells are NaN.
    """
    global _COAST
    if _COAST is None:
        land = china_land_mask().astype(np.uint8)
        dlat_km = 111.32 * C.GRID_D
        dlon_km = 111.32 * np.cos(np.deg2rad(30.0)) * C.GRID_D
        dist = distance_transform_edt(land, sampling=[dlat_km, dlon_km])
        dist = np.where(land.astype(bool), dist, np.nan)
        _COAST = dist
    return _COAST


def coastal_mask(threshold_km=C.COAST_KM_PRIMARY):
    """Boolean land-cell mask: within threshold_km of the coastline."""
    d = coast_distance_km()
    land = china_land_mask()
    return land & (d <= threshold_km)


# ---------------------------------------------------------------------------
# Per-storm TCP
# ---------------------------------------------------------------------------
def load_storm_field(code):
    """Open per-storm pre-processed precipitation file -> DataArray prec(time,lat,lon), mm/day.

    Data are eagerly loaded and the dataset closed so per-storm callers do not
    leak open file handles (which corrupted the heap at interpreter exit:
    'free(): invalid pointer' / SIGABRT).
    """
    with xr.open_dataset(C.pre_nc_path(code)) as ds:
        da = ds['prec'].load()
    return da


def event_depth_field(code, land_only=True):
    """2-D event-total depth (mm) over the storm's days.

    NaN where no valid data. If land_only, masked to China land cells (others NaN).
    """
    prec = load_storm_field(code)
    ev = prec.sum('time', skipna=True)          # mm over storm days, NaN->0 contributions skipped
    ev = ev.where(prec.notnull().any('time'))    # cells with no data stay NaN
    arr = ev.values
    if land_only:
        lm = china_land_mask()
        arr = np.where(lm, arr, np.nan)
    return arr


def storm_tcp(code):
    """Per-storm TCP scalar metrics over China land.

    Returns dict:
      tcp_total_m3   : area-integrated volume (m^3)
      tcp_total      : 10^6 m^3  (primary 'Total TCP')
      tcp_depth      : area-weighted mean depth over affected area (mm) ('Mean event TCP')
      affected_area  : km^2 of wet China cells
      coastal_total  : 10^6 m^3 over coastal (<=200 km) cells
      inland_total   : 10^6 m^3 over inland (>200 km) cells
      coastal_depth  : mm over coastal affected area
      inland_depth   : mm over inland affected area
      n_days         : number of storm days
    """
    ev = event_depth_field(code, land_only=True)
    area = _cell_area_km2()
    wet = ev > C.WET_THRESH_MM

    tcp_total_m3 = np.nansum(np.where(wet, ev * area, 0.0)) * 1e3   # mm*km^2 -> m^3 (x1e3)
    aff_area = np.where(wet, area, 0.0).sum()
    if aff_area > 0:
        depth = np.nansum(np.where(wet, ev * area, 0.0)) / aff_area
    else:
        depth = np.nan

    # coastal / inland split
    cmask = coastal_mask(C.COAST_KM_PRIMARY)
    imask = china_land_mask() & ~cmask
    out = dict(
        tcp_total_m3=float(tcp_total_m3),
        tcp_total=float(tcp_total_m3 * 1e-6),     # 10^6 m^3
        tcp_depth=float(depth),
        affected_area=float(aff_area),
        n_days=int(load_storm_field(code).sizes['time']),
    )
    for key, m in (('coastal', cmask), ('inland', imask)):
        w = wet & m
        tot = np.nansum(np.where(w, ev * area, 0.0)) * 1e3
        a = np.where(w, area, 0.0).sum()
        out[f'{key}_total'] = float(tot * 1e-6)   # 10^6 m^3
        out[f'{key}_depth'] = float(np.nansum(np.where(w, ev * area, 0.0)) / a) if a > 0 else np.nan
        out[f'{key}_area'] = float(a)
    return out


def coastal_inland_by_threshold(code, thresholds=(100.0, 200.0, 300.0)):
    """Coastal/inland event-total TCP (10^6 m^3) at each distance threshold.

    Loads the storm field ONCE and applies each threshold's coastal mask, so the
    sensitivity test (100/200/300 km) costs a single field read per storm rather
    than three. Returns {threshold_km: (coastal_total, inland_total)} in 10^6 m^3
    (same definition as storm_tcp's coastal_total / inland_total).
    """
    ev = event_depth_field(code, land_only=True)
    area = _cell_area_km2()
    wet = ev > C.WET_THRESH_MM
    land = china_land_mask()
    d = coast_distance_km()
    out = {}
    for thr in thresholds:
        cmask = land & (d <= thr)
        wc = wet & cmask
        coast = np.nansum(np.where(wc, ev * area, 0.0)) * 1e3 * 1e-6     # -> 10^6 m^3
        wi = wet & land & ~cmask
        inl = np.nansum(np.where(wi, ev * area, 0.0)) * 1e3 * 1e-6
        out[float(thr)] = (float(coast), float(inl))
    return out


# ---------------------------------------------------------------------------
# Aggregate fields for a set of storms (for cumulative / mean-per-storm maps)
# ---------------------------------------------------------------------------
def accumulate_event_fields(codes, land_only=True):
    """Return (sum_field, count_field, lat, lon).

    sum_field  = sum over storms of event-depth (mm)            [cumulative TCP map]
    count_field= count of storms contributing (non-NaN) per cell [for mean-per-storm]
    """
    lat, lon = grid_vectors()
    s = None
    cnt = None
    for code in codes:
        ev = event_depth_field(code, land_only=land_only)
        valid = ~np.isnan(ev)
        if s is None:
            s = np.where(valid, ev, 0.0)
            cnt = valid.astype(np.float32)
        else:
            s = s + np.where(valid, ev, 0.0)
            cnt = cnt + valid.astype(np.float32)
    return s, cnt, lat, lon


def mean_per_storm_field(codes, land_only=True):
    """Mean event-depth per cell across storms that wet the cell (mm)."""
    s, cnt, lat, lon = accumulate_event_fields(codes, land_only=land_only)
    mean = np.where(cnt > 0, s / np.where(cnt == 0, 1, cnt), np.nan)
    return mean, lat, lon
