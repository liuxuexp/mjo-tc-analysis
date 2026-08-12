"""00 — Build the master event-level table (the keystone for the analysis pipeline).

One row per China-landfalling storm (1960-2024, all months kept; flags added so
downstream figures can filter Jun-Oct + active-MJO and print denominators for
transparency and reproducibility).

Columns:
  ids/time     : year, chinese_code, name, landfall_time, month, in_jjaso
  landfall     : landfall_lat/lon/wind_speed/pressure, landfall_wind_category (int),
                 landfall_region (3-class), landfall_region2 (2-class)
  LMI          : max_wind, lmi_category (int)
  genesis      : genesis_time/lat/lon, formation_region, track_length_km (full),
                 pre_landfall_track_km, ocean_residence_days
  MJO@landfall : phase/amp/group_landfall, active_landfall (amp>=1)
  MJO@genesis  : phase/amp/group_genesis, active_genesis
  TCP          : tcp_total(10^6 m^3), tcp_depth(mm), affected_area(km^2), n_days,
                 coastal/inland total/depth/area (200 km)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import config as C
from lib import tcp as tcp_lib

INFO   = C.TYPHOON_OUT / "landfall_typhoons_info.csv"
TRACKS = C.TYPHOON_OUT / "landfall_typhoons_tracks.csv"


def haversine_km(lat1, lon1, lat2, lon2):
    R = C.EARTH_R_KM
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    info = pd.read_csv(INFO)
    info['landfall_time'] = pd.to_datetime(info['landfall_time'])
    info['month'] = info['landfall_time'].dt.month
    info['in_jjaso'] = info['month'].between(6, 10)
    info['chinese_code'] = info['chinese_code'].astype(int)

    # ---- genesis + track geometry from tracks CSV -------------------------
    tracks = pd.read_csv(TRACKS)
    tracks['TIME'] = pd.to_datetime(tracks['TIME'])
    tracks = tracks.sort_values(['chinese_code', 'TIME'])

    rows = []
    for code, g in tracks.groupby('chinese_code'):
        g = g.sort_values('TIME')
        gen = g.iloc[0]
        lat = g['LAT'].values.astype(float)
        lon = g['LONG'].values.astype(float)
        t = g['TIME'].values
        # full track length (great-circle sum of consecutive segments)
        if len(lat) > 1:
            seg = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
            full_len = float(np.nansum(seg))
        else:
            full_len = 0.0
        rows.append(dict(chinese_code=int(code),
                         genesis_time=gen['TIME'],
                         genesis_lat=float(gen['LAT']),
                         genesis_lon=float(gen['LONG']),
                         track_length_km=full_len,
                         n_track_pts=int(len(g))))
    gen_df = pd.DataFrame(rows)
    info = info.merge(gen_df, on='chinese_code', how='left')

    # pre-landfall track length + ocean-residence days (genesis -> landfall)
    def pre_landfall(r):
        g = tracks[tracks['chinese_code'] == r['chinese_code']].sort_values('TIME')
        lf = r['landfall_time']
        pre = g[g['TIME'] < lf]
        if len(pre) == 0:
            # genesis at/after landfall time-stamp; use first point -> landfall
            pts_lat = np.array([r['genesis_lat'], r['landfall_lat']])
            pts_lon = np.array([r['genesis_lon'], r['landfall_lon']])
        else:
            pts_lat = np.concatenate([pre['LAT'].values.astype(float), [r['landfall_lat']]])
            pts_lon = np.concatenate([pre['LONG'].values.astype(float), [r['landfall_lon']]])
        if len(pts_lat) > 1:
            return float(np.nansum(haversine_km(pts_lat[:-1], pts_lon[:-1], pts_lat[1:], pts_lon[1:])))
        return 0.0

    info['pre_landfall_track_km'] = info.apply(pre_landfall, axis=1)
    info['ocean_residence_days'] = (info['landfall_time'] - info['genesis_time']).dt.total_seconds() / 86400.0

    # ---- regions ----------------------------------------------------------
    info['landfall_region'] = info['landfall_lat'].apply(C.region_from_lat)
    info['landfall_region2'] = info['landfall_lat'].apply(C.landfall_region_2class)
    info['formation_region'] = info.apply(
        lambda r: C.formation_region_from_genesis(r['genesis_lat'], r['genesis_lon']), axis=1)
    info['landfall_wind_category'] = info['landfall_wind_category'].astype(int)
    info['lmi_category'] = info['max_wind_category'].astype(int)

    # ---- MJO at landfall and at genesis -----------------------------------
    mjo = pd.read_csv(C.MJO_CSV)
    mjo['date'] = pd.to_datetime(mjo['date']).dt.normalize()
    info['landfall_date'] = info['landfall_time'].dt.normalize()
    info['genesis_date'] = pd.to_datetime(info['genesis_time']).dt.normalize()

    info = info.merge(mjo[['date', 'phase', 'amplitude']].rename(
        columns={'date': 'landfall_date', 'phase': 'phase_landfall', 'amplitude': 'amp_landfall'}),
        on='landfall_date', how='left')
    info = info.merge(mjo[['date', 'phase', 'amplitude']].rename(
        columns={'date': 'genesis_date', 'phase': 'phase_genesis', 'amplitude': 'amp_genesis'}),
        on='genesis_date', how='left')

    info['group_landfall'] = info['phase_landfall'].apply(
        lambda p: C.phase_to_group(p) if pd.notna(p) else None)
    info['active_landfall'] = info['amp_landfall'].fillna(0) >= 1
    info['group_genesis'] = info['phase_genesis'].apply(
        lambda p: C.phase_to_group(p) if pd.notna(p) else None)
    info['active_genesis'] = info['amp_genesis'].fillna(0) >= 1

    # ---- per-storm TCP ----------------------------------------------------
    print(f"computing per-storm TCP for {len(info)} storms ...")
    tcp_rows = []
    for code in info['chinese_code']:
        try:
            tcp_rows.append(tcp_lib.storm_tcp(f"{int(code):04d}"))
        except Exception as e:
            print(f"  TCP fail code {code}: {e}")
            tcp_rows.append({})
    tcp_df = pd.DataFrame(tcp_rows, index=info['chinese_code'].values)
    tcp_df['chinese_code'] = info['chinese_code'].values
    info = info.merge(tcp_df, on='chinese_code', how='left')

    # ---- column order + save ---------------------------------------------
    keep = ['year', 'chinese_code', 'name', 'landfall_time', 'month', 'in_jjaso',
            'landfall_lat', 'landfall_lon', 'landfall_wind_speed', 'landfall_pressure',
            'landfall_wind_category', 'landfall_region', 'landfall_region2',
            'max_wind', 'lmi_category',
            'genesis_time', 'genesis_lat', 'genesis_lon', 'formation_region',
            'track_length_km', 'pre_landfall_track_km', 'ocean_residence_days',
            'phase_landfall', 'amp_landfall', 'group_landfall', 'active_landfall',
            'phase_genesis', 'amp_genesis', 'group_genesis', 'active_genesis',
            'tcp_total', 'tcp_depth', 'affected_area', 'n_days',
            'coastal_total', 'coastal_depth', 'coastal_area',
            'inland_total', 'inland_depth', 'inland_area']
    info = info[[c for c in keep if c in info.columns]]
    out = C.DATA_DIR / "event_table.csv"
    info.to_csv(out, index=False)
    print(f"saved {out}  rows={len(info)}")
    print("landfall wind cat:", info['landfall_wind_category'].value_counts().sort_index().to_dict())
    print("LMI cat         :", info['lmi_category'].value_counts().sort_index().to_dict())
    print("active_landfall :", info['active_landfall'].value_counts().to_dict())
    print("Jun-Oct & active:", int((info['in_jjaso'] & info['active_landfall']).sum()))


if __name__ == "__main__":
    main()
