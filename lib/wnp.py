"""All-WNP storm genesis helper (shared by Fig 8 panel c and Fig S4)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

import config as C
from lib import mjo as mj


def all_wnp_genesis():
    """Genesis region + genesis-date MJO phase for every WNP storm (all_typhoons).

    Filters to Jun-Oct and active MJO at genesis (amplitude >= 1). Returns a
    DataFrame with genesis_lat/lon, formation_region, group (phase@genesis).
    """
    tr = pd.read_csv(C.TYPHOON_OUT / "all_typhoons_tracks.csv")
    tr['TIME'] = pd.to_datetime(tr['TIME'])
    tr = tr.sort_values(['chinese_code', 'TIME'])
    d2g = mj.date_to_phase_map(months=C.SEASON_MONTHS)
    rows = []
    for code, g in tr.groupby('chinese_code'):
        g = g.sort_values('TIME')
        r = g.iloc[0]
        gdate = pd.Timestamp(r['TIME']).normalize()
        grp = d2g.get(gdate)
        if grp is None:
            continue
        rows.append(dict(chinese_code=int(code),
                         genesis_lat=float(r['LAT']), genesis_lon=float(r['LONG']),
                         genesis_date=gdate, group=grp,
                         lmi_category=int(g['wind_category'].iloc[0])))
    d = pd.DataFrame(rows)
    d['formation_region'] = d.apply(
        lambda r: C.formation_region_from_genesis(r['genesis_lat'], r['genesis_lon']), axis=1)
    return d
