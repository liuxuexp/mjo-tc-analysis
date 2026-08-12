"""11 — Alternative best-track robustness (Figure S3).

Matches CMA landfall storms to IBTrACS (ALL v04r01) by genesis proximity
(first-track point within 300 km and within +/-24 h), then replicates the
headline intensity classification and the phase x intensity landfall breakdown
using IBTrACS winds. TCP itself is precipitation-based and storm-identity is
shared, so robustness is on the track/intensity classification (the dataset that
actually changes). Wind-averaging differences are documented.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from lib import mjo as mj
from lib import plot_style as P


def load_ibtracs_wp():
    """Return list of dicts: sid, genesis_time, lat0, lon0, max_wind (kt)."""
    ds = xr.open_dataset(str(C.IBTRACS_NC))
    basin = ds['basin'].values.astype(str)        # (storm, date_time)
    wp = np.where(basin[:, 0] == 'WP')[0]
    iso = ds['iso_time'].values.astype(str)
    lat = ds['usa_lat'].values
    lon = ds['usa_lon'].values
    wind = ds['usa_wind'].values                    # kt
    season = ds['season'].values
    sid = ds['sid'].values.astype(str)
    out = []
    for i in wp:
        s = int(season[i])
        if s < C.YEAR_START or s > C.YEAR_END:
            continue
        la = lat[i]; lo = lon[i]; wi = wind[i]; tm = iso[i]
        valid = ~(np.isnan(la) | np.isnan(lo)) & (wi >= 0) & (tm != '')
        if valid.sum() == 0:
            continue
        idx0 = np.where(valid)[0][0]
        try:
            t0 = pd.Timestamp(str(tm[idx0]))
        except Exception:
            continue
        out.append(dict(sid=str(sid[i]), season=s,
                        t0=t0, lat0=float(la[idx0]), lon0=float(lo[idx0]),
                        max_wind=float(np.nanmax(wi[valid]))))
    ds.close()
    return out


def _hav(la1, lo1, la2, lo2):
    R = C.EARTH_R_KM
    la1, la2 = np.radians(la1), np.radians(la2)
    dla = np.radians(la2 - la1); dlo = np.radians(lo2 - lo1)
    a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    ib = load_ibtracs_wp()
    print(f"IBTrACS WP storms (1960-2024): {len(ib)}")
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso']].copy()
    df['genesis_time'] = pd.to_datetime(df['genesis_time'])

    matched, dists = [], []
    for _, r in df.iterrows():
        best, bestd = None, 1e9
        for s in ib:
            dt_h = abs((s['t0'] - r['genesis_time']).total_seconds()) / 3600.0
            if dt_h > 24:
                continue
            d = _hav(r['genesis_lat'], r['genesis_lon'], s['lat0'], s['lon0'])
            if d < bestd:
                bestd = d; best = s
        if best is not None and bestd < 300:
            matched.append(dict(chinese_code=r['chinese_code'], ib_sid=best['sid'],
                                match_km=bestd, cma_lmi=r['max_wind'],
                                ib_wind_kt=best['max_wind'],
                                ib_wind_ms=best['max_wind'] / C.MS_TO_KT,
                                group_landfall=r['group_landfall'],
                                landfall_region=r['landfall_region'],
                                active_landfall=r['active_landfall']))
            dists.append(bestd)
    m = pd.DataFrame(matched)
    m['ib_category'] = m['ib_wind_ms'].apply(C.get_wind_category)
    print(f"matched {len(m)}/{len(df)} ({100*len(m)/len(df):.1f}%), median dist {np.median(dists):.0f} km")

    # join CMA LMI cat for comparison
    df['cma_cat'] = df['lmi_category']
    m = m.merge(df[['chinese_code', 'lmi_category']], on='chinese_code', how='left')

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    # (a) match distance histogram + match rate
    ax = axs[0]
    ax.hist(dists, bins=30, color='#455A64', edgecolor='black')
    ax.axvline(np.median(dists), color='red', ls='--', label=f"median {np.median(dists):.0f} km")
    ax.set_xlabel('Genesis match distance (km)', fontsize=P.HOUSE_FS)
    ax.set_ylabel('Matched storms', fontsize=P.HOUSE_FS, labelpad=-3)
    ax.tick_params(axis='both', labelsize=P.HOUSE_FS)
    ax.set_title('(a) CMA–IBTrACS genesis match',
                 fontsize=P.HOUSE_FS, fontweight='bold')
    # match rate as an in-axes annotation; smaller than axis text and dropped
    # below the title edge; median legend matches its size
    ax.text(0.98, 0.90, f'{len(m)}/{len(df)} storms ({100*len(m)/len(df):.0f}%)',
            transform=ax.transAxes, ha='right', va='top', fontsize=P.BARVAL_FS)
    ax.legend(fontsize=P.BARVAL_FS, frameon=False)

    # (b) CMA vs IBTrACS LMI scatter (m/s)
    ax = axs[1]
    ax.scatter(m['cma_lmi'], m['ib_wind_ms'], s=20, alpha=0.5, edgecolor='none')
    mx = max(m['cma_lmi'].max(), m['ib_wind_ms'].max())
    ax.plot([0, mx], [0, mx], 'r--', lw=1)
    corr = np.corrcoef(m['cma_lmi'], m['ib_wind_ms'])[0, 1]
    ax.set_xlabel('CMA LMI (m/s)', fontsize=P.HOUSE_FS); ax.set_ylabel('IBTrACS max wind (m/s)', fontsize=P.HOUSE_FS, labelpad=-3)
    ax.tick_params(axis='both', labelsize=P.HOUSE_FS)
    ax.set_title('(b) Intensity agreement', fontsize=P.HOUSE_FS, fontweight='bold')
    # correlation coefficient in upper-left of the scatter (points hug the 1:1
    # diagonal, leaving the high-IBTrACS/low-CMA corner empty)
    ax.text(0.04, 0.95, f'r = {corr:.2f}', transform=ax.transAxes, ha='left', va='top',
            fontsize=P.HOUSE_FS)

    # (c) phase x intensity landfall: CMA vs IBTrACS classification (active subset)
    ax = axs[2]
    ma = m[m['active_landfall'] == True]
    x = np.arange(4); w = 0.38; bw = 0.30   # bw < w -> fig1-(a) gap within each pair
    for i, (catcol, lab, col) in enumerate([('lmi_category', 'CMA LMI', '#1976D2'),
                                            ('ib_category', 'IBTrACS', '#EF6C00')]):
        # residence-normalized total landfalls per phase (all cats) using each classification
        counts = {g: int((ma[catcol].notna()) & (ma['group_landfall'] == g)).sum() if False
                  else int(((ma['group_landfall'] == g)).sum()) for g in C.GROUP_KEY}
        # per-phase count is the same set; show intensity-major share instead
        major_share = []
        for g in C.GROUP_KEY:
            sub = ma[ma['group_landfall'] == g]
            major_share.append(100 * (sub[catcol] == 3).sum() / max(1, len(sub)))
        ax.bar(x + (i - 0.5) * w, major_share, bw, color=col, edgecolor='black', lw=0.5, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([C.group_to_label(g).replace(' ', '\n') for g in C.GROUP_KEY], fontsize=P.HOUSE_FS)
    ax.tick_params(axis='both', labelsize=P.HOUSE_FS)
    ax.set_ylabel('Super-TC share of landfalls (%)', fontsize=P.HOUSE_FS, labelpad=-3)
    ax.set_title('(c) Super-TC landfall share', fontsize=P.HOUSE_FS, fontweight='bold')
    ax.legend(fontsize=P.HOUSE_FS, frameon=False, loc='upper right',
              bbox_to_anchor=(1, 1.02))
    plt.tight_layout(w_pad=0.5)
    # per-panel horizontal nudge (positive = right), right edge held fixed so
    # the figure boundary doesn't move: (b) left tightens a-b, (c) right widens b-c
    def _nudge(ax, dx):
        p = ax.get_position()
        ax.set_position([p.x0 + dx, p.y0, p.width - dx, p.height])
    _nudge(axs[1], -0.01)   # (a)-(b)
    _nudge(axs[2], -0.02)   # (b)-(c)
    P.save(fig, 'figS3-ibtracs_robustness.png')
    m.to_csv(C.data_path("ibtracs_matched.csv"), index=False)


if __name__ == "__main__":
    main()
