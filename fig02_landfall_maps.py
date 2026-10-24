"""Figure 2 — Chinese landfall point maps by MJO phase group.

Four panels (one per MJO phase group: 1-2, 3-4, 5-6, 7-8) scatter every Chinese
landfall point (Jun-Oct, 1960-2024, active MJO amplitude >= 1) over a shared map
extent (105-125 E, 16-46 N), coloured by LMI intensity category -- Weak /
Moderate / Super TCs. Layout is 2x2 (figsize 9x11.5, sharex/sharey), scatter
s=50 with a thin black edge, 5-deg label-only gridlines, panel titles
"(a) Phases 1-2" ..., and a single shared legend strip at the bottom; dpi 600.

Colouring basis: max_wind_category is the LMI (lifetime maximum intensity), with
Weak / Moderate / Super thresholds at 17.1 / 32.6 / 51 m/s.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import cartopy.crs as ccrs

import config as C
from lib import plot_style as P

# Intensity category labels/colours; top tier = Super TCs (>=51 m/s).
CATEGORY_MAP = {1: 'Weak TCs', 2: 'Moderate TCs', 3: 'Super TCs'}
COLOR_MAP = {'Weak TCs': '#2E7D32', 'Moderate TCs': '#1976D2', 'Super TCs': '#7B1FA2'}


def load_mjo_phase_map(year_start=1960, year_end=2024, amp_min=1.0):
    """date -> phase_group key ('1-2'..'7-8') for active-MJO days."""
    mjo = pd.read_csv(C.MJO_CSV)
    mjo['date'] = pd.to_datetime(mjo['date']).dt.date
    mjo = mjo[(mjo['date'] >= pd.to_datetime(f'{year_start}-01-01').date()) &
              (mjo['date'] <= pd.to_datetime(f'{year_end}-12-31').date())]
    mjo = mjo[mjo['amplitude'] >= amp_min].copy()
    mjo['phase_group'] = mjo['phase'].apply(C.phase_to_group)
    mjo = mjo.dropna(subset=['phase_group'])
    return pd.Series(mjo['phase_group'].values, index=mjo['date']).to_dict()


def main():
    info = pd.read_csv(C.TYPHOON_OUT / "landfall_typhoons_info.csv")
    info['code_str'] = info['chinese_code'].apply(lambda x: f"{x:04d}")
    info['landfall_time'] = pd.to_datetime(info['landfall_time'])
    info['landfall_date'] = info['landfall_time'].dt.date
    info['year'] = info['landfall_time'].dt.year
    info['month'] = info['landfall_time'].dt.month
    info = info[(info['year'] >= 1960) & (info['year'] <= 2024) &
                (info['month'].between(6, 10))]
    info = info[['code_str', 'name', 'landfall_date', 'landfall_lat', 'landfall_lon',
                 'landfall_wind_speed', 'max_wind_category']].rename(
                 columns={'max_wind_category': 'typhoon_category'})

    date_to_phase = load_mjo_phase_map()
    info['phase_group'] = info['landfall_date'].map(date_to_phase)
    info = info.dropna(subset=['phase_group'])
    grouped = {g: info[info['phase_group'] == g] for g in C.GROUP_KEY}

    # figsize 9 x 11.5: height just fits the column-width-limited maps + a
    # reduced inter-row gap (hspace). Outer slack is cropped post-save (see crop
    # below), so figH only needs to be >= the maps+gap.
    fig, axes = plt.subplots(2, 2, figsize=(9, 11.5), sharex=True, sharey=True,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    nrows, ncols = 2, 2
    gridline_interval = 5
    tit = ["(a) ", "(b) ", "(c) ", "(d) "]
    for i, (ax, g, it) in enumerate(zip(axes, C.GROUP_KEY, tit)):
        P.setup_landfall_map(ax, [105, 125, 16, 46])
        data = grouped[g]
        if len(data):
            for cat_id, cat_name in CATEGORY_MAP.items():
                sub = data[data['typhoon_category'] == cat_id]
                if len(sub):
                    ax.scatter(sub['landfall_lon'], sub['landfall_lat'], s=50,
                               c=COLOR_MAP[cat_name], edgecolor='black', linewidth=0.2,
                               alpha=0.8, transform=ccrs.PlateCarree(), label=cat_name, zorder=5)
        row, col = i // ncols, i % ncols
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                          linewidth=0, alpha=0,
                          xlocs=mticker.MultipleLocator(gridline_interval),
                          ylocs=mticker.MultipleLocator(gridline_interval))
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (col == 0)
        gl.bottom_labels = (row == nrows - 1)
        gl.xlabel_style = {'size': 18}
        gl.ylabel_style = {'size': 18}
        # cartopy 0.25 gridliner (draw_labels=True) suppresses set_title on
        # GeoAxes under mpl 3.11 — plain text artist instead.
        # offset 1.01 hugs the map below — title-to-map gap ~0.04in.
        ax.text(0.5, 1.01, it + C.group_to_label(g), transform=ax.transAxes,
                fontsize=18, fontweight='bold', ha='center', va='bottom')

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MAP[c],
                      markeredgecolor='black', markeredgewidth=0.2, markersize=15)
               for c in CATEGORY_MAP.values()]
    fig.legend(handles, ['Weak TCs', 'Moderate TCs', 'Super TCs'], loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=18, frameon=False,
               borderpad=0.1)
    # hspace 0.07: tight inter-row vertical gap. figH 11.5 keeps the bottom
    # legend close to the last map row after crop. Title (axes-frac 1.01) and
    # legend (y=0.0) are preserved.
    plt.subplots_adjust(hspace=0.09, wspace=0.06, top=0.965, bottom=0.07)
    out = C.fig_path('fig2-landfall_location.png')
    # cartopy 0.25 + mpl 3.11: bbox_inches='tight' drops the GeoAxes from the
    # tight bbox -> figure collapses to a strip. Save the full figure instead;
    # the bottom legend sits at y=0.0 (inside the margin) so it is captured.
    plt.rcParams['savefig.bbox'] = 'standard'
    plt.savefig(out, dpi=600)
    plt.close()
    # Crop the outer white margins the standard bbox leaves behind (the #2696
    # workaround preserves them). Factored into P.crop_white — see plot_style.
    P.crop_white(out)
    print({C.group_to_label(g): len(grouped[g]) for g in C.GROUP_KEY})
    print(out)


if __name__ == "__main__":
    main()
