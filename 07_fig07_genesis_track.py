"""Figure 7: Genesis locations and pre-landfall tracks by landfall region.

Three stacked maps show genesis points for TCs that make landfall in South, East,
and North China (June-October 1960-2024, active MJO at landfall). Points are
colored by lifetime maximum intensity. A schematic divider (20°N, 120°E, 140°E)
marks the four genesis regions (South China Sea, South WNP, Open WNP, North WNP).

Each panel adds:
  * a gold X marking the mean genesis location for that landfall region, and
  * a stats annotation box (top-right): sample size, mean/median genesis latitude
    and longitude, pre-landfall track length (mean with 90% bootstrap CI) and
    median track length.

The legend is a single shared strip at the bottom. Genesis coordinates are
continuous; each storm belongs to exactly one landfall region.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import config as C
from lib import bootstrap as bs
from lib import plot_style as P


def _annotate(ax, sub):
    """Draw a stats annotation box (top-right): sample size, mean/median genesis
    latitude and longitude, pre-landfall track length with 90% bootstrap CI."""
    glat, glon = sub['genesis_lat'].values, sub['genesis_lon'].values
    tl = sub['pre_landfall_track_km'].values
    s_lat = bs.summarize(glat)
    s_lon = bs.summarize(glon)
    s_tl = bs.summarize(tl)
    txt = (f"n = {s_lat['n']}\n"
           f"genesis lat: {s_lat['mean']:.1f}{chr(176)}N (med {np.median(glat):.1f})\n"
           f"genesis lon: {s_lon['mean']:.1f}{chr(176)}E (med {np.median(glon):.1f})\n"
           f"track len: {s_tl['mean']:.0f} km "
           f"(90% CI {s_tl['ci_lo']:.0f}-{s_tl['ci_hi']:.0f})\n"
           f"median track: {np.median(tl):.0f} km")
    ax.text(0.98, 0.955, txt, transform=ax.transAxes, fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.5))


def _genesis_schematic(ax):
    """Draw the 20°N / 120°E / 140°E genesis-region divider and label the four
    genesis regions."""
    ax.axhline(20, color='black', linewidth=1, alpha=0.7)
    ax.plot([120, 120], [0, 20], color='black', linewidth=1, alpha=0.7,
            transform=ccrs.PlateCarree())
    ax.plot([140, 140], [0, 20], color='black', linewidth=1, alpha=0.7,
            transform=ccrs.PlateCarree())
    # Genesis boxes labeled with the four region names: South China Sea, South WNP,
    # Open WNP, North WNP. North WNP sits in the gap between the genesis point
    # clusters at ~5° clearance from every storm.
    for lon, lat, lab in [(112.5, 12, 'South China\n Sea'),
                          (130, 12, 'South WNP'),
                          (155, 12, 'Open WNP'),
                          (142, 25, 'North WNP')]:
        ax.text(lon, lat, lab, transform=ccrs.PlateCarree(), alpha=0.6, fontsize=14,
                ha='center', va='center', color='black', fontweight='bold')


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)

    extent = (105, 170, 5, 40)
    fig, axes = plt.subplots(3, 1, figsize=(9, 12.5), sharex=True, sharey=True,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    letters = 'abc'
    for j, region in enumerate(C.REGION_ORDER):
        ax = axes[j]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='#F5F5F5')
        ax.add_feature(cfeature.OCEAN, facecolor='#E0F3F8')
        P.add_china_boundaries(ax)
        _genesis_schematic(ax)

        sub = df[df['landfall_region'] == region]
        for cat in C.CATEGORY_ORDER:
            ss = sub[sub['lmi_category'] == cat]
            if len(ss):
                ax.scatter(ss['genesis_lon'], ss['genesis_lat'], s=15,
                           c=C.COLOR_MAP[C.CATEGORY_LABEL[cat]], edgecolor='black',
                           linewidth=0.2, alpha=0.8,
                           label=f"{C.CATEGORY_FULL[cat]} ({len(ss)})",
                           transform=ccrs.PlateCarree(), zorder=5)
        # mean genesis marker
        if len(sub):
            ax.scatter(sub['genesis_lon'].mean(), sub['genesis_lat'].mean(),
                       marker='X', s=120, c='gold', edgecolor='black', linewidth=1.0,
                       zorder=6, transform=ccrs.PlateCarree(), label='mean genesis')
            # stats annotation box
            _annotate(ax, sub)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle='--',
                          color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = True
        gl.bottom_labels = (j == 2)
        gl.xlabel_style = {'size': P.HOUSE_FS}
        gl.ylabel_style = {'size': P.HOUSE_FS}
        # cartopy 0.25's gridliner (draw_labels=True) suppresses ax.set_title on
        # GeoAxes under mpl 3.11. Draw the panel title in axes-fraction coords instead.
        ax.text(0.5, 1.02, f"({letters[j]}) {region}", transform=ax.transAxes,
                fontsize=P.HOUSE_FS, fontweight='bold', ha='center', va='bottom')

    # Shared figure-level legend at the bottom: the three intensity categories plus
    # the gold X mean-genesis marker, shrunk to a compact single row.
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=C.COLOR_MAP[C.CATEGORY_LABEL[c]],
                          markersize=10, label=C.CATEGORY_FULL[c])
               for c in C.CATEGORY_ORDER]
    handles.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='gold',
                              markeredgecolor='black', markersize=11, label='mean genesis'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=16,
               frameon=False, bbox_to_anchor=(0.47, 0.005),
               handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    plt.subplots_adjust(top=0.96, bottom=0.07, hspace=0.12)
    # cartopy 0.25 + matplotlib 3.11: bbox_inches='tight' drops the GeoAxes from the
    # tight bbox and renders only the bottom legend strip. Save the full figure instead;
    # the legend sits inside the bottom margin (y=0.005) so it is fully captured.
    plt.rcParams['savefig.bbox'] = 'standard'
    fig.savefig(C.fig_path('fig7-genesis_track.png'), dpi=600)
    plt.close(fig)
    # Crop the outer white margins the standard bbox leaves behind.
    P.crop_white(C.fig_path('fig7-genesis_track.png'))
    print('saved', C.fig_path('fig7-genesis_track.png'))
    for region in C.REGION_ORDER:
        n = int((df['landfall_region'] == region).sum())
        print(f"{region}: n={n}")


if __name__ == "__main__":
    main()
