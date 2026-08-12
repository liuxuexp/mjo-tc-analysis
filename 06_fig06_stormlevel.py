"""06 — Figure 6: storm-level TCP by LMI category x MJO phase.

Layout (references Fig 1-5 design + Fig 4 map idiom):
  Left column (a over b), shared x-axis (phase groups on b only):
    (a) Event-total TCP distributions for every intensity x phase group
        (individual storms, median diamond, 90% CI, n printed).
    (b) Mean precipitation area (>0.5 mm wet area) by intensity x phase.
  Right block (c-f) 2x2 mean-TCP-per-storm maps, one per phase, on an identical
  scale with ONE shared horizontal colorbar (Fig 4 idiom: shared axes -- lat
  labels on the left column only, lon labels on the bottom row only; Blues
  set_under('white'); no per-panel colorbar). Caption states these are grid-cell
  means per storm (not area integrals).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import config as C
from lib import tcp as tcp_lib
from lib import bootstrap as bs
from lib import plot_style as P


def _map_ax(fig, rect, title, gleft=True, gbottom=True,
            extent=(100, 135.5, 17.5, 50)):
    """One China precip-map panel (Fig 4 idiom). gleft/gbottom gate the shared
    lon/lat labels; default (equal) aspect keeps the maps in correct geographic
    proportion (no stretching)."""
    ax = fig.add_subplot(rect, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    P.add_china_boundaries(ax)
    for yy in (24, 34):
        ax.plot([extent[0], extent[1]], [yy, yy], ls='--', lw=1.5, color='gray',
                alpha=0.9, transform=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      linewidth=0, alpha=0,
                      xlocs=mticker.MultipleLocator(10),
                      ylocs=mticker.MultipleLocator(10))
    gl.top_labels = False; gl.right_labels = False
    gl.left_labels = gleft; gl.bottom_labels = gbottom
    gl.xlabel_style = {'size': P.HOUSE_FS}
    ys = {'size': P.HOUSE_FS}
    if gleft:                       # c/e (left column): lat labels vertical (竖排)
        ys['rotation'] = 'vertical'
    gl.ylabel_style = ys
    # outward tick marks at gridline loci on edges that carry labels
    # GeoAxes axis-ticks are separate from the gridliner; axis tick labels disabled here
    ax.set_xticks([100, 110, 120, 130], crs=ccrs.PlateCarree())
    ax.set_yticks([20, 30, 40, 50], crs=ccrs.PlateCarree())
    ax.tick_params(direction='out', length=6, width=1.0, color='black',
                   bottom=gbottom, left=gleft, top=False, right=False,
                   labelbottom=False, labelleft=False, labeltop=False, labelright=False)
    # cartopy 0.25 gridliner (draw_labels=True) suppresses set_title on GeoAxes
    # under mpl 3.11 — draw the title as a plain text artist instead
    # y 1.005 (was 1.02): hug the map top — titles brought closer
    ax.text(0.0, 1.005, title, transform=ax.transAxes, fontsize=P.HOUSE_FS,
            fontweight='bold', ha='left', va='bottom')
    return ax


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso'] & df['active_landfall']].reset_index(drop=True)

    # 2 blocks: left = a/b column (shared x), right = 2x2 phase maps + shared cbar
    fig = plt.figure(figsize=(16.32, 10.098))
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05,
                              width_ratios=[0.82, 1.0],
                              left=0.057, right=0.99, top=0.955, bottom=0.08)
    gsL = outer[0].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.12)
    # right block: 2x2 maps over a shared colorbar. Nested gridspecs so the maps'
    # inter-row gap (c/d <-> e/f) and the colorbar-to-maps gap are independent
    gsR = outer[1].subgridspec(2, 1, height_ratios=[1.0, 0.11], hspace=0.05)
    gsMaps = gsR[0].subgridspec(2, 2, hspace=0.17, wspace=0.02)

    x = np.arange(len(C.GROUP_KEY))
    w = 0.26

    # ---- (a) intensity x phase distribution matrix (top of left column) ----
    axa = fig.add_subplot(gsL[0])
    rng = np.random.default_rng(7)
    axa.set_yscale('log')
    max_wtop = 0.0
    for i, cat in enumerate(C.CATEGORY_ORDER):
        data, positions = [], []
        for gi, g in enumerate(C.GROUP_KEY):
            v = df[(df['lmi_category'] == cat) & (df['group_landfall'] == g)]['tcp_total'].values
            if len(v) == 0:
                continue
            data.append(v); positions.append(gi + (i - 1) * w)
        bp = axa.boxplot(data, positions=positions, widths=0.20, showfliers=False, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(C.COLOR_MAP[C.CATEGORY_LABEL[cat]]); patch.set_alpha(0.45)
        for k, (v, p) in enumerate(zip(data, positions)):
            m = bs.summarize_median(v)
            axa.errorbar(p, m['median'], yerr=[[m['median'] - m['ci_lo']], [m['ci_hi'] - m['median']]],
                         fmt='D', color='black', ms=5, capsize=3, zorder=5)
            wtop = float(np.max(bp['whiskers'][2 * k + 1].get_ydata()))
            max_wtop = max(max_wtop, wtop)
            axa.text(p, wtop * 1.12, f"{len(v)}", ha='center', va='bottom',
                     fontsize=P.BARVAL_FS)   # plain count, horizontal (no "n=")
    # log-axis headroom: extend the top past the tallest count label
    _lo, _hi = axa.get_ylim(); axa.set_ylim(_lo, max_wtop * 2.2)
    axa.set_xticks(x)
    axa.tick_params(axis='y', labelsize=P.HOUSE_FS)
    axa.set_ylabel('Event-total TCP per storm (10$^6$ m$^3$)', fontsize=15)
    axa.set_title('(a) Storm-level event TCP by LMI × phase',
                  fontsize=P.HOUSE_FS, fontweight='bold', loc='left')
    handles = [plt.Rectangle((0, 0), 1, 1, color=C.COLOR_MAP[C.CATEGORY_LABEL[c]])
               for c in C.CATEGORY_ORDER]   # solid (no alpha)
    axa.legend(handles, [C.CATEGORY_FULL[c] for c in C.CATEGORY_ORDER],
               fontsize=P.HOUSE_FS, frameon=False, loc='lower left',
               bbox_to_anchor=(0, -0.02), labelspacing=0.2)

    # ---- (b) precipitation area by intensity x phase (shares x with a) ----
    axb = fig.add_subplot(gsL[1], sharex=axa)
    bw = 0.20   # bar width < dodge w=0.26 -> visible gap between bars within a group
    all_means = []
    for i, cat in enumerate(C.CATEGORY_ORDER):
        means = []
        for g in C.GROUP_KEY:
            v = df[(df['lmi_category'] == cat) & (df['group_landfall'] == g)]['affected_area'].values
            means.append(bs.summarize(v)['mean'] / 1e3)   # 10^3 km^2
        all_means.extend(means)
        xc = x + (i - 1) * w
        axb.bar(xc, means, bw, color=C.COLOR_MAP[C.CATEGORY_LABEL[cat]],
                edgecolor='black', lw=0.5)
        for j, val in enumerate(means):    # value above each bar
            axb.text(xc[j], val, f"{val:.0f}", ha='center', va='bottom',
                     fontsize=P.BARVAL_FS)
    axb.set_xticks(x)
    axb.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY], fontsize=P.HOUSE_FS)
    axb.tick_params(axis='both', labelsize=P.HOUSE_FS)
    axb.set_ylabel('Mean precip area (10$^3$ km$^2$)', fontsize=15)
    axb.set_title('(b) Precipitation area per storm',
                  fontsize=P.HOUSE_FS, fontweight='bold', loc='left')
    axb.set_ylim(0, max(all_means) * 1.22)   # headroom for value labels
    # shared x-axis: phase labels on (b) only, hide on (a)
    axa.tick_params(labelbottom=False)

    # ---- (c-f) mean-per-storm maps by phase (common scale, shared axes) ----
    fields = {}
    lat = lon = None
    for g in C.GROUP_KEY:
        codes = [f"{c:04d}" for c in df[df['group_landfall'] == g]['chinese_code']]
        m, lat, lon = tcp_lib.mean_per_storm_field(codes, land_only=True)
        fields[g] = m
    # Fixed colorbar range: mean TCP/storm 0-70
    vmax = 70
    cmap = matplotlib.colormaps['Blues'].copy(); cmap.set_under('white')
    norm = matplotlib.colors.Normalize(vmin=0.01, vmax=vmax)
    mesh = None
    # phase order -> 2x2 grid: (c)=1-2 (d)=3-4 top row, (e)=5-6 (f)=7-8 bottom row
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for j, g in enumerate(C.GROUP_KEY):
        r, ccol = slots[j]
        ax = _map_ax(fig, gsMaps[r, ccol], f"({chr(99 + j)}) {C.group_to_label(g)}",
                     gleft=(ccol == 0), gbottom=(r == 1))
        mesh = ax.pcolormesh(lon, lat, np.ma.masked_where(~(fields[g] > 0.01), fields[g]),
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto')
    # ONE shared colorbar below the 2x2 map block (Fig 4 idiom). Manual axes so
    # the bar thickness (→30% of the reserved strip) and the width (→narrower,
    # centered) are independent of the maps block
    strip = gsR[1].get_position(fig)
    bar_h = strip.height * 0.30
    bar_w = strip.width * 0.70
    bx = strip.x0 + (strip.width - bar_w) / 2
    by = strip.y0 + (strip.height - bar_h) / 2
    cax = fig.add_axes([bx, by, bar_w, bar_h])
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal', extend='max')
    cb.set_label('Mean TCP/storm (mm)', fontsize=17)
    cb.ax.tick_params(labelsize=17)

    # cartopy 0.25 + mpl 3.11: bbox_inches='tight' drops GeoAxes -> figure collapses
    plt.rcParams['savefig.bbox'] = 'standard'
    P.save_crop(fig, 'fig6-stormlevel_tcp.png')
    print("map vmax mm/storm:", round(vmax, 1))


if __name__ == "__main__":
    main()
