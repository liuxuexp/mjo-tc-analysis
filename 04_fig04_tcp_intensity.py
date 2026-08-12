"""04 — Figure 4: total + event-normalized TCP by intensity.

Column layout (each column spans the full figure height):
  Column A: (a-c) accumulated TCP for Weak / Moderate / Super TCs (LMI), China
            land, stacked vertically, sharing ONE horizontal colorbar below.
  Column B: (d-f) corresponding mean TCP per storm, stacked, sharing ONE
            horizontal colorbar below.
  Column C: (g) storm-level event-total TCP distributions; (h) coastal vs inland.

The two map columns share axes -- latitude labels only on column A (a-c),
longitude labels only on the bottom maps (c, f). Map titles are trimmed to
just metric + category (e.g. "(a) Total — Weak", "(d) Mean — Weak").
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

MAP_FS = 16   # map panel (a-f) axis-tick font, sized for the narrow map columns


def _map_ax(fig, gs_rect, title, gleft=True, gbottom=True,
            extent=(100, 135.5, 17.5, 50)):
    """One China map panel. gleft/gbottom gate the shared lon/lat labels."""
    ax = fig.add_subplot(gs_rect, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_aspect('auto')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    P.add_china_boundaries(ax, nine_lw=2.0)   # bold nine-dash line
    for yy in (24, 34):
        ax.plot([extent[0], extent[1]], [yy, yy], ls='--', lw=1.5, color='gray',
                alpha=0.9, transform=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      linewidth=0, alpha=0,
                      xlocs=mticker.MultipleLocator(10),
                      ylocs=mticker.MultipleLocator(10))
    gl.top_labels = False; gl.right_labels = False
    gl.left_labels = gleft; gl.bottom_labels = gbottom
    gl.xlabel_style = {'size': MAP_FS}; gl.ylabel_style = {'size': MAP_FS}
    # cartopy 0.25 gridliner (draw_labels=True) suppresses set_title on GeoAxes
    # under mpl 3.11 — draw the title as a plain text artist instead
    ax.text(0.0, 1.02, title, transform=ax.transAxes, fontsize=P.HOUSE_FS,
            fontweight='bold', ha='left', va='bottom')
    return ax


def _fill(ax, field, lat, lon, vmax):
    """Pcolormesh the TCP field. No per-panel colorbar -- each column shares one."""
    cmap = matplotlib.colormaps['Blues'].copy(); cmap.set_under('white')
    norm = matplotlib.colors.Normalize(vmin=0.01, vmax=vmax)
    return ax.pcolormesh(lon, lat, np.ma.masked_where(~(field > 0.01), field),
                         cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                         shading='auto')


def main():
    df = pd.read_csv(C.data_path("event_table.csv"))
    df = df[df['in_jjaso']].reset_index(drop=True)

    fields_total, fields_mean = {}, {}
    lat = lon = None
    for cat in C.CATEGORY_ORDER:
        codes = [f"{c:04d}" for c in df[df['lmi_category'] == cat]['chinese_code']]
        s, cnt, lat, lon = tcp_lib.accumulate_event_fields(codes, land_only=True)
        mean = np.where(cnt > 0, s / np.where(cnt == 0, 1, cnt), np.nan)
        fields_total[cat] = s; fields_mean[cat] = mean

    # Fixed colorbar ranges: total TCP 0-3000 (a-c), mean/storm 0-60 (d-f)
    vmax_tot = 3000
    vmax_mean = 60

    fig = plt.figure(figsize=(15.5, 13.5))
    # 5 slots: A | small gap | B | wide gap | C
    outer = gridspec.GridSpec(1, 5, figure=fig,
                              width_ratios=[1, 0.04, 1, 0.27, 1.3], wspace=0,
                              left=0.055, right=0.99, top=0.965, bottom=0.05)
    # Column A: a-c total TCP + bottom colorbar; Column B: d-f mean TCP + bottom cbar
    gsA = outer[0].subgridspec(4, 1, height_ratios=[1, 1, 1, 0.06], hspace=0.15)
    gsB = outer[2].subgridspec(4, 1, height_ratios=[1, 1, 1, 0.06], hspace=0.15)
    # Column C: g (top) + h (bottom) fill the ENTIRE column height (no bottom pad).
    gsC = outer[4].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    letters = 'abcdef'

    # Column A -- (a-c) total TCP, shared axes (lat labels here, lon on c)
    for i, cat in enumerate(C.CATEGORY_ORDER):
        ax = _map_ax(fig, gsA[i],
                     f"({letters[i]}) Total — {C.CATEGORY_LABEL[cat]}",
                     gleft=True, gbottom=(i == 2))
        mesh_tot = _fill(ax, fields_total[cat], lat, lon, vmax_tot)
    cax_tot = fig.add_subplot(gsA[3])
    cb_tot = fig.colorbar(mesh_tot, cax=cax_tot, orientation='horizontal', extend='max')
    cb_tot.set_label('Total TCP (mm)', fontsize=14)
    cb_tot.ax.tick_params(labelsize=14)

    # Column B -- (d-f) mean TCP/storm, shared axes (no lat labels, lon on f)
    for i, cat in enumerate(C.CATEGORY_ORDER):
        ax = _map_ax(fig, gsB[i],
                     f"({letters[i+3]}) Mean — {C.CATEGORY_LABEL[cat]}",
                     gleft=False, gbottom=(i == 2))
        mesh_mean = _fill(ax, fields_mean[cat], lat, lon, vmax_mean)
    cax_mean = fig.add_subplot(gsB[3])
    cb_mean = fig.colorbar(mesh_mean, cax=cax_mean, orientation='horizontal', extend='max')
    cb_mean.set_label('Mean TCP/storm (mm)', fontsize=14)
    cb_mean.ax.tick_params(labelsize=14)

    # Column C -- (g) storm-level distributions
    axg = fig.add_subplot(gsC[0])
    data, labels = [], []
    for cat in C.CATEGORY_ORDER:
        v = df[df['lmi_category'] == cat]['tcp_total'].values
        data.append(v); labels.append(f"{C.CATEGORY_LABEL[cat]}\nn={len(v)}")
    bp = axg.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.6)
    for patch, cat in zip(bp['boxes'], C.CATEGORY_ORDER):
        patch.set_facecolor(C.COLOR_MAP[C.CATEGORY_LABEL[cat]]); patch.set_alpha(0.5)
    for i, cat in enumerate(C.CATEGORY_ORDER):
        v = df[df['lmi_category'] == cat]['tcp_total'].values
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.12, 0.12, len(v))
        axg.scatter(np.full(len(v), i + 1) + jitter, v, s=8, alpha=0.5,
                    color=C.COLOR_MAP[C.CATEGORY_LABEL[cat]])
        m = bs.summarize_median(v)
        axg.errorbar(i + 1, m['median'], yerr=[[m['median'] - m['ci_lo']], [m['ci_hi'] - m['median']]],
                     fmt='D', color='black', ms=7, capsize=4, zorder=5)
    axg.set_yscale('log'); axg.set_ylabel('Event-total TCP (10$^6$ m$^3$)', fontsize=P.HOUSE_FS)
    axg.tick_params(axis='both', labelsize=P.HOUSE_FS)
    axg.set_title('(g) Storm-level event TCP', fontsize=P.HOUSE_FS, fontweight='bold', loc='left')

    # Column C -- (h) coastal vs inland
    axh = fig.add_subplot(gsC[1])
    x = np.arange(3); w = 0.36
    coast_m = [bs.summarize(df[df['lmi_category']==c]['coastal_total'])['mean'] for c in C.CATEGORY_ORDER]
    inland_m = [bs.summarize(df[df['lmi_category']==c]['inland_total'])['mean'] for c in C.CATEGORY_ORDER]
    axh.bar(x - w/2, coast_m, w, color='#2E7D32', edgecolor='black', lw=0.5, label='Coastal (≤200 km)')
    axh.bar(x + w/2, inland_m, w, color='#EF6C00', edgecolor='black', lw=0.5, label='Inland (>200 km)')
    axh.set_xticks(x); axh.set_xticklabels([C.CATEGORY_LABEL[c] for c in C.CATEGORY_ORDER], fontsize=P.HOUSE_FS)
    axh.tick_params(axis='both', labelsize=P.HOUSE_FS, length=3)
    # y-axis in scientific notation: ScalarFormatter prints mantissa ticks
    # with ×10^4 as the standard matplotlib axis offset. Max mean ~1.8e4
    exp = int(np.floor(np.log10(max(max(coast_m), max(inland_m)))))
    scale = 10 ** exp
    axh.set_ylim(0, scale * 2.1)
    axh.yaxis.set_major_locator(mticker.MultipleLocator(scale / 2))
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))           # force sci -> ×10^n as the standard offset
    axh.yaxis.set_major_formatter(fmt)
    axh.yaxis.get_offset_text().set_fontsize(P.HOUSE_FS)
    axh.set_ylabel('Mean event TCP (10$^6$ m$^3$)', fontsize=P.HOUSE_FS)
    axh.set_title('(h) Coastal vs inland', fontsize=P.HOUSE_FS, fontweight='bold', loc='left')
    axh.legend(fontsize=16, frameon=False, loc='upper left')

    # cartopy 0.25 + mpl 3.11: bbox_inches='tight' drops GeoAxes -> figure collapses
    plt.rcParams['savefig.bbox'] = 'standard'
    P.save_crop(fig, 'fig4-tcp_intensity.png')
    print("vmax total mm:", round(vmax_tot, 1), "| vmax mean mm/storm:", round(vmax_mean, 1))


if __name__ == "__main__":
    main()
