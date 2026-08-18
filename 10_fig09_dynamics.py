"""10 — Figure 9 + SI circulation figures.

Reads the calendar-month composites from data03/composite_*.nc (produced by 09)
and plots:
  Fig 9   : Jun-Sep (Oct excluded). Columns = phases 1-2 vs 5-6.
            (a-b) 850-hPa relative vorticity (shading) + 850-hPa wind anomalies.
            (c-d) 500-hPa geopotential height (shading) + 500-hPa steering winds.
            Stippling marks a=0.10; non-significant vectors omitted.
  Fig S5  : Jun-Sep 200-hPa zonal wind + mean sea-level pressure (4 phases).
  Fig S6  : Jun+Jul calendar-month composites (vort_850 + hgt_500), 4 phases.
  Fig S7  : Aug+Sep, as S6.
  Fig S8  : October (vs October climatology), as S6.

Style: RdBu_r, white significance stipple (s=10, alpha=0.8), quiver (scale 15,
width 0.003, stride 3), coastline 0.5, china lw0.6 + nine-dash lw 1.2, visible
gridlines (lw 0.5, alpha 0.5) with size-18 labels, panel titles 18 bold,
dpi 600. No suptitle — the descriptive title is the filename.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader

import config as C
from lib import plot_style as P

EXTENT = [30, 180, 0, 60]        # composite domain (cartopy W,E,S,N)
LON_S, LON_E, LAT_S, LAT_N = EXTENT


def _ax(fig, rect, title, gleft=True, gbottom=True, xticks=None, yticks=None):
    """One composite-map panel: visible gridlines (lw 0.5 gray alpha 0.5) with
    size-18 labels, optional fixed tick locators, coastline 0.5 + china lw0.6
    nine lw1.2. gleft/gbottom gate the shared lon/lat labels; the panel title
    carries the phase label (the variable label is drawn separately as a bold
    block header)."""
    ax = fig.add_subplot(rect, projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    # cartopy default equal aspect -> maps keep correct geo proportions (no distortion)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.left_labels = gleft
    gl.bottom_labels = gbottom
    if xticks is not None:
        gl.xlocator = mticker.FixedLocator(xticks)
    if yticks is not None:
        gl.ylocator = mticker.FixedLocator(yticks)
    gl.xlabel_style = {'size': P.HOUSE_FS}
    gl.ylabel_style = {'size': P.HOUSE_FS}
    try:
        ax.add_geometries(Reader(str(C.CHINA_SHP)).geometries(), ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black', lw=0.6)
        ax.add_geometries(Reader(str(C.NINE_LINE)).geometries(), ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black', lw=1.2)
    except Exception:
        pass
    # cartopy 0.25 gridliner (draw_labels=True) suppresses set_title on GeoAxes
    # under mpl 3.11 — draw the title as a plain text artist instead.
    ax.text(0.5, 1.005, title, transform=ax.transAxes, fontsize=P.HOUSE_FS,
            fontweight='bold', ha='center', va='bottom')
    return ax


def _panel(ax, ds, base, group, u_base=None, v_base=None, levels=None, cmap=None,
           vector=True, stride=3):
    anom = ds[f"{base}_{group}_anom"]
    sig = ds[f"{base}_{group}_sig"]
    lat = anom['lat'].values; lon = anom['lon'].values
    # restrict to domain
    lonsel = (lon >= LON_S) & (lon <= LON_E); latsel = (lat <= LAT_N) & (lat >= LAT_S)
    A = anom.values[np.ix_(latsel, lonsel)]
    S = sig.values[np.ix_(latsel, lonsel)]
    latd = lat[latsel]; lond = lon[lonsel]
    if cmap is None:
        cmap = matplotlib.colormaps['RdBu_r']
    if levels is None:
        mx = np.nanpercentile(np.abs(A), 98)
        levels = np.linspace(-mx, mx, 11)
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
    mesh = ax.contourf(lond, latd, A, levels=levels, cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), extend='both')
    # significance stipple (white s=10)
    P.add_stipple(ax, S, latd, lond)
    # vectors where significant (u or v)
    if vector and u_base is not None:
        u = ds[f"{u_base}_{group}_anom"].values[np.ix_(latsel, lonsel)]
        v = ds[f"{v_base}_{group}_anom"].values[np.ix_(latsel, lonsel)]
        usig = ds[f"{u_base}_{group}_sig"].values[np.ix_(latsel, lonsel)]
        vsig = ds[f"{v_base}_{group}_sig"].values[np.ix_(latsel, lonsel)]
        mask = ((usig >= 1) | (vsig >= 1))
        P.add_quiver(ax, u, v, latd, lond, stride=stride, color='black', mask=mask)
    return mesh, levels


def _cbar(fig, mesh, axes_slice, label):
    cb = fig.colorbar(mesh, ax=axes_slice, orientation='horizontal',
                      fraction=0.046, pad=0.08, shrink=0.7, extend='both')
    cb.set_label(label, fontsize=14)
    cb.ax.tick_params(labelsize=14)
    return cb


def fig9():
    """Fig 9 -- Jun-Sep circulation anomalies (calendar-month climatology).

Layout: two variable-blocks stacked, each a 1x2 of phases 1-2 / 5-6; phase
labels on the panels and a bold variable header above each block; ONE shared
bottom colorbar; fixed lon/lat ticks; visible gridlines; RdBu_r +/-0.8; white
a=0.10 stipple; wind vectors only where significant."""
    ds = xr.open_dataset(C.data_path("composite_jjas.nc"))
    fig = plt.figure(figsize=(14, 6.8))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.22, top=0.90, bottom=0.16)
    blocks = [
        ('vort_850', 'uwnd_850', 'vwnd_850',
         '850-hPa relative vorticity + horizontal wind'),
        ('hgt_500', 'uwnd_500', 'vwnd_500',
         '500-hPa geopotential height + steering wind'),
    ]
    levels = np.linspace(-0.8, 0.8, 9)
    cmap = matplotlib.colormaps['RdBu_r']
    xticks = np.arange(0, 181, 50); yticks = np.arange(0, 56, 15)
    last_mesh = None
    for i, (base, ub, vb, header) in enumerate(blocks):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.05)
        for j, g in enumerate(['1-2', '5-6']):
            letter = chr(ord('a') + i * 2 + j)   # a,b (vort row) ; c,d (hgt row)
            ax = _ax(fig, inner[j], f"({letter}) {C.group_to_label(g)}",
                     gleft=(j == 0), gbottom=(i == 1), xticks=xticks, yticks=yticks)
            last_mesh, _ = _panel(ax, ds, base, g, ub, vb, levels=levels, cmap=cmap)
        # bold variable header above the block, hugging the block top (small
        # offset) so it sits close to the maps below.
        pos = outer[i].get_position(fig)
        fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.035, header,
                 ha='center', va='bottom', fontsize=P.HOUSE_FS, fontweight='bold')
    # ONE shared bottom colorbar -- both fields are standardised anomalies on
    # the same +/-0.8 scale, so a single bar suffices.
    cax = fig.add_axes([0.14, 0.09, 0.72, 0.025])
    cb = fig.colorbar(last_mesh, cax=cax, orientation='horizontal', extend='both')
    cb.ax.tick_params(labelsize=14)
    cb.set_label('Standardized anomaly', fontsize=14)
    P.save_crop(fig, 'fig9-circulation_anomaly.png')


def fig_s5():
    """Fig S5 -- Jun-Sep upper-level/surface circulation anomalies for all 4
    MJO phase groups.

Layout: TWO parts (one per variable), each a 2x2: 200-hPa zonal wind (top)
and mean sea-level pressure (bottom), each a 2x2 of phase groups
(1-2 / 3-4 over 5-6 / 7-8). Bold variable header above each part; ONE
shared bottom colorbar; fixed lon/lat ticks; RdBu_r +/-0.8; white a=0.10
stipple. No suptitle -- the descriptive title is the filename."""
    ds = xr.open_dataset(C.data_path("composite_jjas.nc"))
    fig = plt.figure(figsize=(10, 9.0))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.275, top=0.94, bottom=0.15)   # cd↔ef block gap
    parts = [
        ('uwnd_200', '200-hPa zonal wind'),
        ('slp',      'Mean sea-level pressure'),
    ]
    levels = np.linspace(-0.8, 0.8, 9)
    cmap = matplotlib.colormaps['RdBu_r']
    xticks = np.arange(0, 181, 50); yticks = np.arange(0, 56, 15)
    last_mesh = None
    letter = 0
    for i, (base, header) in enumerate(parts):
        # each part mirrors fig9's drawing: a 2x2 of phase groups, header above
        inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[i],
                                                 wspace=0.10, hspace=0.22)
        for r in range(2):
            for c in range(2):
                g = C.GROUP_KEY[r * 2 + c]            # 1-2, 3-4 / 5-6, 7-8
                lc = chr(ord('a') + letter); letter += 1
                ax = _ax(fig, inner[r, c], f"({lc}) {C.group_to_label(g)}",
                         gleft=(c == 0), gbottom=(r == 1),
                         xticks=xticks, yticks=yticks)
                last_mesh, _ = _panel(ax, ds, base, g, vector=False,
                                      levels=levels, cmap=cmap)
        pos = outer[i].get_position(fig)
        fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.03, header,
                 ha='center', va='bottom', fontsize=P.HOUSE_FS, fontweight='bold')
    # ONE shared bottom colorbar -- both parts are standardised anomalies on
    # the same +/-0.8 scale, so a single bar suffices.
    cax = fig.add_axes([0.2, 0.095, 0.6, 0.022])
    cb = fig.colorbar(last_mesh, cax=cax, orientation='horizontal', extend='both')
    cb.ax.tick_params(labelsize=14)
    cb.set_label('Standardized anomaly', fontsize=14)
    P.save_crop(fig, 'figS5-200hpa_slp.png')


def fig_monthly(name, outname, label_prefix):
    """Fig S6/S7/S8 -- calendar-month circulation composites (Jun-Jul / Aug-Sep /
    Oct).

Layout: two parts (850-hPa vorticity + wind; 500-hPa height + wind), each a
2x2 of all 4 phase groups; bold header per part; ONE shared bottom colorbar;
shared axes (lat left col, lon global bottom row); RdBu_r +/-0.8; white a=0.10
stipple; wind vectors where significant. No suptitle -- the descriptive title
is the filename."""
    ds = xr.open_dataset(C.data_path(f"composite_{name}.nc"))
    fig = plt.figure(figsize=(10, 9.0))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.275, top=0.94, bottom=0.15)   # cd↔ef block gap
    parts = [
        ('vort_850', 'uwnd_850', 'vwnd_850', '850-hPa relative vorticity + horizontal wind'),
        ('hgt_500', 'uwnd_500', 'vwnd_500', '500-hPa geopotential height + steering wind'),
    ]
    levels = np.linspace(-0.8, 0.8, 9)
    cmap = matplotlib.colormaps['RdBu_r']
    xticks = np.arange(0, 181, 50); yticks = np.arange(0, 56, 15)
    last_mesh = None
    letter = 0
    for i, (base, ub, vb, header) in enumerate(parts):
        inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[i],
                                                 wspace=0.10, hspace=0.22)
        for r in range(2):
            for c in range(2):
                g = C.GROUP_KEY[r * 2 + c]            # 1-2, 3-4 / 5-6, 7-8
                lc = chr(ord('a') + letter); letter += 1
                ax = _ax(fig, inner[r, c], f"({lc}) {C.group_to_label(g)}",
                         gleft=(c == 0), gbottom=(r == 1),
                         xticks=xticks, yticks=yticks)
                last_mesh, _ = _panel(ax, ds, base, g, ub, vb, levels=levels, cmap=cmap)
        pos = outer[i].get_position(fig)
        fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.03, header,
                 ha='center', va='bottom', fontsize=P.HOUSE_FS, fontweight='bold')
    cax = fig.add_axes([0.2, 0.095, 0.6, 0.022])
    cb = fig.colorbar(last_mesh, cax=cax, orientation='horizontal', extend='both')
    cb.ax.tick_params(labelsize=14)
    cb.set_label('Standardized anomaly', fontsize=14)
    P.save_crop(fig, outname)


def main():
    # cartopy 0.25 + mpl 3.11: bbox_inches='tight' drops GeoAxes -> figure
    # collapses. P.save reads bbox from rcParams; set 'standard' for all the
    # GeoAxes composites below.
    plt.rcParams['savefig.bbox'] = 'standard'
    fig9()
    fig_s5()
    fig_monthly('jun_jul', 'figS6-jun_jul.png', 'June + July')
    fig_monthly('aug_sep', 'figS7-aug_sep.png', 'August + September')
    fig_monthly('oct', 'figS8-october_circulation.png', 'October')


if __name__ == "__main__":
    main()
