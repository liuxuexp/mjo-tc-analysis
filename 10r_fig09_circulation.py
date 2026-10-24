"""Figure 9: JJASO circulation composites in eight panels.

Reads data/data03/composite_jjaso.nc from 09r_composite_jjaso.py.
  (a-d) 850-hPa relative vorticity anomaly (shading) with 850-hPa wind
        anomaly vectors for phase groups 1-2, 3-4, 5-6 and 7-8. Vectors are
        drawn where u or v has q < 0.05.
  (e-h) mean sea-level pressure anomaly for the same phase groups.

Each variable block shares one colour scale and colorbar. Significance is
stippled at q < 0.05, every second grid point. The 1 m/s reference arrow is
drawn inside the figure. Output: fig/fig9-circulation_jjaso.png.
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

EXTENT = [30, 180, 0, 60]
XTICKS = np.arange(0, 181, 50)   # 50/100/150 labelled
YTICKS = np.arange(0, 56, 15)    # 15/30/45 visible
QUIVER_SCALE = 38          # one scale for all panels (compact vectors)
QUIVER_STRIDE = 4          # every 4th grid point (10 degrees)
STIPPLE_STRIDE = 2         # every 2nd grid point (5 degrees)
STIPPLE_SIZE = 15          # stipple marker size


def _ax(fig, rect, title, gleft=True, gbottom=True):
    """One composite-map panel: gridlines, coastline, China boundary; title as
    a text artist (cartopy 0.25 gridliner suppresses set_title under mpl 3.11)."""
    ax = fig.add_subplot(rect, projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.left_labels = gleft
    gl.bottom_labels = gbottom
    gl.xlocator = mticker.FixedLocator(XTICKS)
    gl.ylocator = mticker.FixedLocator(YTICKS)
    gl.xlabel_style = {'size': P.HOUSE_FS}
    gl.ylabel_style = {'size': P.HOUSE_FS}
    try:
        ax.add_geometries(Reader(str(C.CHINA_SHP)).geometries(),
                          ccrs.PlateCarree(), facecolor='none',
                          edgecolor='black', lw=0.6)
        ax.add_geometries(Reader(str(C.NINE_LINE)).geometries(),
                          ccrs.PlateCarree(), facecolor='none',
                          edgecolor='black', lw=1.2)
    except Exception:
        pass
    ax.text(0.5, 1.005, title, transform=ax.transAxes, fontsize=P.HOUSE_FS,
            fontweight='bold', ha='center', va='bottom')
    return ax


def _boundaries_on_top(ax):
    """Re-draw coastline + China boundaries ABOVE the stipple layer."""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=4)
    try:
        ax.add_geometries(Reader(str(C.CHINA_SHP)).geometries(),
                          ccrs.PlateCarree(), facecolor='none',
                          edgecolor='black', lw=0.6, zorder=4.2)
        ax.add_geometries(Reader(str(C.NINE_LINE)).geometries(),
                          ccrs.PlateCarree(), facecolor='none',
                          edgecolor='black', lw=1.2, zorder=4.2)
    except Exception:
        pass


def _stipple(ax, q, lat, lon):
    """Black dots where BH-adjusted q < 0.05 (enlarged, every 2nd grid point)."""
    LON, LAT = np.meshgrid(lon[::STIPPLE_STRIDE], lat[::STIPPLE_STRIDE])
    m = (q[::STIPPLE_STRIDE, ::STIPPLE_STRIDE] < 0.05)
    if m.any():
        ax.scatter(LON[m], LAT[m], s=STIPPLE_SIZE, c='0.05', marker='.',
                   alpha=0.8, linewidths=0, transform=ccrs.PlateCarree(),
                   zorder=3.5)


def _panel(ax, ds, base, g, levels, cmap, vectors=False):
    """One map panel; returns (mesh, quiver-or-None)."""
    anom = ds[f'{base}_anom'].sel(group=g)
    q = ds[f'{base}_q'].sel(group=g)
    lat, lon = anom['lat'].values, anom['lon'].values
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
    mesh = ax.contourf(lon, lat, anom.values, levels=levels, cmap=cmap,
                       norm=norm, transform=ccrs.PlateCarree(), extend='both')
    _stipple(ax, q.values, lat, lon)
    Q = None
    if vectors:
        u = ds['uwnd_850_anom'].sel(group=g).values
        v = ds['vwnd_850_anom'].sel(group=g).values
        qu = ds['uwnd_850_q'].sel(group=g).values
        qv = ds['vwnd_850_q'].sel(group=g).values
        mask = (qu < 0.05) | (qv < 0.05)
        lo = lon[::QUIVER_STRIDE]; la = lat[::QUIVER_STRIDE]
        uu = u[::QUIVER_STRIDE, ::QUIVER_STRIDE].copy()
        vv = v[::QUIVER_STRIDE, ::QUIVER_STRIDE].copy()
        mm = mask[::QUIVER_STRIDE, ::QUIVER_STRIDE]
        uu[~mm] = np.nan; vv[~mm] = np.nan
        LON, LAT = np.meshgrid(lo, la)
        # bold black vectors, drawn above the stipple layer
        Q = ax.quiver(LON, LAT, uu, vv, scale=QUIVER_SCALE, pivot='middle',
                      color='black', width=0.0052, headwidth=5.0,
                      headlength=5.2, headaxislength=4.4,
                      transform=ccrs.PlateCarree(), zorder=3.7)
    _boundaries_on_top(ax)
    return mesh, Q


def _levels_for(ds, base):
    """Unified per-variable levels: at least +/-0.8, widened to the rounded
    98th percentile of |anomaly| over the four panels if that is larger."""
    A = np.abs(ds[f'{base}_anom'].values)
    mx = max(0.8, np.ceil(np.nanpercentile(A, 98) * 10) / 10)
    return np.linspace(-mx, mx, 9)


def _soft_rdbu():
    """RdBu_r with the saturated ends trimmed (12%-88%): the mid-range
    pastel shading lets the black vectors, stipple and coastlines pop."""
    base = matplotlib.colormaps['RdBu_r'](np.linspace(0.12, 0.88, 256))
    return matplotlib.colors.ListedColormap(base)


def fig9():
    ds = xr.open_dataset(C.data_path('composite_jjaso.nc'))
    cmap = _soft_rdbu()
    fig = plt.figure(figsize=(10, 9.0))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.275, top=0.94,
                              bottom=0.15)   # gap between the two variable blocks
    blocks = [
        ('vort_850', True, '850-hPa relative vorticity + horizontal wind'),
        ('slp', False, 'Mean sea-level pressure'),
    ]
    # One shared standardized-anomaly scale for both variables: both fields sit
    # at or below 0.8 sigma, so a single bottom colorbar covers them exactly.
    levels = max((_levels_for(ds, b).max() for b, _, _ in blocks))
    levels = np.linspace(-levels, levels, 9)
    for i, (base, vectors, header) in enumerate(blocks):
        inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[i],
                                                 wspace=0.01, hspace=0.22)
        first_Q = None
        for r in range(2):
            for c in range(2):
                g = C.GROUP_KEY[r * 2 + c]   # 1-2, 3-4 / 5-6, 7-8 (row-major)
                letter = chr(ord('a') + i * 4 + r * 2 + c)
                ax = _ax(fig, inner[r, c], f'({letter}) {C.group_to_label(g)}',
                         gleft=(c == 0), gbottom=(r == 1))
                mesh, Q = _panel(ax, ds, base, g, levels, cmap,
                                 vectors=vectors)
                if Q is not None and first_Q is None:
                    first_Q = Q
                    # unified-scale reference arrow: 2 m/s
                    ax.quiverkey(Q, 0.09, 0.06, 2, '2 m s$^{-1}$',
                                 coordinates='axes', labelpos='E',
                                 fontproperties={'size': P.BARVAL_FS})
        pos = outer[i].get_position(fig)
        fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.03, header,
                 ha='center', va='bottom', fontsize=P.HOUSE_FS,
                 fontweight='bold')
    # One shared bottom colorbar, since both variables use the same scale.
    cax = fig.add_axes([0.2, 0.095, 0.6, 0.022])
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal', extend='both')
    cb.ax.tick_params(labelsize=P.BARVAL_FS)
    cb.set_label('Standardized anomaly', fontsize=P.BARVAL_FS)
    P.save_crop(fig, 'fig9-circulation_jjaso.png')

    # console summary for the numbers report
    D = ds['n_days'].values
    for i, (base, vectors, _) in enumerate(blocks):
        for j, g in enumerate(C.GROUP_KEY):
            a = ds[f'{base}_anom'].sel(group=g).values
            q = ds[f'{base}_q'].sel(group=g).values
            print(f'{base} {g}: D={int(D[j])}, max|anom|={np.abs(a).max():.2f}, '
                  f'q<0.05 at {100 * (q < 0.05).mean():.1f}% of grid points')
    if 'uwnd_850_anom' in ds:
        for j, g in enumerate(C.GROUP_KEY):
            u = ds['uwnd_850_anom'].sel(group=g).values
            v = ds['vwnd_850_anom'].sel(group=g).values
            sp = np.hypot(u, v)
            print(f'wind {g}: max vector speed={sp.max():.1f} m/s, '
                  f'p95={np.nanpercentile(sp, 95):.1f} m/s')


def main():
    plt.rcParams['savefig.bbox'] = 'standard'   # GeoAxes + mpl 3.11 quirk
    fig9()


if __name__ == '__main__':
    main()
