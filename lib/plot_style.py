"""Central plotting style for the revised figures.

Figures share a consistent idiom applied inline on every element (axis labels /
ticks / titles / legend fontsize 18, bar value 14, asterisk 16 bold, dpi 600,
bbox_inches='tight', legend frameon=False). This module reproduces that exact
idiom as shared defaults for all figures.

Per-context font sizes (NOT a single global font.size — varies by context):
  HOUSE_FS  = 18   axis labels / tick labels / titles / legend  (the main style)
  BARVAL_FS = 14   bar value text
  STAR_FS   = 16   significance asterisk, bold

Map conventions (do not deviate without reason):
  precip maps : extent (100,135.5,17.5,50), Blues set_under('white'),
                china lw0.5 & nine shp lw 2.0, 24/34 N dividers lw1.5 gray dashed
                alpha0.9, gridlines lw0 alpha0 interval10 (labels only),
                title 18 bold, colorbar 'Precipitation (mm)' 14 ticks 14.
  scatter/region maps : extent (105,125,16,46), land #F5F5F5 ocean #E0F3F8,
                coastline lw0.4, china lw0.5 nine lw2.0 dashed.
  composites : RdBu_r levels arange(-3,3.1,0.5), white stipple s10,
               quiver scale15 width0.003.

NO suptitle on any figure — the descriptive title is carried by the output
filename (e.g. fig1-ace_tcp.png, fig2-landfall_location.png).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

# ---------------------------------------------------------------------------
# Global rcParams — family + dpi + bbox + grid + axes lw ONLY.
# (Never force a global font.size; matplotlib default applies to un-styled
#  elements, exactly as in the original figures.)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'figure.dpi': 120,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    # Render scientific-notation offsets as ×10^n (mathtext superscript), not the
    # computing-style "1en". Applies to every figure's default ScalarFormatter, so
    # all TCP / ACE axes print publication-standard powers (journal convention).
    'axes.formatter.use_mathtext': True,
})

# Per-context font sizes (house style)
HOUSE_FS = 18      # axis labels / ticks / titles / legend
BARVAL_FS = 14     # bar value text
STAR_FS = 16       # significance asterisk (bold)

# MJO phase colours (warm -> cool progression across the 4 pairs)
PHASE_COLORS = {
    '1-2': '#B2182B', '3-4': '#EF8A62', '5-6': '#67A9CF', '7-8': '#2166AC',
}
INTENSITY_COLORS = dict(C.COLOR_MAP)   # Weak/Moderate/Major (#2E7D32/#1976D2/#7B1FA2)


# ---------------------------------------------------------------------------
# China boundary
# ---------------------------------------------------------------------------
def add_china_boundaries(ax, projection=ccrs.PlateCarree(), nine_lw=2.0):
    try:
        ax.add_geometries(Reader(str(C.CHINA_SHP)).geometries(), projection,
                          facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.8)
    except Exception as e:
        print(f"china boundary failed: {e}")
    try:
        ax.add_geometries(Reader(str(C.NINE_LINE)).geometries(), projection,
                          facecolor='none', edgecolor='black', linewidth=nine_lw, linestyle='--')
    except Exception as e:
        print(f"nine-dash failed: {e}")


# ---------------------------------------------------------------------------
# Scatter / region landfall map
# New code should use fig.add_subplot(..., projection=ccrs.PlateCarree()) +
# setup_landfall_map(ax) (not manual add_axes rects).
# ---------------------------------------------------------------------------
def setup_landfall_map(ax, extent=(105, 125, 16, 46), dividers=True):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='black')
    ax.add_feature(cfeature.LAND, facecolor='#F5F5F5')
    ax.add_feature(cfeature.OCEAN, facecolor='#E0F3F8')
    add_china_boundaries(ax)
    if dividers:
        for yy in (24, 34):
            ax.plot([extent[0], extent[1]], [yy, yy], linestyle='--', linewidth=1.5,
                    color='gray', alpha=0.9, transform=ccrs.PlateCarree())
    return ax


def china_ax(fig, rect, extent=(100, 135.5, 17.5, 50), dividers=True,
             projection=ccrs.PlateCarree()):
    """China-focused cartopy axes at a manual rect (kept for back-compat).

    Prefer fig.add_subplot + setup_landfall_map for new figures. The coastline /
    land / ocean / divider styling uses dividers lw 1.5.
    """
    ax = fig.add_axes(rect, projection=projection)
    ax.set_extent(extent, crs=projection)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='black')
    ax.add_feature(cfeature.LAND, facecolor='#F5F5F5')
    ax.add_feature(cfeature.OCEAN, facecolor='#E0F3F8')
    add_china_boundaries(ax)
    if dividers:
        for yy in (24, 34):
            ax.plot([extent[0], extent[1]], [yy, yy], ls='--', lw=1.5,
                    color='gray', alpha=0.9, transform=projection)
    return ax


# ---------------------------------------------------------------------------
# Precipitation field
# ---------------------------------------------------------------------------
def plot_precip_field(ax, field, lat, lon, vmax=1000, vmin=0.01,
                      title=None, cmap=None, extend='max', cbar=True,
                      label='Precipitation (mm)', title_fs=HOUSE_FS):
    """Filled China precip map (Blues, white under vmin). NaN/0 -> white.

    cbar label & ticks at fontsize 14, title 18 bold.
    """
    if cmap is None:
        cmap = plt.cm.Blues.copy()
        cmap.set_under('white')
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_where(~(field > vmin), field),
                         cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
                         shading='auto')
    if title:
        ax.set_title(title, fontsize=title_fs, fontweight='bold')
    if cbar:
        cb = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04,
                          shrink=0.85, extend=extend)
        cb.set_label(label, fontsize=14)
        cb.ax.tick_params(labelsize=14)
    return mesh


# ---------------------------------------------------------------------------
# Stipple / quiver
# ---------------------------------------------------------------------------
def add_stipple(ax, sig, lat, lon, color='white', s=10, alpha=0.8, stride=1):
    """Significance dots where sig>=1 (white, s=10)."""
    LON, LAT = np.meshgrid(lon[::stride], lat[::stride])
    m = sig[::stride, ::stride] >= 1
    if np.any(m):
        ax.scatter(LON[m], LAT[m], s=s, c=color, marker='.', alpha=alpha,
                   linewidths=0, transform=ccrs.PlateCarree())


def add_quiver(ax, u, v, lat, lon, stride=3, scale=15, color='black',
               mask=None):
    """Anomaly wind vectors (scale 15, width 0.003)."""
    lo = lon[::stride]; la = lat[::stride]
    uu = u[::stride, ::stride]; vv = v[::stride, ::stride]
    if mask is not None:
        mm = mask[::stride, ::stride]
        uu = np.where(mm, uu, np.nan); vv = np.where(mm, vv, np.nan)
    LON, LAT = np.meshgrid(lo, la)
    ax.quiver(LON, LAT, uu, vv, scale=scale, width=0.003, pivot='middle',
              color=color, transform=ccrs.PlateCarree())


# ---------------------------------------------------------------------------
# Bar annotations / phase-ratio bars
# ---------------------------------------------------------------------------
def annotate_counts(ax, bars, counts, rates=None, ylim_top=None, fontsize=BARVAL_FS):
    """Print n (and optional rate) above each bar (value text 14)."""
    for i, b in enumerate(bars):
        x = b.get_x() + b.get_width() / 2
        y = b.get_height()
        txt = f"n={counts[i]}"
        if rates is not None:
            txt += f"\n{rates[i]:.2f}"
        ax.text(x, y, txt, ha='center', va='bottom', fontsize=fontsize)
    if ylim_top is not None:
        ax.set_ylim(0, ylim_top)


def phase_ratio_bars(ax, res, color='#444444', title=None, ylabel=None,
                     ylim=(0, 2.4), null_line=1.0, fontsize=HOUSE_FS):
    """Bar plot of residence-ratio (multiples of no-modulation) per phase group.

    Styled to the house idiom: tick labels / title / ylabel at fontsize 18,
    `n` and significance stars (16 bold) printed ABOVE each bar (star hugs the
    error-bar top, n one line above it), dashed null line at `null_line`.

    res : dict group -> {ratio, ci_lo, ci_hi, stars, n, ndays}.
    """
    x = np.arange(len(C.GROUP_KEY))
    ratio = [res[g]['ratio'] for g in C.GROUP_KEY]
    lo = [max(0.0, res[g]['ratio'] - res[g]['ci_lo']) for g in C.GROUP_KEY]
    hi = [max(0.0, res[g]['ci_hi'] - res[g]['ratio']) for g in C.GROUP_KEY]
    bars = ax.bar(x, ratio, 0.55, color=color, edgecolor='black', linewidth=0.6,
                  yerr=[lo, hi], capsize=4, error_kw=dict(elinewidth=1.0, alpha=0.8))
    ax.axhline(null_line, color='gray', ls='--', lw=1.3, alpha=0.8)
    yrange = ylim[1] - ylim[0]
    # Numeric labels go ABOVE each bar (not under the axis): the significance
    # star hugs the error-bar top, the sample size n sits one line higher.
    # Extend ylim for headroom so the topmost label never clips.
    top_needed = max(ratio[i] + hi[i] for i in range(len(C.GROUP_KEY))) + 0.20 * yrange
    for i, g in enumerate(C.GROUP_KEY):
        xc = bars[i].get_x() + bars[i].get_width() / 2
        top = ratio[i] + hi[i]
        if res[g]['stars']:
            ax.text(xc, top + 0.03 * yrange, res[g]['stars'], ha='center',
                    va='bottom', fontsize=STAR_FS, fontweight='bold')
        ax.text(xc, top + 0.11 * yrange, f"n={res[g]['n']}", ha='center',
                va='bottom', fontsize=BARVAL_FS, color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels([C.group_to_label(g) for g in C.GROUP_KEY], fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_ylim(ylim[0], max(ylim[1], top_needed))
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    return bars


# ---------------------------------------------------------------------------
# Save — dpi 600; bbox read from rcParams['savefig.bbox'] (default 'tight', set
# above). GeoAxes figures under cartopy 0.25 + mpl 3.11 override it to 'standard'
# locally before calling save(): 'tight' drops the GeoAxes from the tight bbox
# and the figure collapses to a legend-only strip (cartopy #2696). Removing the
# hardcoded kwarg makes that override effective.
# NO suptitle: the figure's descriptive title is embedded in `name` as fig{N}-XX.png.
# ---------------------------------------------------------------------------
def save(fig, name):
    fig.savefig(C.fig_path(name), dpi=600)
    plt.close(fig)
    print("saved", C.fig_path(name))


def crop_white(path, border_in=0.1, dpi=600):
    """Crop the outer white margins from a saved figure, in place.

    Restores the tight bbox that bbox_inches='tight' would produce but cannot be
    used on GeoAxes figures under the cartopy #2696 standard-bbox workaround
    (aspect-locked GeoAxes + the standard bbox leave irregular outer white margins
    that tight-bbox used to trim). Deterministic: bbox of all non-white content
    via ImageChops.difference vs pure white, then a uniform `border_in` margin
    re-added at `dpi` px/in. No-op if the image is all white; never clips ink
    (the content bbox is inclusive of every drawn element).

    Call AFTER the figure is saved with savefig.bbox='standard'.
    """
    from PIL import Image, ImageChops
    im = Image.open(path).convert('RGB')
    bb = ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255))).getbbox()
    if not bb:
        return
    m = int(round(border_in * dpi))
    im.crop((max(0, bb[0] - m), max(0, bb[1] - m),
             min(im.width, bb[2] + m), min(im.height, bb[3] + m))).save(path)


def save_crop(fig, name, border_in=0.1):
    """save() then crop_white() — drop-in for save() on GeoAxes figures that have
    set savefig.bbox='standard'. Crops the outer margins the standard bbox leaves
    behind so the result matches what bbox_inches='tight' would have produced.
    See crop_white for the rationale and the cartopy #2696 workaround."""
    save(fig, name)
    crop_white(C.fig_path(name), border_in=border_in)
