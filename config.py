"""Central paths + shared constants for the Paper-4-1 shared analysis code.

This repository reproduces every revised figure and table of the Paper-4-1 major
revision (manuscript HYDROL-S-26-03282): *Role of the Madden-Julian Oscillation in
modulating tropical cyclone landfalls and intense rainfall over China*. It is
self-contained -- all paths resolve relative to this file, i.e. relative to the
repository root. Small pure helpers (wind category, phase->group, region rules) are
included here to avoid cross-directory import issues.

Layout (see README.md for full data provenance). All data lives under data/;
code (*.py + lib/) and the frozen archive/ sit at the repository root:
  Inputs (committed):
    data/typhoon_output/   landfall_*.csv, all_*.csv (derived from CMA Best Track)
    data/ERA5_MJO_1950-2024.csv   RMM1/RMM2/phase/amplitude index (version matters)
    data/shapefiles/       China boundaries (china_country / china / nine-dash)
  Inputs (placeholders -- fetch per the download-url.txt / README in each dir):
    data/pre/              per-storm TCP pre_{code}.nc (derived from CHM_PRE_V2)
    data/CHM_PRE_V2/daily/ CHM_PRE_V2_daily_{year}.nc  (TPDC, 0.1 deg daily precip)
    data/Data-NCAR/daily/  {var}.{year}.nc             (NCEP/NCAR daily reanalysis)
    data/IBTrACS/          IBTrACS.ALL v04r01          (alternative best-track robustness)
  Outputs:
    data/data03/   event-level + intermediate CSVs/NCs
    data/tables03/ Table 2 / Table 3 / Table S1
    fig/           revised main + SI figures
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directories -- all HERE-relative so the repo is relocatable.
# ---------------------------------------------------------------------------
HERE     = Path(__file__).resolve().parent                  # repository root
PROJECT  = HERE                                              # (kept name: used by make_fig_doc_pdf.py)

# ---------------------------------------------------------------------------
# Inputs -- committed derived outputs (CMA Best Track -> typhoon_output)
# ---------------------------------------------------------------------------
TYPHOON_OUT = HERE / "data" / "typhoon_output"                # landfall_*.csv, all_*.csv
PRE_DIR     = HERE / "data" / "pre"                           # pre_{code}.nc per-storm TCP
MJO_CSV     = HERE / "data" / "ERA5_MJO_1950-2024.csv"
SHAPEDIR    = HERE / "data" / "shapefiles"
COUNTRY_SHP = SHAPEDIR / "china_country.shp"                 # dissolved China land (1 geom, EPSG:4326)
CHINA_SHP   = SHAPEDIR / "china.shp"                         # detailed boundaries
NINE_LINE   = SHAPEDIR / "china_nine_dotted_line.shp"

# ---------------------------------------------------------------------------
# Inputs -- external gridded datasets (placeholders; fetch per each dir's README)
# ---------------------------------------------------------------------------
CHM_PRE_DIR = HERE / "data" / "CHM_PRE_V2" / "daily"          # CHM_PRE_V2_daily_{year}.nc
NCAR_DIR    = HERE / "data" / "Data-NCAR" / "daily"           # {var}.{year}.nc
IBTRACS_NC  = HERE / "data" / "IBTrACS" / "IBTrACS.ALL.v04r01.nc"

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
DATA_DIR    = HERE / "data" / "data03"
TABLES_DIR  = HERE / "data" / "tables03"
FIG_DIR     = HERE / "fig"
ARCHIVE_DIR = HERE / "archive"

for _d in (DATA_DIR, TABLES_DIR, FIG_DIR, ARCHIVE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Domain / time constants
# ---------------------------------------------------------------------------
YEAR_START, YEAR_END = 1960, 2024
SEASON_MONTHS   = [6, 7, 8, 9, 10]     # Jun-Oct (analysis season)
MAIN_COMP_MONTHS = [6, 7, 8, 9]         # Jun-Sep main circulation composite (Oct excluded)
OCT_MONTH        = 10

# CHM_PRE_V2 grid
GRID_LAT0, GRID_LAT1 = 18.05, 53.95
GRID_LON0, GRID_LON1 = 72.05, 135.95
GRID_D = 0.1
WET_THRESH_MM = 0.5          # "wet" cell threshold for precip-area / affected-area

# Coastal/inland split
COAST_KM_PRIMARY = 200.0
COAST_KM_SENS    = [100.0, 300.0]

# Significance
ALPHA       = 0.10
BOOT_NITER  = 1000
TCRIT_90    = 1.645           # two-tailed 90% (one-sample t on standardized anomalies)

# ACE
MS_TO_KT = 1.94384
ACE_UNIT = 1e4               # ACE in 10^4 kt^2

# Genesis boxes (kept for labelling only; primary stats use continuous coords)
EARTH_R_KM = 6371.0

# ---------------------------------------------------------------------------
# Intensity categories (top tier = Super TC, >=51 m/s)
#   get_wind_category preserves the original cutoffs
# ---------------------------------------------------------------------------
def get_wind_category(wind_speed):
    if wind_speed is None:
        return 0
    try:
        if wind_speed != wind_speed:        # NaN
            return 0
    except TypeError:
        return 0
    if wind_speed <= 17.1:       # TD
        return 0
    elif wind_speed <= 32.6:     # Weak TCs
        return 1
    elif wind_speed < 51.0:      # Moderate TCs
        return 2
    else:                        # Super TCs (>=51 m/s)
        return 3


CATEGORY_LABEL = {1: 'Weak', 2: 'Moderate', 3: 'Super'}          # short
CATEGORY_FULL  = {1: 'Weak TC', 2: 'Moderate TC', 3: 'Super TC'}  # figure labels
CATEGORY_ORDER = [1, 2, 3]
COLOR_MAP      = {'Weak': '#2E7D32', 'Moderate': '#1976D2', 'Super': '#7B1FA2'}
COLOR_FULL     = {'Weak TC': '#2E7D32', 'Moderate TC': '#1976D2', 'Super TC': '#7B1FA2'}

# ---------------------------------------------------------------------------
# MJO phase groups
# ---------------------------------------------------------------------------
GROUP_KEY   = ['1-2', '3-4', '5-6', '7-8']                       # plot keys (compact)
GROUP_LABEL = ['Phases 1-2', 'Phases 3-4', 'Phases 5-6', 'Phases 7-8']
GROUP_PHASES = {'1-2': (1, 2), '3-4': (3, 4), '5-6': (5, 6), '7-8': (7, 8)}


def phase_to_group(phase):
    """8 phases -> 4 group keys ('1-2' .. '7-8'). Returns None if inactive/invalid."""
    try:
        p = int(phase)
    except (TypeError, ValueError):
        return None
    return {1: '1-2', 2: '1-2', 3: '3-4', 4: '3-4',
            5: '5-6', 6: '5-6', 7: '7-8', 8: '7-8'}.get(p)


def group_to_label(g):
    return {'1-2': 'Phases 1-2', '3-4': 'Phases 3-4',
            '5-6': 'Phases 5-6', '7-8': 'Phases 7-8'}.get(g, g)


# ---------------------------------------------------------------------------
# Region rules (24/34°N dividers)
# ---------------------------------------------------------------------------
def region_from_lat(lat):
    """Landfall coastal region: South (<24), East (24-34), North (>=34)."""
    if lat < 24:
        return 'South China'
    elif lat <= 34:
        return 'East China'
    else:
        return 'North China'


REGION_ORDER = ['South China', 'East China', 'North China']


def landfall_region_2class(lat):
    """Fig-8 two-class landfall region."""
    return 'South China' if lat < 24 else 'East/North China'


def formation_region_from_genesis(lat, lon):
    """Genesis region from first-track point (labels kept for Fig7/8)."""
    if lat >= 20:
        return 'North WNP'
    if lon < 120:
        return 'South China Sea'
    if lon < 140:
        return 'Western Tropical WNP'
    return 'Eastern Tropical WNP'


GENESIS_ORDER = ['South China Sea', 'Western Tropical WNP',
                 'Eastern Tropical WNP', 'North WNP']


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------
def data_path(name):
    return str(DATA_DIR / name)


def table_path(name):
    return str(TABLES_DIR / name)


def fig_path(name):
    return str(FIG_DIR / name)


def pre_nc_path(code):
    """Per-storm TCP file for a 4-digit chinese_code."""
    return str(PRE_DIR / f"pre_{str(code).zfill(4)}.nc")
