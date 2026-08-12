# Paper-4-1 Major-Revision Reproduducible Analysis Package

Manuscript: **HYDROL-S-26-03282** (major revision) — *Role of the Madden–Julian
Oscillation in modulating tropical cyclone landfalls and intense rainfall over
China.*

This directory (`archive/package/`) freezes the derived data needed to
reproduce every revised figure and table. Regenerate from source with
`cd <repo-root> && python run_all.py && python archive/export.py`.

## Contents
| File | Description |
|---|---|
| `ERA5_MJO_1950-2024.csv` | Exact ERA5-derived RMM1/RMM2/phase/amplitude index used (the version matters) |
| `event_table.csv` | One row per China-landfalling storm (490): landfall + LMI intensity, genesis, MJO phase at landfall & genesis, per-storm TCP, coastal/inland split, track length, ocean-residence |
| `figure_ready_phase_summary.csv` | One row per phase-group × intensity: n, total/mean/median TCP |
| `summary_Table2_counts_TCP_by_intensity.csv` | Table 2 (counts + TCP by intensity, coastal/inland) |
| `summary_Table3_LMI_vs_landfall_crosstab.csv` | Table 3 (LMI × intensity-at-landfall cross-tab) |
| `summary_TableS1_phase_intensity_sample_sizes.csv` | Table S1 (n + NDAYS + CIs) |
| `MANIFEST.json` | Machine-readable metadata (this file's content in JSON) |

External gridded inputs (not archived here — see `download-url.txt`): CMA Best
Track (`data/CMABSTdata`), CHM_PRE_V2 daily precipitation (TPDC, 0.1°,
1960–2024), NCEP/NCAR daily reanalysis (NOAA PSL), IBTrACS.ALL v04r01.

## Variable definitions & units
- **tcp_total** — event-total TCP over China land cells, area-weighted, **10⁶ m³**
- **tcp_depth** — area-weighted mean event depth over wet China cells, **mm**
- **affected_area** — wet China cell area (prec > 0.5 mm), **km²**
- **coastal_total / inland_total** — TCP over cells ≤200 km / >200 km from the
  coastline (china_country.shp); sensitivity at 100/300 km in Fig S9
- **NDAYS_group** — active-MJO days (amplitude ≥ 1) in Jun–Oct with phase in group
- **rate_per_1000_phase_days** — storm count / NDAYS_group × 1000
- **ratio_no_modulation** — rate ÷ (total events / total active days); 1.0 = null
- **vort_850** — 850-hPa relative vorticity ∂v/∂x − ∂u/∂y, **s⁻¹**
- **z_anomaly** — (X − X̄_month) / σ_month, calendar-month climatology 1960–2024

Intensity (LMI / at-landfall): Weak ≤32.6, Moderate <51.0, Major ≥51.0 m/s.
MJO: amplitude ≥ 1 active; 8 RMM phases merged into 4 pairs (1-2, 3-4, 5-6, 7-8).

## Significance
Two-sided α = 0.10. Phase comparisons use a residence-time permutation null
(reassign each event's phase from the active-day multinomial, 1000×, 5–95th pct
CI). Field composites use one-sample t = composite·√n, |t| > 1.645, on
calendar-month anomalies.

## Software environment
Python 3.12.12, pandas 3.0.3, xarray, numpy, scipy, geopandas, shapely, cartopy,
matplotlib. Platform: Linux x86_64 (glibc 2.34). (Exact versions captured in
`MANIFEST.json` at export time.)
