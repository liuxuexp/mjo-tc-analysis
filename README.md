# MJO modulation of tropical-cyclone landfalls and intense rainfall over China

Analysis code for *Role of the Madden-Julian Oscillation in modulating tropical
cyclone landfalls and intense rainfall over China* (HYDROL-S-26-03282).

The code reproduces Figures 1-9 and S1-S4 and Tables 2, 3, S1 and S2 from the
committed event tables and reference outputs.

## Layout

```
.
├── config.py                     paths and shared constants
├── lib/                          shared helpers (MJO phases, TCP, statistics, plotting)
├── 00_build_event_table.py       event table (requires data/pre)
├── r2_inference.py               year-block CI / permutation p / BH FDR
├── 01_tables.py                  Tables 2, 3, S1, S2
├── 02_fig01.py ... 10r_fig09*.py main figures 1-9
├── 11_ibtracs_robustness.py      Figure S3
├── fig02_landfall_maps.py        Figure 2
├── figS1_monthly.py              Figure S1
├── figS2_landfall_intensity.py   Figure S2
├── figS4_coastal_sensitivity.py  Figure S4
├── make_fig_doc_data.py          plotting values for all figures
├── sensitivity_daylevel.py       day-level sensitivity check
├── make_numbers_report.py        numerical report
├── tests_stats_yb.py             unit tests for the inference engine
├── run_all.py                    pipeline runner
├── data/                         inputs and reference outputs
└── fig/                          generated figures (not committed)
```

## Data

Committed inputs:

| Path | Content |
|---|---|
| `data/typhoon_output/` | China landfall and track tables derived from CMA Best Track |
| `data/ERA5_MJO_1950-2024.csv` | RMM1, RMM2, phase, amplitude used for MJO grouping |
| `data/shapefiles/` | China boundary, coastline and nine-dash line |

Committed reference outputs:

| Path | Content |
|---|---|
| `data/data03/event_table.csv` | one row per China-landfalling storm |
| `data/data03/r2_stats_pergrp.csv` | per-group estimates, CIs and q values |
| `data/data03/r2_stats_contrast.csv` | pre-specified phases 5-6 vs 1-2 contrasts |
| `data/data03/fig_doc_data.json` | plotting values for every figure |
| `data/data03/ibtracs_matched.csv` | IBTrACS match used for Figure S3 |
| `data/tables03/` | Tables 2, 3, S1 and S2 (CSV and Markdown) |

External inputs are not committed. Each directory contains a `README.md` and,
where applicable, `download-url.txt`:

| Directory | Dataset |
|---|---|
| `data/CHM_PRE_V2/` | CHM_PRE_V2 daily precipitation, 0.1 deg, 1960-2024 (TPDC) |
| `data/pre/` | per-storm precipitation fields derived from CHM_PRE_V2 |
| `data/Data-NCAR/` | NCEP/NCAR Reanalysis 1 daily fields, 1960-2024 (NOAA PSL) |
| `data/IBTrACS/` | IBTrACS.ALL v04r01 (NOAA NCEI) |
| `data/CMABSTdata/` | CMA Best Track provenance |

## Environment

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 was used for the published results. Cartopy needs its own coastline
data on first use; the maps in this repository draw the committed shapefiles.

## Run

```
python run_all.py --check     # list which optional inputs are present
python run_all.py             # run every step whose inputs are available
```

If `data/pre/` is absent, the pipeline uses the committed
`data/data03/event_table.csv` and skips the four steps that read per-storm
precipitation fields (00, 04, figS4, make_fig_doc_data). If `data/Data-NCAR/`
is absent, the circulation composites and Figure 9 are skipped. If IBTrACS is
absent, Figure S3 is skipped. Logs are written to `logs/`.

Individual steps can be run directly, for example:

```
python 01_tables.py
python 02_fig01.py
```

## Statistical treatment

Confidence intervals use a year-block bootstrap: the 65 complete June-October
seasons (1960-2024) are resampled with replacement, 10,000 times, and the 95%
interval is reported. Significance uses a year-block permutation test: complete
seasonal MJO sequences are reassigned among years while TC records stay in
their original years, giving two-sided empirical p values from 10,000
permutations. Benjamini-Hochberg FDR is applied within pre-specified test
families; q < 0.05 is the significance criterion. `lib/stats_yb.py` implements
both procedures, and `tests_stats_yb.py` checks their invariants.

## Outputs

- `fig/`: Figures 1-9 and S1-S4 as PNG.
- `data/tables03/`: Tables 2, 3, S1 and S2 as CSV and Markdown.
- `data/data03/`: event table, inference tables, figure plotting values.
- `reports/`: numerical summary and day-level sensitivity check.

## Scope and limitations

- The analysis covers June-October 1960-2024. Genesis-based statistics use the
  genesis-day MJO phase; landfall-based statistics use the landfall-day phase.
- The circulation composites use NCEP/NCAR Reanalysis 1 at 2.5 deg. ERA5 was
  not used because a complete local record was unavailable.
- The IBTrACS robustness check matches about 47% of CMA landfalling storms; the
  unmatched cases are mainly weak or short-lived South China Sea systems.
- MJO phases are grouped as 1-2, 3-4, 5-6 and 7-8, with amplitude >= 1 defining
  active days.

## Licence

Code: MIT (`LICENSE`). Datasets follow their providers' terms.
