# CHM_PRE_V2 — daily precipitation (external, not committed)

**Status:** placeholder — the gridded NetCDF files are large (~40 GB for 1960–2024)
and are **not** stored in this repository. Fetch them before running the steps that
re-derive per-storm TCP (`00_build_event_table.py` with a fresh `pre/`, plus
`04_fig04`, `06_fig06`, `figS9_coastal_sensitivity`).

**Dataset:** China Merged Precipitation Analysis V2.0 (CHM_PRE_V2) — daily, 0.1°,
gauge–satellite merged.

**Download:** see `download-url.txt` (TPDC). Place the yearly files here as

```
data/CHM_PRE_V2/daily/CHM_PRE_V2_daily_{year}.nc     # year = 1960 … 2024
```

**Grid (must match `config.py`):** 18.05–53.95 °N, 72.05–135.95 °E, 0.1°; variable
`prec` in mm/day.

**What this is used for:** building the per-storm tropical-cyclone precipitation
(TCP) fields in `../pre/pre_{code}.nc`, which are then integrated over China land
cells (China-land TCP, coastal/inland split). If you already have `../pre/`, you do
not need this directory.
