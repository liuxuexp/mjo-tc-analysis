# Data-NCAR — NCEP/NCAR daily reanalysis (external, not committed)

**Status:** placeholder — the gridded NetCDF files are large and **not** stored here.
Fetch them before running `09_seasonal_composites.py` (and therefore
`10_fig09_dynamics.py`, the Fig 9 + S5–S8 circulation composites).

**Dataset:** NCEP/NCAR Reanalysis 1, daily means.

**Download:** see `download-url.txt` (NOAA PSL). Place the yearly files here as

```
data/Data-NCAR/daily/{var}.{year}.nc     # var ∈ {uwnd, vwnd, hgt, slp}; year = 1960 … 2024
```

**Variables used:** `uwnd` (200/500/850 hPa), `vwnd` (500/850 hPa), `hgt` (500 hPa),
`slp`. The 850-hPa relative vorticity is computed in code as the curl of `u,v` at
850 hPa (cosine-latitude weighted); it is not read directly.

**What this is used for:** calendar-month circulation composites by MJO phase group
(Fig 9 main; Fig S5–S8). Each daily field is standardised by its per-calendar-month
climatology (1960–2024) before compositing. Significance uses a one-sample
`t = composite·√n`, `|t| > 1.645` (two-sided α = 0.10).
