# Data-NCAR: NCEP/NCAR daily reanalysis (external, not committed)

The gridded files are large and are not stored here. They are needed only for
`09r_composite_jjaso.py` and `10r_fig09_circulation.py` (Figure 9).

Dataset: NCEP/NCAR Reanalysis 1, daily means.

Download: see `download-url.txt` (NOAA PSL). Place the yearly files as

```
data/Data-NCAR/daily/{var}.{year}.nc     # var in {uwnd, vwnd, slp}; year = 1960 ... 2024
```

Variables used: `uwnd` and `vwnd` at 850 hPa, and `slp`. Relative vorticity at
850 hPa is computed in `lib/dynamics.py` as the curl of u and v, cosine-latitude
weighted, and is not read directly.

Each daily field is standardised by its calendar-month climatology over
1960-2024 before compositing. Significance uses a grid-point year-block
permutation test with field-wise Benjamini-Hochberg FDR; q < 0.05.
