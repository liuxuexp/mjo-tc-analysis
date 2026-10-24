# CHM_PRE_V2: daily precipitation (external, not committed)

The gridded files are about 40 GB for 1960-2024 and are not stored in this
repository. They are needed only to rebuild `data/pre/`, the per-storm
precipitation fields, which in turn feed `00_build_event_table.py`,
`04_fig04_tcp_intensity.py` and `figS4_coastal_sensitivity.py`.

Dataset: China Merged Precipitation Analysis V2.0, daily, 0.1 degree,
gauge-satellite merged.

Download: see `download-url.txt` (TPDC). Place the yearly files as

```
data/CHM_PRE_V2/daily/CHM_PRE_V2_daily_{year}.nc     # year = 1960 ... 2024
```

Grid, which must match `config.py`: 18.05-53.95 N, 72.05-135.95 E, 0.1 degree;
variable `prec` in mm/day.

If `data/pre/` is already present, this directory is not needed.
