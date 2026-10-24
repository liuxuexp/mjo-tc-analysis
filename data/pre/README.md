# pre/: per-storm precipitation fields (intermediate, not committed)

These files are about 18 MB each, roughly 8.8 GB for 490 storms, and are not
stored in this repository. They are derived from `../CHM_PRE_V2/` and are used
by three steps:

| Step | Use |
|---|---|
| `00_build_event_table.py` | integrates each storm field over China land to give `tcp_total`, `tcp_depth`, `affected_area` and the coastal/inland split |
| `04_fig04_tcp_intensity.py` | accumulated and mean TCP maps by intensity (Figure 4a-f) |
| `figS4_coastal_sensitivity.py` | coastal/inland TCP at 100, 200 and 300 km |

The committed `data/data03/event_table.csv` already carries the TCP columns, so
`01_tables.py`, `02_fig01.py`, `03_fig03_regional.py`, `05_fig05_phase_decomp.py`,
`06_fig06_genesis_track.py`, `07_fig07_genesis_mjo.py`, `08_fig08_genesis_landfall.py`
and `figS1`/`figS2`/`fig02` run without this directory.

## File format

```
pre/pre_{code}.nc      # one file per storm, {code} = zero-padded 4-digit CMA chinese_code
```

Each file holds daily precipitation in mm/day on the CHM_PRE_V2 0.1 degree grid,
accumulated over the storm's track days and retained only inside a 500-km buffer
of the track; cells outside the buffer are NaN. The variable is `prec`.

## Regenerating

The fields are built by summing CHM_PRE_V2 daily precipitation along each CMA
Best Track for the storm's life days, masked to the 500-km track buffer. See
`lib/tcp.py` for the China land mask, coastline distance raster and area
weighting applied downstream.

With `../CHM_PRE_V2/daily/` and `../typhoon_output/` in place, write one file per
storm in `typhoon_output/landfall_typhoons_info.csv`, then run
`python 00_build_event_table.py` to rebuild `data/data03/event_table.csv`.
