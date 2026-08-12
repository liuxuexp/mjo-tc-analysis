# pre/ — per-storm tropical-cyclone precipitation fields (intermediate, not committed)

**Status:** placeholder. These NetCDF files (~18 MB × 490 storms ≈ 8.8 GB) are **not**
stored in this repository. They are an intermediate product derived from
[`../CHM_PRE_V2/`](../CHM_PRE_V2/) and are consumed by four steps:

| Step | Use |
|------|-----|
| `00_build_event_table.py` | integrates each storm's field over China land → `tcp_total`, `tcp_depth`, `affected_area`, coastal/inland split (writes the TCP columns of `data/data03/event_table.csv`) |
| `04_fig04_tcp_intensity.py` | cumulative + mean TCP maps by intensity (Fig 4 a–f) |
| `06_fig06_stormlevel.py` | mean-TCP-per-storm maps by phase (Fig 6 c–f) |
| `figS9_coastal_sensitivity.py` | coastal/inland TCP at 100/200/300 km (Fig S9) |

> If you only want the **tables and most figures**, you can skip this directory:
> `data/data03/event_table.csv` (with its TCP columns already computed) **is committed**, so
> `01_tables`, `02_fig01`, `03_fig03`, `05_fig05`, `07_fig07`, `08_fig08`,
> `12_all_wnp`, `13_october`, `figS1`, `figS2`, and `fig02` run without it.

## File format

```
pre/pre_{code}.nc      # one file per storm; {code} = zero-padded 4-digit CMA chinese_code
```

Each file holds daily precipitation (mm/day) on the CHM_PRE_V2 0.1° grid
(18.05–53.95 °N, 72.05–135.95 °E), accumulated over the storm's track days, retained
**only inside a 500-km buffer** of the track (ocean + land); cells outside the buffer
are `NaN`. The variable is `prec`.

## How to (re)generate

The fields are built by accumulating CHM_PRE_V2 daily precipitation along each track
(CMA Best Track) for the storm's life days, masked to the 500-km track buffer. The
upstream implementation is the `02_extract_precip.py` → `03_accumulate_precip.py`
pair of the reproduction pipeline (see `lib/tcp.py` for the exact China-land mask,
coastal-distance raster, and area-weighting used downstream on these fields).

After fetching `../CHM_PRE_V2/daily/` and `../typhoon_output/`, regenerate with:

```bash
# upstream accumulation (not included in this src03 repository):
#   for each storm code in typhoon_output/landfall_typhoons_info.csv,
#   sum CHM_PRE_V2 daily precip over its track days within 500 km -> pre/pre_{code}.nc
```

Then `cd ../.. && python 00_build_event_table.py` recomputes `data/data03/event_table.csv`
(including its TCP columns) from these fields.
