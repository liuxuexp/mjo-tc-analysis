# typhoon_output/ — CMA-Best-Track-derived landfall & track tables (committed)

**Status:** committed. These are the primary track inputs consumed directly by this
repository (CMA Best Track → landfall detection → these CSVs).

**Source dataset:** CMA/STI Best Track (see [`../CMABSTdata/README.md`](../CMABSTdata/README.md)
for provenance and download).

**Derivation:** the upstream landfall processor (`01_landfall.py`) reads CMA Best
Track, detects each track segment's precise geometric intersection with the China
boundary (mainland + Taiwan + Hainan, selected via reference points with an area-sort
fallback — not the nearest track vertex), and writes:

| File | Content |
|------|---------|
| `landfall_typhoons_info.csv` | one row per China-**landfalling** storm: id, landfall time/lat/lon/wind/pressure, LMI (`max_wind`), `phase_group` (MJO at landfall), etc. (490 storms, 1960–2024) |
| `landfall_typhoons_tracks.csv` | 6-hourly track points for the landfalling storms |
| `all_typhoons_tracks.csv` | 6-hourly track points for **all** WNP TCs (used for ACE / all-WNP genesis) |
| `non_landfall_typhoons_tracks.csv` | tracks for WNP TCs that did **not** make China landfall |
| `landfall_typhoons_with_region.csv` | landfalling storms + `region` (South/East/North China) + `formation_region` (genesis region, from the literal first track point) |

**Wind categories** (`get_wind_category`, mirrored in `config.py`): 0 = TD, 1 = Weak
(≤32.6 m/s), 2 = Moderate (<51.0), 3 = Major (≥51.0). A storm is skipped entirely if
its `chinese_code == '0000'` or its max wind category is 0.

**Period:** 1960–2024 (analyses restrict to Jun–Oct + active MJO at landfall; see
`config.py` and the README).
