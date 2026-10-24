# IBTrACS — alternative best-track (external, not committed)

**Status:** placeholder — fetch this file before running `11_ibtracs_robustness.py`
(Figure S3).

**Dataset:** International Best Track Archive for Climate Stewardship (IBTrACS),
`IBTrACS.ALL` version 04r01 (all-agency merged).

**Download (NOAA NCEI):** https://www.ncei.noaa.gov/products/international-best-track-archive

Place the file here as:

```
data/IBTrACS/IBTrACS.ALL.v04r01.nc
```

**Cite:** Knapp, K. R., Kruk, M. C., Levinson, D. H., Diamond, H. J., & Neumann, C. J.
(2010). The International Best Track Archive for Climate Stewardship (IBTrACS):
unifying tropical cyclone data. Bull. Amer. Meteor. Soc., 91(3), 363–376.
https://doi.org/10.1175/2009BAMS2755.1

**What this is used for:** robustness check on the intensity classification. CMA
China-landfall storms are matched to IBTrACS by genesis proximity (first-track point
within 300 km and ±24 h); ~47 % match (USA-agency subset; weak/short-lived South
China Sea storms are largely absent from that subset). Wind-averaging differences
between agencies are documented in the Methods. TCP itself is precipitation-based and
storm identity is shared, so the robustness test targets the track/intensity
classification — the dataset that actually changes.
