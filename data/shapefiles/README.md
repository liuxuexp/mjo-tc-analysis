# shapefiles/ — China boundary geometries (committed)

**Status:** committed (required input for masks, maps, and the coastal-distance raster).

**Files used by the code** (see `config.py`):

| File | Role |
|------|------|
| `china_country.shp` | dissolved China land polygon (single geometry, EPSG:4326) — the **China-land mask** for TCP integration and the coastline used for the coastal/inland (≤200 km) split |
| `china.shp` | detailed China boundary — drawn on the landfall/genesis maps (`fig02`, `fig04`, `10_fig09_dynamics`) |
| `china_nine_dotted_line.shp` | nine/ten-dash line — drawn on every map for geographic completeness |

The remaining files (`.dbf`/`.prj`/`.shx`, simplified variants, basin files, copies)
are the supporting sidecar geometries; keep them so the shapefiles read correctly.

**Provenance:** standard China administrative-boundary geometries, used here purely as
geographic reference (land masks and map coastlines). They carry no scientific claim;
replace them with your own authorised boundary source if your environment requires it —
all analyses depend only on `china_country.shp` (land mask / coastline) and the two
drawn boundaries, not on any attribute data.
