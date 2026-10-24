# shapefiles: China boundary geometries (committed)

Required input for the land mask, the coast distance raster and all maps.

| File | Role |
|---|---|
| `china_country.shp` | dissolved China land polygon, single geometry, EPSG:4326; the China-land mask for TCP integration and the coastline for the coastal/inland split |
| `china.shp` | detailed China boundary, drawn by `fig02_landfall_maps.py`, `04_fig04_tcp_intensity.py`, `06_fig06_genesis_track.py` and `10r_fig09_circulation.py` |
| `china_nine_dotted_line.shp` | nine-dash line, drawn on every map |

Each shapefile keeps its `.dbf`, `.prj` and `.shx` sidecars.

Provenance: standard China administrative boundary geometries, used only as
geographic reference for land masks and map outlines. They carry no scientific
claim; replace them with an authorised boundary source if your environment
requires it. All analyses depend only on these three geometries.
