# Vansar parcel base layer

Eight parcels **digitised by hand** from open drone imagery, in real
coordinates. This is the georeferenced base the 3D ULPIN work needed: a
parcel with known ground, so a volume can stand on something real.

Rebuild with `python3 tools/digitise_vansar.py` (needs the imagery:
`python3 tools/fetch_drone_sample.py`).

| | |
| --- | --- |
| Source imagery | OpenAerialMap, "Vansar", UAV, 2024-06-07, **2.2 cm/px** |
| Imagery credit | **Imagery © Hari Prasad K, via OpenAerialMap, CC-BY 4.0** |
| Location | 72.7134–72.7143 E, 22.7178–22.7186 N (Gujarat) |
| CRS | **EPSG:4326** (WGS 84), read from the GeoTIFF itself |
| Parcels | 8 — 6 agricultural, 1 industrial yard, 1 water body |
| Digitised area | 0.35 ha of a 1.43 ha scene |

| File | What it is |
| --- | --- |
| `vansar_parcels.geojson` | The layer. Attributes: `parcel_id`, `land_use`, `area_m2` |
| `vansar_parcels.shp` (+ `.dbf`, `.shx`, `.prj`) | ESRI Shapefile, written by `ogr2ogr` |
| `vansar_parcels.city.json` | CityJSON 2.0, loadable by the import endpoint |

## How it was made, and what that means

The boundaries were traced **by eye** against the orthomosaic — the field
bunds, the yard edge and the pond are all plainly visible at 2.2 cm. Vertices
were placed in pixel coordinates, transformed with the GeoTIFF's own affine
transform (no control points, no fitting, so no error added beyond the
imagery's own georeferencing), then clipped to the orthomosaic's data extent
and checked with shapely: **no overlapping parcels, no invalid geometry.**

So the coordinates are as good as the drone survey. The *boundaries* are one
reader's interpretation of what is visible from above.

**This is not a legal cadastre.** It is not a survey, it carries no title, no
khasra number and no owner, and nothing here was checked against a record. It
is a base layer for development and demonstration: real ground, real
coordinates, honestly drawn.

## Why it exists

Automatic extraction was tried first and measured. `cadastral.vectorize()`
works on drawn cadastral sheets — 14 parcels from the synthetic village map,
15 from the real Bhu-Naksha sheet — and **fails on photography**: run against
this orthomosaic it returned 4 polygons, every one of them the image's own
data boundary, and not a single field bund.

That is the expected result rather than a defect. The module traces inked
lines, and an orthomosaic has none — only a texture change along a bund. Doing
it automatically needs a segmentation model (HiSup or similar), which needs a
GPU. Doing it by hand takes twenty minutes and is how cadastral mapping has
always worked.
