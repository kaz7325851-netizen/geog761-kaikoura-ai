# Obtaining inputs without committing data

## Sentinel-1 VV

Open https://code.earthengine.google.com/ with your own authorised Earth Engine
account/project. Paste `gee/export_vv.js`, review the study grids and run it.
The script filters `COPERNICUS/S1_GRD`, IW, Sentinel-1A, ascending orbit 52, VV,
at the specified UTC dates. It uses SRTM terrain and retains geometry/validity bands.

The console prints metadata and one temporary download URL per region. Download
each GeoTIFF and name it exactly `<region_id>.tif` from `configs/vv_design.json`.
Those are 17-band data files, not screenshots of the map. Links may expire; rerun
the script instead of publishing old URLs. Do not commit the links or credentials.
The packaged GEE script was taken from the successful historical export; it has
not been rerun against a fresh account during packaging.

The Python CLI checks CRS, shape, affine grid, band names when declared, dVV units,
masks and numerical output. Matching numbers does not certify physical alignment.

## Official references

GNS V3 publication: https://doi.org/10.17603/ds2-rpyg-0h84

Use the source and debris shapefiles together with all required sidecars and
metadata. The historical download stages are in `experiments/rf_vv/scripts/`.
The public download flow uses its own anonymous session; do not insert account
cookies or private tokens. These stages expect their own generated listing files.

V2 source centroids were a diagnostic source, not slide extents:
https://gis.gns.cri.nz/server/rest/services/EILD/EILD_V2_Map/MapServer/0

Raw official vector files are not redistributed here. Review metadata/terms and
retain source attribution. The proposal configuration stores source/debris IDs
and review groups, not full polygon boundaries.

## Historical model inputs

BBUnet needs two 12-band Sentinel-2 L2A images plus four terrain channels and
cloud screening. CNN M_ALL_006 uses two RGB inputs and slope/hillshade; its exact
RGB construction is unresolved. SAR-LRA's tested weights need VV+VH with the
matching author architecture. None can accept arbitrary map screenshots.

These assets and author checkpoints must be obtained separately from the source
pages; see `docs/sources.md` and `docs/environments.md`. Their dated working trees
and all derived rasters are excluded from this repository.
