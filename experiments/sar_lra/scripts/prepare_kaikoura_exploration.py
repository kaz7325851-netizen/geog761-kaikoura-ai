"""Freeze a geometry-selected pilot and prepare real inventory references, not predictions."""
from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone
import pyproj
import shapefile
import rasterio
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape, mapping, box
from shapely.ops import transform, unary_union, nearest_points

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'kaikoura_exploration'
OUT.mkdir(exist_ok=True)
forward = pyproj.Transformer.from_crs(4326, 2193, always_xy=True).transform
reverse = pyproj.Transformer.from_crs(2193, 4326, always_xy=True).transform
roads_path = ROOT.parent / 'sentinel1/study_route_lines.geojson'
road_features = json.loads(roads_path.read_text(encoding='utf-8'))['features']
roads = unary_union([transform(forward, shape(x['geometry'])) for x in road_features])
sh1 = unary_union([transform(forward, shape(x['geometry'])) for x in road_features if x['properties']['route'].startswith('SH1')])
corridor = roads.buffer(2000)
center = nearest_points(sh1, corridor.centroid)[0]
# No imagery, inventory density, or prediction is consulted to select the pilot.
west = int(center.x // 10) * 10 - 3200
north = int(center.y // 10) * 10 + 3200
pilot = box(west, north-6400, west+6400, north)
pilot_wgs84 = transform(reverse, pilot)
route_buffer = corridor.intersection(pilot)
config = {
    'prepared_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'Prepared locally; Earth Engine execution and raster download pending',
    'event_utc': '2016-11-13T11:02:56Z', 'event': 'Kaikoura 2016',
    'selection_rule': '6400m square centered on nearest SH1 point to centroid of old 2km road corridor; snapped to 10m NZTM grid, fixed before predictions or label inspection',
    'final_team_aoi': False, 'pilot_area_km2': 40.96, 'bounds_wgs84': list(pilot_wgs84.bounds),
    'crs': 'EPSG:2193', 'crs_transform': [10, 0, west, 0, -10, north], 'dimensions': [640, 640],
    'pass': 'ASCENDING', 'relative_orbit': 52, 'platform': 'Sentinel-1A',
    'pre': ['2016-08-23','2016-08-24'], 'post': ['2016-11-15','2016-11-16'],
    'recent_vv': ['2016-09-14T11:02:56Z','2016-11-13T11:02:56Z'], 'end_exclusive': True,
    'pre_age_days_at_event': 82.15320601851852, 'post_acquisition_delay_hours': 44.321666666666665,
    'model_band_order': ['postVV','postVH','diffVV','diffVH'],
    'export_band_order': ['postVV','postVH','diffVV','diffVH','valid','preVV','preVH','recentPreVV','baselineVVShift','recentVVValid'],
    'nodata': -9999, 'input_normalisation': 'None; GEE dB values and dB differences',
    'experimental_changes': ['82-day-old single-date dual-polarization baseline rather than original pre60 stack', 'Single post date retains both bands from identical dates/platform', 'Fixed exact 10m NZTM output grid'],
    'limitations': ['Not the original 60/12 deployment', 'Single-date baseline has no multi-date speckle/weather averaging', 'Recent VV comparison cannot prove VH stability', 'Acquisition masks are not SAR shadow/layover masks', 'Window classification is not road passability', 'Road geometry is current OSM-derived, not independently audited 2016 historic roads'],
    'road_source_sha256': hashlib.sha256(roads_path.read_bytes()).hexdigest(),
    'inference_performed': False, 'training_performed': False
}
(OUT / 'experiment_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
for name, geom in [('pilot_aoi.geojson', pilot_wgs84), ('pilot_road_corridor.geojson', transform(reverse,route_buffer))]:
    (OUT / name).write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':{'final_team_aoi':False},'geometry':mapping(geom)}]}), encoding='utf-8')

# Reuse already downloaded and checksum-verified author archive.
archive_path = ROOT / 'data/landslide_inventories_polygons.zip'
source = json.loads((ROOT / 'inventory_source.json').read_text(encoding='utf-8'))
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == source['sha256']
ref = OUT / 'reference'
ref.mkdir(exist_ok=True)
with zipfile.ZipFile(archive_path) as archive:
    names = [x for x in archive.namelist() if x.startswith('Kaikora_2016.')]
    assert names and any(x.endswith('.prj') for x in names)
    for name in names:
        dest = (ref / name).resolve()
        assert dest.is_relative_to(ref.resolve())
        dest.write_bytes(archive.read(name))
source_crs = pyproj.CRS.from_wkt((ref / 'Kaikora_2016.prj').read_text())
convert = pyproj.Transformer.from_crs(source_crs, 2193, always_xy=True).transform
reader = shapefile.Reader(str(ref / 'Kaikora_2016.shp'))
features = []
repairs = 0
for index, shp in enumerate(reader.shapes()):
    geom = shape(shp.__geo_interface__)
    if not geom.is_valid:
        geom = geom.buffer(0)
        repairs += 1
    geom = transform(convert, geom)
    if geom.intersects(pilot):
        geom = geom.intersection(pilot)
        if not geom.is_empty and geom.area > 0:
            features.append({'type':'Feature','properties':{'source_index':index},'geometry':mapping(transform(reverse,geom))})
(ref / 'pilot_inventory.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}), encoding='utf-8')
checks = {'source': 'Author Zenodo 15159492 Kaikora_2016.shp (spelling retained)', 'source_crs':source_crs.to_string(),
          'total_source_features':len(reader), 'geometry_repairs_total':repairs, 'pilot_intersecting_features':len(features),
          'source_shapefile_sha256':hashlib.sha256((ref/'Kaikora_2016.shp').read_bytes()).hexdigest(),
          'caveats':['Raw optical-derived inventory, no pass-specific shadow/layover cleaning', 'Not actual road closure labels', 'Exact checkpoint training/test independence not independently audited'],
          'rasterization_performed':True, 'inference_performed':False}
affine = Affine(10, 0, west, 0, -10, north)
projected = [transform(forward, shape(x['geometry'])) for x in features]
label = rasterize([(g,1) for g in projected], out_shape=(640,640), transform=affine, fill=0, dtype='uint8', all_touched=False)
road_mask = rasterize([(route_buffer,1)], out_shape=(640,640), transform=affine, fill=0, dtype='uint8', all_touched=False)
for filename, pixels in [('reference_mask.tif',label),('road_corridor_mask.tif',road_mask)]:
    with rasterio.open(ref/filename, 'w', driver='GTiff', width=640, height=640, count=1,
                       crs='EPSG:2193', transform=affine, dtype='uint8', compress='deflate') as dst:
        dst.write(pixels,1)
        dst.update_tags(prediction='false', rule='Pixel centres, all_touched=False; acquisition validity not yet applied')
    with rasterio.open(ref/filename) as src:
        assert src.transform == affine and src.crs.to_epsg() == 2193 and np.array_equal(src.read(1),pixels)
checks.update(reference_pixels=int(label.sum()), reference_fraction=float(label.mean()), road_corridor_pixels=int(road_mask.sum()),
              shape=[640,640], all_touched=False, independent_raster_reread_passed=True)
(ref/'reference_preparation.json').write_text(json.dumps(checks,indent=2), encoding='utf-8')
print(json.dumps({'config':config,'reference':checks}, ensure_ascii=False, indent=2), flush=True)
