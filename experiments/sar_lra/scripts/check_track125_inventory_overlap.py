"""Inspect real reference overlap, not predictions, before proposing a new pilot."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping, box
from shapely.ops import transform, unary_union
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'kaikoura_full_coverage/20261005T192444Z'
ref = ROOT / 'kaikoura_exploration/reference/Kaikora_2016.shp'
source_crs = CRS.from_wkt(ref.with_suffix('.prj').read_text())
to_metric = Transformer.from_crs(4326, 2193, always_xy=True).transform
to_geo = Transformer.from_crs(2193, 4326, always_xy=True).transform
inventory_to_metric = Transformer.from_crs(source_crs, 2193, always_xy=True).transform
features = json.loads((OUT / 'ASF_raw.json').read_text())['features']
pre = [f for f in features if f['properties']['pathNumber'] == 125 and f['properties']['startTime'].startswith('2016-09-21')]
post = [f for f in features if f['properties']['pathNumber'] == 125 and f['properties']['startTime'].startswith('2016-11-20')]
assert len(pre) == len(post) == 1
pre_footprint = transform(to_metric, shape(pre[0]['geometry']))
post_footprint = transform(to_metric, shape(post[0]['geometry']))
overlap = pre_footprint.intersection(post_footprint)
inside = []
for index, shp in enumerate(shapefile.Reader(str(ref)).iterShapes()):
    geom = shape(shp.__geo_interface__)
    if not geom.is_valid:
        geom = geom.buffer(0)
    geom = transform(inventory_to_metric, geom)
    if geom.intersects(overlap):
        clipped = geom.intersection(overlap)
        if not clipped.is_empty and clipped.area > 0:
            inside.append((index, geom, clipped))
roads_path = ROOT.parent / 'sentinel1/study_route_lines.geojson'
roads = unary_union([transform(to_metric, shape(f['geometry'])) for f in json.loads(roads_path.read_text())['features']])
summary = {'created_utc': datetime.now(timezone.utc).isoformat(),
    'type': 'Catalogue footprint and raw optical inventory check, no raster validation or prediction',
    'pre_scene': pre[0]['properties']['sceneName'], 'post_scene': post[0]['properties']['sceneName'],
    'pass': 'ASCENDING', 'relative_orbit': 125, 'polarization': 'VV+VH',
    'inventory_polygons_intersecting_common_footprint': len(inside),
    'clipped_inventory_area_km2': sum(g.area for _, _, g in inside) / 1e6,
    'polygons_within_2km_old_project_roads': sum(g.distance(roads) <= 2000 for _, g, _ in inside),
    'common_footprint_intersects_old_road_corridor': overlap.intersects(roads.buffer(2000)),
    'limitations': ['Footprints are not usable pixel masks or SAR geometry visibility',
        'Raw inventory polygons are not road closure labels',
        'GEE ingestion not checked', 'No predictions or score-based area selection']}
candidates = []
for index, geom, clipped in sorted(inside, key=lambda item: item[2].area, reverse=True):
    center = clipped.representative_point()
    x = int(center.x // 10) * 10
    y = int(center.y // 10) * 10
    pilot = box(x - 3200, y - 3200, x + 3200, y + 3200)
    fraction = pilot.intersection(overlap).area / pilot.area
    if fraction >= 0.999:
        local = [(i, g.intersection(pilot)) for i, g, _ in inside if g.intersects(pilot)]
        local = [(i, g) for i, g in local if not g.is_empty and g.area > 0]
        candidates.append({'source_index': index, 'selection_rule': 'Largest inventory polygon with 6.4km square fully inside both catalogue footprints; no prediction inspected',
            'center_wgs84': list(transform(to_geo, center).coords)[0],
            'bounds_wgs84': list(transform(to_geo, pilot).bounds),
            'bounds_nztm': list(pilot.bounds), 'footprint_fraction': fraction,
            'largest_source_clipped_area_km2': clipped.area / 1e6,
            'pilot_inventory_polygons': len(local), 'pilot_raw_inventory_area_km2': sum(g.area for _, g in local) / 1e6,
            'distance_to_old_roads_km': center.distance(roads) / 1000,
            'area_km2': pilot.area / 1e6,
            'geometry': mapping(transform(to_geo, pilot))})
        if len(candidates) == 3:
            break
summary['largest_eligible_pilots'] = candidates
windows = []
for index, geom, clipped in sorted(inside, key=lambda item: item[2].area, reverse=True):
    center = clipped.representative_point()
    # Check a 640m model window, not just a 6.4km export tile. Inspect several
    # offsets fixed around the labelled polygon, never model scores.
    for dx in [-160, 0, 160]:
        for dy in [-160, 0, 160]:
            x = int(center.x // 10) * 10 + dx
            y = int(center.y // 10) * 10 + dy
            window = box(x - 320, y - 320, x + 320, y + 320)
            if window.intersection(overlap).area / window.area < 0.999:
                continue
            local_geoms = [g for _, g, _ in inside if g.intersects(window)]
            pixels = rasterize([(g, 1) for g in local_geoms], out_shape=(64, 64),
                transform=from_origin(x-320, y+320, 10, 10), fill=0, dtype='uint8', all_touched=False)
            windows.append({'source_index': index, 'center_wgs84': list(transform(to_geo, window.centroid).coords)[0],
                'bounds_nztm': list(window.bounds), 'raw_inventory_pixels': int(pixels.sum()),
                'raw_inventory_fraction': float(pixels.mean()), 'footprint_fraction': 1.0})
windows.sort(key=lambda w: w['raw_inventory_pixels'], reverse=True)
summary['model_window_check'] = {'window_m': 640, 'offsets_m': [-160, 0, 160],
    'checked_windows_not_independent_events': len(windows),
    'raw_positive_windows_gt_5pct': sum(w['raw_inventory_fraction'] > 0.05 for w in windows),
    'best_raw_window': windows[0] if windows else None,
    'geometry_cleaned_or_actual_valid_pixels': False,
    'selection_is_inventory_guided_not_random_or_prediction_guided': True}
(OUT / 'track125_raw_reference_intersections.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': [
    {'type': 'Feature', 'properties': {'source_index': i, 'raw_inventory': True}, 'geometry': mapping(transform(to_geo, clipped))}
    for i, _, clipped in inside]}), encoding='utf-8')
(OUT / 'track125_overlap.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
(OUT / 'track125_common_footprint.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': [
    {'type': 'Feature', 'properties': {'track': 125, 'valid_pixels_verified': False}, 'geometry': mapping(transform(to_geo, overlap))}]}), encoding='utf-8')
print(json.dumps(summary, indent=2), flush=True)
