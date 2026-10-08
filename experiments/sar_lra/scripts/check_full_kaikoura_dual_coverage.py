"""Read-only public catalogue search over the full local event inventory extent.

Downloads only metadata; does not change the team AOI, request imagery exports,
load models, or train. All outputs remain below testdata.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from collections import defaultdict
import json
import hashlib
import re
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'kaikoura_full_coverage' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True, exist_ok=False)
source = ROOT / 'kaikoura_exploration/reference/Kaikora_2016.shp'
reader = shapefile.Reader(str(source))
crs = CRS.from_wkt(source.with_suffix('.prj').read_text())
project = Transformer.from_crs(crs, 4326, always_xy=True)
bounds = project.transform_bounds(*reader.bbox, densify_pts=21)
query_box = box(*bounds)
EVENT = datetime.fromisoformat('2016-11-13T11:02:56+00:00')
PRE = EVENT - timedelta(days=60)
POST = EVENT + timedelta(days=12)
# Buffered calendar query encompasses exact event-relative windows as well.
start, end = '2016-09-13T00:00:00Z', '2016-11-27T00:00:00Z'
asf_url = 'https://api.daac.asf.alaska.edu/services/search/param?' + urlencode({
    'dataset': 'SENTINEL-1', 'beamMode': 'IW', 'processingLevel': 'GRD_HD,GRD_HS',
    'intersectsWith': query_box.wkt, 'start': start, 'end': end,
    'maxResults': 1000, 'output': 'geojson'})
cdse_filter = ("Collection/Name eq 'SENTINEL-1' and contains(Name,'IW_GRDH') and "
    f"ContentDate/Start ge {start} and ContentDate/Start lt {end} and "
    "OData.CSC.Intersects(area=geography'SRID=4326;" + query_box.wkt + "')")
cdse_url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?' + urlencode({
    '$filter': cdse_filter, '$top': 1000, '$expand': 'Attributes',
    '$orderby': 'ContentDate/Start asc', '$count': 'true'})

def fetch(name, url):
    record = {'source': name, 'url': url}
    try:
        with urlopen(Request(url, headers={'User-Agent': 'GEOG761-catalogue-check/1.0'}), timeout=45) as response:
            body = response.read()
            record['http_status'] = response.status
        data = json.loads(body)
        items = data.get('features', []) if name == 'ASF' else data.get('value', [])
        record.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
            records=len(items), pagination_complete=len(items) < 1000 and not data.get('@odata.nextLink'))
        (OUT / f'{name}_raw.json').write_bytes(body)
        return record, data
    except Exception as exc:
        record.update(error_type=type(exc).__name__, error=str(exc))
        return record, None

manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Full bounding rectangle of downloaded 2016 Kaikoura landslide inventory; NOT final project AOI',
    'bounds_wgs84': bounds, 'inventory_features': len(reader),
    'source_shapefile_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'exact_pre60': [PRE.isoformat(), EVENT.isoformat()],
    'exact_post12': [EVENT.isoformat(), POST.isoformat()],
    'imagery_downloaded': False, 'inference_performed': False, 'requests': []}
with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(lambda arg: fetch(*arg), [('ASF', asf_url), ('CDSE', cdse_url)]))
manifest['requests'] = [r[0] for r in results]
(OUT / 'query_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
if not all(r[0].get('http_status') == 200 and r[0].get('pagination_complete') for r in results):
    print(json.dumps({'directory': str(OUT), 'status': 'Incomplete query; no absence conclusion', 'requests': manifest['requests']}, indent=2), flush=True)
    raise SystemExit(2)

def key(name):
    return '_'.join(name.replace('.SAFE', '').split('_')[:8])

def polarization(value):
    return '+'.join(sorted(set(re.findall(r'VV|VH|HH|HV', value))))

def parse(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))

asf = {key(f['properties']['sceneName']): f for f in results[0][1]['features']}
cdse = defaultdict(list)
for item in results[1][1]['value']:
    cdse[key(item['Name'])].append(item)
common = set(asf) & set(cdse)
for item_key in common:
    props = asf[item_key]['properties']
    for item in cdse[item_key]:
        attributes = {a['Name']: a['Value'] for a in item['Attributes']}
        assert props['flightDirection'] == attributes['orbitDirection']
        assert props['pathNumber'] == attributes['relativeOrbitNumber']
        assert polarization(props['polarization']) == polarization(attributes['polarisationChannels'])
        assert abs((parse(props['startTime']) - parse(item['ContentDate']['Start'])).total_seconds()) < 1

rows = []
for item_key, feature in asf.items():
    p = feature['properties']
    dt = parse(p['startTime'])
    period = 'pre60' if PRE <= dt < EVENT else 'post12' if EVENT <= dt < POST else 'buffer'
    rows.append({'key': item_key, 'time_utc': dt.isoformat(), 'period': period,
        'pass': p['flightDirection'], 'relative_orbit': p['pathNumber'],
        'platform': p['platform'], 'absolute_orbit': p['orbit'],
        'polarization': polarization(p['polarization']), 'cdse_agrees': item_key in common})
rows.sort(key=lambda r: r['time_utc'])
pre_dual = [r for r in rows if r['period'] == 'pre60' and r['polarization'] == 'VH+VV']
post_dual = [r for r in rows if r['period'] == 'post12' and r['polarization'] == 'VH+VV']
tracks = []
for direction, orbit in sorted({(r['pass'], r['relative_orbit']) for r in rows}):
    subset = [r for r in rows if (r['pass'], r['relative_orbit']) == (direction, orbit)]
    track = {'pass': direction, 'relative_orbit': orbit}
    for period in ['pre60', 'post12']:
        selected = [r for r in subset if r['period'] == period]
        track[period + '_slices'] = len(selected)
        track[period + '_dual_slices'] = sum(r['polarization'] == 'VH+VV' for r in selected)
        track[period + '_dates'] = sorted({r['time_utc'][:10] for r in selected})
    tracks.append(track)
exact_keys_agree = set(asf) == set(cdse)
summary = {'status': 'Metadata-only full-inventory-extent check', 'directory': str(OUT),
    'bounds_wgs84': bounds, 'asf_unique_slices': len(asf), 'cdse_product_records': sum(map(len, cdse.values())),
    'cdse_unique_slices': len(cdse), 'catalogue_slice_sets_identical': exact_keys_agree,
    'pre60_dual_slices': len(pre_dual), 'post12_dual_slices': len(post_dual),
    'pre60_dual_records': pre_dual, 'tracks': tracks,
    'same_track_dual_pre_post_exists': any(t['pre60_dual_slices'] and t['post12_dual_slices'] for t in tracks),
    'limitations': ['Inventory extent, not all possible Kaikoura affected terrain',
        'IW GRD products only; footprints are not valid/geometry-clean pixels',
        'Earth Engine ingestion and raster quality not checked',
        'No new raster download, inference, or change to final team research area']}
if exact_keys_agree:
    summary['conclusion'] = ('No pre-event dual-polarization IW GRD scene in exact 60-day window intersects even the full inventory bounding rectangle; moving within this inventory extent cannot supply missing VH.'
        if not pre_dual else 'Pre-event dual scenes exist; inspect footprint, inventory, same-track post coverage and usable pixels before selecting a pilot.')
else:
    summary['conclusion'] = 'Catalogues disagree; reconcile records before concluding absence or selecting an area.'
(OUT / 'slice_metadata.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2), flush=True)
