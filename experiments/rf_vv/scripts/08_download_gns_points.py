"""Read-only official GNS V2 source-centroid cross-check, NOT polygon labels."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parent
LAYER = 'https://gis.gns.cri.nz/server/rest/services/EILD/EILD_V2_Map/MapServer/0'
OUT = ROOT / 'official_reference_points'
OUT.mkdir(exist_ok=True)
design = json.loads((ROOT / 'frozen_data_design.json').read_text(encoding='utf-8'))
bounds = [s['bounds'] for s in design['sites']]
envelope = [min(b[0] for b in bounds)-1000, min(b[1] for b in bounds)-1000,
            max(b[2] for b in bounds)+1000, max(b[3] for b in bounds)+1000]
session = requests.Session()
session.headers['User-Agent'] = 'GEOG761-reference-audit/1.0'
urls = []

def fetch(path, params):
    response = session.get(path, params=params, timeout=40)
    response.raise_for_status()
    data = response.json()
    if 'error' in data:
        raise RuntimeError(data['error'])
    urls.append(response.url)
    return data

metadata = fetch(LAYER, {'f': 'pjson'})
assert metadata['geometryType'] == 'esriGeometryPoint'
ids = fetch(LAYER+'/query', {'f': 'json', 'where': '1=1', 'returnIdsOnly': 'true',
            'geometry': ','.join(map(str, envelope)), 'geometryType': 'esriGeometryEnvelope',
            'inSR': '2193', 'spatialRel': 'esriSpatialRelIntersects'})
object_ids = sorted(set(ids['objectIds'] or []))
features = []
for start in range(0, len(object_ids), 500):
    wanted = object_ids[start:start+500]
    data = fetch(LAYER+'/query', {'f': 'json', 'objectIds': ','.join(map(str, wanted)),
                 'outFields': '*', 'returnGeometry': 'true', 'outSR': '2193'})
    assert not data.get('exceededTransferLimit', False)
    got = data['features']
    assert sorted(f['attributes']['objectid'] for f in got) == wanted
    assert data['spatialReference']['wkid'] == 2193
    features.extend(got)
    print(f'GNS source-centroid records {len(features)}/{len(object_ids)}', flush=True)
assert len(features) == len(object_ids)
payload = {'source': LAYER, 'crs': 'EPSG:2193', 'features': features}
dest = OUT/'GNS_V2_source_centroids_2193.json'
dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT/'layer_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
manifest = {'downloaded_utc': datetime.now(timezone.utc).isoformat(), 'source': LAYER,
            'envelope_2193': envelope, 'count': len(features), 'requests': urls,
            'file': str(dest), 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
            'status': 'official_source_centroids_downloaded_NOT_polygon_ground_truth',
            'copyright': metadata.get('copyrightText'),
            'limitations': ['Centroids are not slide extents or debris trails',
                            'V2 not V3; absence is not verified non-slide',
                            'Cannot establish blockage, observation date, or completeness',
                            'Do not use centroid buffers as positive polygon labels']}
(OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'status': manifest['status'], 'count': len(features), 'file': str(dest)}, ensure_ascii=False))
