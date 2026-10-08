"""Read-only public raster catalogue check for frozen 2016 SH1 pilot."""
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import urlopen,Request
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,time,sys
ROOT=Path(__file__).resolve().parent
DESIGN=ROOT/'20261005T200713Z/frozen_pilot_design.json'
design=json.loads(DESIGN.read_text())
OUT=ROOT/'imagery_checks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True)
bbox=','.join(map(str,design['selected']['bounds_wgs84']))
queries=[('earthsearch_c1','https://earth-search.aws.element84.com/v1/search','sentinel-2-c1-l2a'),
    ('earthsearch_legacy','https://earth-search.aws.element84.com/v1/search','sentinel-2-l2a'),
    ('planetary','https://planetarycomputer.microsoft.com/api/stac/v1/search','sentinel-2-l2a')]
if '--planetary-only' in sys.argv: queries=queries[-1:]
def fetch(q):
    name,base,collection=q
    url=base+'?'+urlencode({'collections':collection,'bbox':bbox,'datetime':'2016-09-01T00:00:00Z/2016-11-24T00:00:00Z','limit':100})
    record={'source':name,'url':url}
    try:
        for attempt in range(3):
            try:
                with urlopen(Request(url,headers={'User-Agent':'GEOG761-pilot-quality/1.0'}),timeout=35) as r:
                    body=r.read();record['http_status']=r.status
                break
            except Exception:
                if attempt==2: raise
                time.sleep(1)
        (OUT/(name+'_raw.json')).write_bytes(body)
        data=json.loads(body)
        features=data.get('features',[])
        record.update(sha256=hashlib.sha256(body).hexdigest(),count=len(features),complete=not any(x['rel']=='next' for x in data.get('links',[])))
        record['items']=[{'id':f['id'],'datetime':f['properties'].get('datetime'),'cloud':f['properties'].get('eo:cloud_cover'),
            'assets':list(f['assets']),'sample_asset':next((a.get('href') for k,a in f['assets'].items() if k in ['B04','red']),None)} for f in features]
    except Exception as e: record['error']=str(e)
    print(json.dumps(record),flush=True)
    return record
with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,queries))
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'frozen_design_sha256':hashlib.sha256(DESIGN.read_bytes()).hexdigest(),'records':records,'raster_downloaded':False}
(OUT/'public_source_checks.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('OUTPUT='+str(OUT),flush=True)
