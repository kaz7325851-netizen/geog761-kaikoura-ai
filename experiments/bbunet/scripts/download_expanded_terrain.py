"""Acquire public author-compatible PC ALOS assets; no private credentials."""
import requests,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'expanded_test'/'20261007T210924Z'/'sources'
OUT.mkdir(exist_ok=True)
design=json.loads((OUT.parent/'frozen_design.json').read_text())
b=[r['bounds_wgs84'] for r in design['regions']+[design['anchor']]]
bbox=[min(x[0] for x in b),min(x[1] for x in b),max(x[2] for x in b),max(x[3] for x in b)]
r=requests.get('https://planetarycomputer.microsoft.com/api/stac/v1/search',params={'collections':'alos-dem','bbox':','.join(map(str,bbox)),'limit':20},timeout=30)
r.raise_for_status();catalog=r.json()
(OUT/'ALOS_STAC_catalog.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
records=[]
for item in catalog['features']:
    u=item['assets']['data']['href'];dest=OUT/(item['id']+'.tif')
    if not dest.exists():
        signed=requests.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':u},timeout=30)
        signed.raise_for_status()
        signed_url=signed.json()['href']
        # Public PC read-only asset signing, not user-account credentials.
        with requests.get(signed_url,stream=True,timeout=(20,60)) as resp:
            resp.raise_for_status()
            if 'text/html' in resp.headers.get('Content-Type',''):raise RuntimeError('HTML not DEM')
            part=dest.with_suffix('.tif.partial')
            assert not part.exists(),'Preserve previous partial for diagnosis'
            count=0;last=time.monotonic()
            with part.open('xb') as f:
                for block in resp.iter_content(1024*1024):
                    if block:f.write(block);count+=len(block)
                    if time.monotonic()-last>10:
                        print(f'{item["id"]}: {count:,} bytes',flush=True);last=time.monotonic()
            if resp.headers.get('Content-Length'):assert count==int(resp.headers['Content-Length'])
            part.rename(dest)
    import rasterio
    with rasterio.open(dest) as ds:
        a=ds.read(1);assert ds.crs and a.size>0
        rec={'item_id':item['id'],'href':u,'file':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
             'bytes':dest.stat().st_size,'crs':ds.crs.to_string(),'shape':list(ds.shape),'transform':list(ds.transform)[:6],
             'nodata':ds.nodata,'tags':ds.tags()}
    records.append(rec);print(json.dumps(rec),flush=True)
(OUT/'terrain_acquisition.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
    'bbox':bbox,'records':records,'published_file_checksum_available':False,'training':False},indent=2),encoding='utf-8')
