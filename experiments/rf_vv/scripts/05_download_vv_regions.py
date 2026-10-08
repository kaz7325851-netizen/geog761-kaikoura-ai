"""Download only observed GEE pixel URLs and verify exact frozen grids.

No authentication extraction, model fitting, or changes to prior data.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import urllib.request
import json
import hashlib
import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent
design_path = ROOT/'frozen_data_design.json'
observed_path = ROOT/'earthengine_observed_exports.json'
design = json.loads(design_path.read_text(encoding='utf-8'))
observed = json.loads(observed_path.read_text(encoding='utf-8'))
assert observed['observed_in_earthengine_console'] is True
assert set(observed['urls']) == {s['id'] for s in design['sites']}
assert observed['common']['band_order'] == design['band_order']
OUT = ROOT/'downloads'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True,exist_ok=False)
def download(site):
    sid=site['id']; url=observed['urls'][sid]; parsed=urlparse(url)
    assert parsed.scheme=='https' and parsed.hostname=='earthengine.googleapis.com'
    assert '/thumbnails/' in parsed.path and parsed.path.endswith(':getPixels')
    dest=OUT/(sid+'.tif')
    with urllib.request.urlopen(url,timeout=90) as response,dest.open('xb') as f:
        assert response.status==200
        total=0
        while chunk:=response.read(1024*1024):
            total+=len(chunk); assert total<35_000_000
            f.write(chunk)
    with rasterio.open(dest) as src:
        assert src.shape==tuple(site['shape']) and src.crs.to_epsg()==2193
        assert src.transform.almost_equals(Affine(*site['grid']),precision=1e-10)
        assert src.count==17 and all(dt=='float32' for dt in src.dtypes)
        a=src.read(); descriptions=list(src.descriptions)
    if any(descriptions): assert descriptions==design['band_order']
    for k in range(12,17): assert np.isin(a[k],[0,1,-9999]).all()
    data_valid=(a[12]==1)&(a[13]==1)&np.isfinite(a[:3]).all(axis=0)&(a[:3]!=-9999).all(axis=0)
    assert data_valid.any()
    assert np.allclose(a[2,data_valid],a[1,data_valid]-a[0,data_valid],atol=2e-5,rtol=1e-5)
    med_valid=data_valid&(a[14]==1)&(a[3]!=-9999)&(a[4]!=-9999)
    assert np.allclose(a[4,med_valid],a[1,med_valid]-a[3,med_valid],atol=2e-5,rtol=1e-5)
    geom=data_valid&(a[15]==1)&(a[16]==1)
    record={'id':sid,'file':str(dest),'bytes':total,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
            'grid':site['grid'],'shape':site['shape'],'band_order':design['band_order'],
            'data_valid_pixels':int(data_valid.sum()),'data_valid_fraction':float(data_valid.mean()),
            'both_local_geometry_visible_pixels':int(geom.sum()),
            'both_local_geometry_visible_fraction':float(geom.mean()),
            'delta_db_quantiles_p1_p50_p99':np.percentile(a[2,data_valid],[1,50,99]).tolist(),
            'checks':'PASS grid, finite data, binary masks, two difference formulas',
            'limitations':['Acquisition masks do not guarantee terrain visibility','Active local geometry approximation only']}
    print(json.dumps({k:record[k] for k in ['id','bytes','data_valid_fraction','both_local_geometry_visible_fraction','checks']}),flush=True)
    return record
with ThreadPoolExecutor(max_workers=3) as pool:
    records=list(pool.map(download,design['sites']))
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'directory':str(OUT),
          'status':'8_actual_VV_regions_downloaded_and_numerically_checked',
          'records':records,'common':observed['common'],
          'design_sha256':hashlib.sha256(design_path.read_bytes()).hexdigest(),
          'observed_export_sha256':hashlib.sha256(observed_path.read_bytes()).hexdigest(),
          'script_sha256':hashlib.sha256((ROOT/'02_export_vv.js').read_bytes()).hexdigest(),
          'training_ready':False,'training_performed':False,'prediction_performed':False}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'latest_vv_download.json').write_text(json.dumps({'directory':str(OUT)},ensure_ascii=False,indent=2),encoding='utf-8')
print('DOWNLOAD_DIRECTORY='+str(OUT),flush=True)
