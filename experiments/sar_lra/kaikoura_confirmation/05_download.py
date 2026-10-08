"""Download only the four explicitly observed public GEE pixel URLs; validate grids."""
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import urllib.request,json,hashlib
import rasterio
from rasterio.transform import Affine
import numpy as np
ROOT=Path(__file__).resolve().parent
design=json.loads((ROOT/'frozen_design.json').read_text(encoding='utf-8'))
control=json.loads((ROOT/'control_grid.json').read_text(encoding='utf-8'))
urls=json.loads((ROOT/'earthengine_urls.json').read_text(encoding='utf-8'))
sites=design['sites']+[control]
common={'scene_ids':['S1A_IW_GRDH_1SDV_20160823T072219_20160823T072248_012724_01403F_DA98','S1A_IW_GRDH_1SDV_20161115T072214_20161115T072243_013949_016775_19DF','S1A_IW_GRDH_1SDV_20161115T072243_20161115T072308_013949_016775_0C46'],
        'headings_deg':[280.09269175909117,280.26130744845557,281.57461516894506],
        'band_order':['postVV','postVH','diffVV','diffVH','valid','preVV','preVH','elevation','slope_deg','aspect_deg','scene0_angle','scene0_alpha_r','scene0_coverage','scene1_angle','scene1_alpha_r','scene1_coverage','scene2_angle','scene2_alpha_r','scene2_coverage'],
        'metadata_observed_in_gee_console':True,'relative_orbit':52,'pass':'ASCENDING'}
out=ROOT/'downloads'/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
def download(site):
    sid=site['id'];url=urls[sid];u=urlparse(url)
    assert u.scheme=='https' and u.hostname=='earthengine.googleapis.com' and '/thumbnails/' in u.path and u.path.endswith(':getPixels')
    dest=out/(sid+'.tif')
    with urllib.request.urlopen(url,timeout=120) as response,dest.open('wb') as f:
        total=0
        while chunk:=response.read(1024*1024):
            total+=len(chunk);assert total<45_000_000;f.write(chunk)
    with rasterio.open(dest) as src:
        assert src.crs.to_string()==site['crs'] and src.shape==tuple(reversed(site['dimensions']))
        assert src.transform.almost_equals(Affine(*site['grid']),precision=1e-10)
        assert src.count==(7 if sid=='OLD_GEO' else 19) and all(t=='float32' for t in src.dtypes)
        data=src.read();valid=data[4]==1
        assert np.isin(data[4],[0,1,-9999]).all() and valid.any()
        assert np.isfinite(data[:7,valid]).all() and (data[:7,valid]!=-9999).all()
        assert np.allclose(data[2,valid],data[0,valid]-data[5,valid],atol=2e-5,rtol=1e-5)
        assert np.allclose(data[3,valid],data[1,valid]-data[6,valid],atol=2e-5,rtol=1e-5)
        descriptions=list(src.descriptions)
    metadata={**common,'id':sid,'crs':site['crs'],'grid':site['grid'],'dimensions':site['dimensions'],'nodata':-9999}
    if sid=='OLD_GEO':metadata['band_order']=metadata['band_order'][:7]
    if all(descriptions):assert descriptions==metadata['band_order']
    result={'id':sid,'file':str(dest),'bytes':total,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
            'valid_pixels':int(valid.sum()),'valid_fraction':float(valid.mean()),'band_descriptions':descriptions,'metadata':metadata,
            'quantiles_p1_p25_p50_p75_p99':np.quantile(data[:4,valid],[.01,.25,.5,.75,.99],axis=1).T.tolist(),'input_checks':'PASS'}
    print(json.dumps({k:result[k] for k in ['id','bytes','valid_fraction','input_checks']},indent=2),flush=True)
    return result
with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(download,sites))
manifest={'created':datetime.now().isoformat(),'frozen_design_sha256':hashlib.sha256((ROOT/'frozen_design.json').read_bytes()).hexdigest(),
          'records':records,'downloaded_input_count':4,'training':False,'inference':False}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(ROOT/'latest_download.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('DOWNLOAD_DIRECTORY='+str(out))
