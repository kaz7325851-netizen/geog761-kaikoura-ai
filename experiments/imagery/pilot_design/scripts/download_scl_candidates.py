"""Download only pilot-sized SCL windows; compare dates before model inference."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
from urllib.parse import urlencode
import json,hashlib,time
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform,unary_union

ROOT=Path(__file__).resolve().parent
DESIGN_DIR=ROOT/'20261005T200713Z'
CATALOG=ROOT/'imagery_checks/20261005T201810Z/planetary_raw.json'
OUT=ROOT/'imagery_checks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True)
target=from_origin(1667250,5328030,10,10)
to_metric=Transformer.from_crs(4326,2193,always_xy=True).transform
road=unary_union([transform(to_metric,shape(f['geometry'])) for f in json.loads((DESIGN_DIR/'sh1_pilot_road.geojson').read_text())['features']])
roadmask=rasterize([(road.buffer(100),1)],out_shape=(640,640),transform=target,fill=0,dtype='uint8').astype(bool)
features=json.loads(CATALOG.read_text())['features']
items=[f for f in features if f['properties'].get('s2:mgrs_tile')=='60GTU' and f['properties']['datetime'][:10] in ['2016-11-22','2016-11-12','2016-10-26','2016-10-03']]

def sign(href):
    url='https://planetarycomputer.microsoft.com/api/sas/v1/sign?'+urlencode({'href':href})
    for attempt in range(3):
        try:
            with urlopen(url,timeout=30) as r: return json.load(r)['href']
        except Exception:
            if attempt==2: raise
            time.sleep(1)

def dilate(mask,radius=6):
    padded=np.pad(mask,radius)
    result=np.zeros(mask.shape,bool)
    h,w=mask.shape
    for dx in range(-radius,radius+1):
        for dy in range(-radius,radius+1):
            if dx*dx+dy*dy<=radius*radius:
                result |= padded[radius+dy:radius+dy+h,radius+dx:radius+dx+w]
    return result

def quality(a):
    invalid=np.isin(a,[0,1])
    cloudshadow=np.isin(a,[3,8,9,10])
    uncertain=np.isin(a,[2,7,11])
    basic=~(invalid|cloudshadow|uncertain)
    strict=basic & ~dilate(cloudshadow)
    values,counts=np.unique(a,return_counts=True)
    return {'class_counts':{str(v):int(c) for v,c in zip(values,counts)},
        'aoi_strict_usable_fraction':float(strict.mean()),
        'road100m_strict_usable_fraction':float(strict[roadmask].mean()),
        'road100m_basic_usable_fraction':float(basic[roadmask].mean()),
        'road100m_cloud_shadow_fraction':float(cloudshadow[roadmask].mean()),
        'road100m_uncertain_fraction':float(uncertain[roadmask].mean()),
        'road100m_nodata_defective_fraction':float(invalid[roadmask].mean())}

def download(item):
    record={'item_id':item['id'],'datetime':item['properties']['datetime'],'baseline':item['properties'].get('s2:processing_baseline'),
        'source_href':item['assets']['SCL']['href'],'whole_tile_cloud_percent':item['properties'].get('eo:cloud_cover')}
    try:
        href=sign(record['source_href'])
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_USE_HEAD='NO'):
            with rasterio.open(href) as src:
                record['native_crs']=str(src.crs);record['native_resolution_m']=list(src.res)
                with WarpedVRT(src,crs='EPSG:2193',transform=target,width=640,height=640,resampling=Resampling.nearest,nodata=0) as vrt: a=vrt.read(1)
        path=OUT/(item['id']+'_SCL.tif')
        with rasterio.open(path,'w',driver='GTiff',width=640,height=640,count=1,dtype='uint8',crs='EPSG:2193',transform=target,nodata=0,compress='deflate',tiled=True) as dst:
            dst.write(a,1);dst.set_band_description(1,'SCL');dst.update_tags(source_item=item['id'],resampling='nearest',native_resolution_m=20)
        record.update(file=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),quality=quality(a),status='downloaded_and_checked')
    except Exception as e: record['error']=str(e).split('?')[0]
    print(json.dumps(record),flush=True)
    return record

with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(download,items))
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'directory':str(OUT),'records':records,
    'original_pre_date_available_on_this_mirror':False,'original_design_unchanged':True,
    'strict_mask_rule':'Exclude SCL0,1,2,3,7,8,9,10,11; dilate cloud/shadow 3,8,9,10 by60m',
    'road_buffer_m':100,'raster_grid_crs':'EPSG:2193','grid':[10,0,1667250,0,-10,5328030],
    'limitations':['SCL is imperfect and this is conservative quality screening, not road-status classification',
        'SCL2 represents dark features/shadows in these older baselines; treated uncertain, not definitive cloud',
        'No reflectance bands downloaded at this stage, no registration or model accuracy checked']}
(OUT/'scl_quality_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('OUTPUT='+str(OUT),flush=True)
