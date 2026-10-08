"""Stage public pixel crops. No training, inference or final model normalization."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
from urllib.parse import urlparse
import json, hashlib, time
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform, unary_union

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'gee_full_input_urls.json').read_text())
OUT=ROOT/'prepared_inputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True)
DESIGN=ROOT/'20261005T200713Z'
QUALITY=ROOT/'imagery_checks'/'20261005T202623Z'
affine=Affine(*CONFIG['grid'])
project=Transformer.from_crs(4326,2193,always_xy=True).transform
road=unary_union([transform(project,shape(f['geometry'])) for f in json.loads((DESIGN/'sh1_pilot_road.geojson').read_text())['features']])
roadmask=rasterize([(road.buffer(100),1)],out_shape=(640,640),transform=affine,dtype='uint8').astype(bool)

def get(key):
    srcinfo=CONFIG[key]; url=srcinfo['url']; parsed=urlparse(url)
    assert parsed.scheme=='https' and parsed.hostname=='earthengine.googleapis.com' and parsed.path.endswith(':getPixels')
    optical=key in ['pre','post']; bands=CONFIG['optical_bands'] if optical else CONFIG['terrain_bands']
    dest=OUT/({'pre':'pre_20161103NZ_TOA_12bands.tif','post':'post_20161123NZ_TOA_12bands.tif','terrain':'ALOS_terrain_staged_10m.tif','terrain_audit':'ALOS_terrain_30m_audit.tif'}[key])
    for attempt in range(3):
        try:
            with urlopen(url,timeout=60) as response, dest.open('wb') as f:
                total=0
                while chunk:=response.read(1024*1024):
                    total+=len(chunk);assert total<32_000_000;f.write(chunk)
            break
        except Exception:
            if attempt==2:raise
            time.sleep(1)
    targetaffine=Affine(*srcinfo.get('grid',CONFIG['grid']))
    targetshape=tuple(srcinfo.get('shape',[640,640]))
    with rasterio.open(dest) as src:
        assert src.crs.to_string()==CONFIG['crs'] and src.shape==targetshape and src.count==len(bands)
        assert src.transform.almost_equals(targetaffine,precision=1e-8)
        array=src.read()
        descriptions=list(src.descriptions)
        if all(descriptions):assert descriptions==bands
    # Record explicit units and nodata without changing numeric source data.
    with rasterio.open(dest,'r+') as dst:
        dst.nodata=-9999
        for i,b in enumerate(bands,1):dst.set_band_description(i,b)
        dst.update_tags(processing_role='staging_only_not_model_tensor',source_level='L1C_TOA' if optical else 'DSM_and_derived_terrain',resampling=CONFIG['optical_resampling'] if optical else 'derivatives_at_30m_nearest_to_10m',reflectance_divisor=10000 if optical else 'not_applicable')
    valid=np.all(np.isfinite(array)&(array!=-9999),axis=0)
    if optical:
        valid&=np.all(array>0,axis=0)
    else:
        # STK is an auxiliary stack-count layer, not a terrain model feature.
        # It is absent over sea and some gap-fill areas where DSM remains valid.
        valid=np.all(np.isfinite(array[:7])&(array[:7]!=-9999),axis=0)
        valid&=array[6]!=1 # AW3D30 MSK=1 is cloud/snow-invalid; sea DSM=0 is valid.
        assert np.all((array[1][valid]>=0)&(array[1][valid]<=90))
        assert np.all((array[2][valid]>=0)&(array[2][valid]<=360))
        assert np.all(np.abs(array[4][valid]**2+array[5][valid]**2-1)<1e-5)
    record={'key':key,'file':str(dest),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
      'source':{k:v for k,v in srcinfo.items() if k!='url'},'shape':list(targetshape),'grid':list(targetaffine)[:6],
      'bands':bands,'valid_aoi_fraction':float(valid.mean()),'per_band_valid_fraction':{b:float(np.mean(np.isfinite(array[i])&(array[i]!=-9999))) for i,b in enumerate(bands)},
      'band_ranges':{b:[float(array[i][np.isfinite(array[i])&(array[i]!=-9999)].min()),float(array[i][np.isfinite(array[i])&(array[i]!=-9999)].max())] for i,b in enumerate(bands)}}
    if key!='terrain_audit':record['valid_road100m_fraction']=float(valid[roadmask].mean())
    if optical:
        with rasterio.open(QUALITY/(key+'_TOA_quality.tif')) as q:preview=q.read()[:4]
        dif=np.abs(array[[1,2,3,7]]-preview)
        record['rgb_nir_match_to_quality_export_max_abs']=float(dif[:,valid].max())
        assert record['rgb_nir_match_to_quality_export_max_abs']<.01
    np.savez_compressed(OUT/(key+'_validity.npz'),valid=valid)
    print(json.dumps(record),flush=True)
    return record

with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(get,['pre','post','terrain','terrain_audit']))
valids={key:np.load(OUT/(key+'_validity.npz'))['valid'] for key in ['pre','post','terrain']}
with rasterio.open(QUALITY/'joint_quality_mask.tif') as src:clear=src.read(1).astype(bool)
joint=clear&valids['pre']&valids['post']&valids['terrain']
with rasterio.open(OUT/'staged_input_validity_mask.tif','w',driver='GTiff',width=640,height=640,count=1,dtype='uint8',crs=CONFIG['crs'],transform=affine,compress='deflate') as dst:
    dst.write(joint.astype('uint8'),1);dst.set_band_description(1,'staged_input_validity');dst.update_tags(meaning='1=specified visibility rule and all staged channels valid; NOT road passability')
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'output_directory':str(OUT),'records':records,
    'quality_directory':str(QUALITY),'joint_staged_valid_aoi_fraction':float(joint.mean()),'joint_staged_valid_road100m_fraction':float(joint[roadmask].mean()),
    'grid_alignment_verified':True,'model_ready':False,'model_run':False,'training':False,
    'remaining':['No matched pre L2A acquired for BBUnet','Model-specific scaling, band order, terrain version and hillshade settings not yet verified against checkpoint code','Current road geometry and inventory proximity do not establish actual 2016 passability','Training-area overlap and independent-test status not established','Subpixel registration not independently checked']}
(OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='records'},indent=2),flush=True)
