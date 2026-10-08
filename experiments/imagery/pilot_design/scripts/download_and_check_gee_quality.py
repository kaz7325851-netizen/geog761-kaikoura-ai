"""Download observed GEE quality crops, NOT ready-to-use model inputs."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
from urllib.parse import urlparse
import json,hashlib,time
import numpy as np
import rasterio
from rasterio.transform import Affine,rowcol
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform,unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DESIGN=ROOT/'20261005T200713Z'
urls=json.loads((ROOT/'gee_quality_urls.json').read_text())
OUT=ROOT/'imagery_checks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True)
affine=Affine(*urls['grid'])
project=Transformer.from_crs(4326,2193,always_xy=True).transform
road=unary_union([transform(project,shape(f['geometry'])) for f in json.loads((DESIGN/'sh1_pilot_road.geojson').read_text())['features']])
roadmask=rasterize([(road.buffer(100),1)],out_shape=(640,640),transform=affine,fill=0,dtype='uint8').astype(bool)
slides=json.loads((DESIGN/'reference_inventory.geojson').read_text())['features']
slide_geoms=[transform(project,shape(f['geometry'])) for f in slides]

def dilate(mask,radius=6):
    padded=np.pad(mask,radius)
    result=np.zeros(mask.shape,bool)
    h,w=mask.shape
    for dx in range(-radius,radius+1):
        for dy in range(-radius,radius+1):
            if dx*dx+dy*dy<=radius*radius:
                result|=padded[radius+dy:radius+dy+h,radius+dx:radius+dx+w]
    return result

def download(key):
    u=urls[key]['url'];parsed=urlparse(u)
    assert parsed.scheme=='https' and parsed.hostname=='earthengine.googleapis.com' and parsed.path.endswith(':getPixels')
    dest=OUT/(key+'_TOA_quality.tif')
    for attempt in range(3):
        try:
            with urlopen(u,timeout=60) as response,dest.open('wb') as f:
                total=0
                while chunk:=response.read(1024*1024):
                    total+=len(chunk);assert total<30_000_000;f.write(chunk)
            break
        except Exception:
            if attempt==2: raise
            time.sleep(1)
    with rasterio.open(dest) as src:
        assert src.crs.to_string()==urls['crs'] and src.shape==(640,640) and src.count==7
        assert src.transform.almost_equals(affine,precision=1e-8)
        data=src.read()
        if all(src.descriptions): assert list(src.descriptions)==urls['bands']
    valid=np.all(np.isfinite(data),axis=0)&np.all(data!=-9999,axis=0)&np.all(data[:4]>0,axis=0)
    qa=data[4].clip(0).astype(np.uint16)
    qa_cloud=(qa & ((1<<10)|(1<<11)))!=0
    scores=data[6]
    clear=valid&~qa_cloud&(scores>=.60)
    strict=clear&~dilate(valid&(~clear))
    metrics={'valid_aoi_fraction':float(valid.mean()),'clear_aoi_fraction':float(clear.mean()),'strict_aoi_fraction':float(strict.mean()),
        'road100m_clear_fraction':float(clear[roadmask].mean()),'road100m_strict_fraction':float(strict[roadmask].mean()),
        'road100m_invalid_fraction':float((~valid)[roadmask].mean()),
        'threshold_sensitivity_road100m':{str(t):float((valid&~qa_cloud&(scores>=t))[roadmask].mean()) for t in [.50,.60,.65]}}
    record={'key':key,'file':str(dest),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'source_metadata':{k:v for k,v in urls[key].items() if k!='url'},'bands':urls['bands'],'metrics':metrics}
    print(json.dumps(record),flush=True)
    return key,data,strict,record

with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(download,['pre','post']))
data={r[0]:r[1] for r in results}
masks={r[0]:r[2] for r in results}
joint=masks['pre']&masks['post']
with rasterio.open(OUT/'joint_quality_mask.tif','w',driver='GTiff',width=640,height=640,count=1,dtype='uint8',crs='EPSG:2193',transform=affine,compress='deflate',tiled=True) as dst:
    dst.write(joint.astype('uint8'),1);dst.set_band_description(1,'joint_clear_mask');dst.update_tags(meaning='1=passes specified quality rule;0=not accepted, NOT road safety',cloud_score_cdf_threshold=.60,buffer_m=60)

road_segment_features=json.loads((DESIGN/'road_evidence_segments.geojson').read_text())['features']
segment_records=[]
for feature in road_segment_features:
    props=feature['properties'];g=transform(project,shape(feature['geometry']))
    sm=rasterize([(g.buffer(50),1)],out_shape=(640,640),transform=affine,fill=0,dtype='uint8').astype(bool)
    pts=[g.interpolate(float(s)) for s in np.arange(5,g.length,10)]
    rr,cc=rowcol(affine,[p.x for p in pts],[p.y for p in pts])
    fraction=float(np.mean(joint[np.asarray(rr),np.asarray(cc)]))
    segment_records.append({**props,'joint_clear_50m_buffer_fraction':float(joint[sm].mean()),'joint_clear_sampled_centerline_fraction':fraction,
        'quality_ge_90pct':bool(fraction>=.90),'road_status':'unknown'})

slide_records=[]
for f,g in zip(slides,slide_geoms):
    pixels=rasterize([(g,1)],out_shape=(640,640),transform=affine,fill=0,dtype='uint8').astype(bool)
    count=int(pixels.sum())
    if not count: continue
    p=f['properties']
    slide_records.append({**p,'raster_pixels':count,'joint_clear_pixels':int((pixels&joint).sum()),'joint_clear_fraction':float(joint[pixels].mean()),
        'near_sh1_100m':p['distance_to_sh1_m']<=100})

# Diagnostic phase correlation on stable-looking non-inventory land windows.
# This is NOT a ground-control-point registration validation or a correction.
inventory_buffer=rasterize([(g.buffer(100),1) for g in slide_geoms],out_shape=(640,640),transform=affine,fill=0,dtype='uint8').astype(bool)
post_ndvi=(data['post'][3]-data['post'][2])/(data['post'][3]+data['post'][2]+1e-6)
stable=joint&~inventory_buffer&(post_ndvi>.25)
registration=[]
for y in range(0,513,128):
    for x in range(0,513,128):
        m=stable[y:y+128,x:x+128]
        if m.mean()<.70: continue
        aa=data['pre'][2,y:y+128,x:x+128].astype(float)
        bb=data['post'][2,y:y+128,x:x+128].astype(float)
        taper=np.outer(np.hanning(128),np.hanning(128))*m
        aa=(aa-aa[m].mean())*taper;bb=(bb-bb[m].mean())*taper
        cross=np.fft.fft2(aa)*np.conj(np.fft.fft2(bb));cross/=np.maximum(np.abs(cross),1e-12)
        corr=np.abs(np.fft.ifft2(cross));dy,dx=np.unravel_index(np.argmax(corr),corr.shape)
        dy=dy if dy<=64 else dy-128;dx=dx if dx<=64 else dx-128
        registration.append({'window_col':x,'window_row':y,'stable_fraction':float(m.mean()),'shift_xy_pixels':[int(dx),int(dy)],'peak_to_mean':float(corr.max()/corr.mean())})

fig,axes=plt.subplots(2,3,figsize=(13,8))
extent=[1667.25,1673.65,5321.63,5328.03]
for i,key in enumerate(['pre','post']):
    rgb=np.moveaxis(data[key][[2,1,0]],0,-1);rgb=np.clip(rgb/3000,0,1)
    axes[i,0].imshow(rgb,extent=extent)
    axes[i,0].set_title(key+' RGB (TOA preview, fixed stretch)')
    axes[i,1].imshow(data[key][6],extent=extent,vmin=0,vmax=1,cmap='viridis');axes[i,1].set_title(key+' Cloud Score+ cs_cdf')
    axes[i,2].imshow(masks[key],extent=extent,vmin=0,vmax=1,cmap='Greys_r');axes[i,2].set_title(key+' accepted quality mask')
    for ax in axes[i]:
        parts=list(road.geoms) if hasattr(road,'geoms') else [road]
        for line in parts:
            xx,yy=line.xy;ax.plot(np.array(xx)/1000,np.array(yy)/1000,color='#e64b35',lw=.9)
        ax.set_xlabel('NZTM easting (km)');ax.set_ylabel('NZTM northing (km)')
fig.suptitle('Original SH1 pilot dates: 3 Nov / 23 Nov 2016 NZ | quality only',fontsize=14)
fig.tight_layout(rect=[0,.04,1,.95]);fig.text(.5,.015,'TOA is not L2A model input. Clear mask is visibility screening, NOT safe-road classification.',ha='center',fontsize=10)
fig.savefig(OUT/'original_dates_quality.png',dpi=150);plt.close(fig)

summary={'created_utc':datetime.now(timezone.utc).isoformat(),'directory':str(OUT),'records':[r[3] for r in results],
    'joint_strict_aoi_fraction':float(joint.mean()),'joint_strict_road100m_fraction':float(joint[roadmask].mean()),
    'sampled_road_centerline_joint_clear_fraction':float(np.mean([s['joint_clear_sampled_centerline_fraction'] for s in segment_records])),
    'segments_total':len(segment_records),'segments_ge90pct':sum(s['quality_ge_90pct'] for s in segment_records),
    'segment_support_by_role':{role:sum(s['quality_ge_90pct'] for s in segment_records if s['role']==role) for role in sorted(set(s['role'] for s in segment_records))},
    'near_road_inventory_ge80pct_joint_clear':sum(s['near_sh1_100m'] and s['joint_clear_fraction']>=.8 for s in slide_records),
    'phase_correlation_diagnostics':registration,'same_export_grid_verified':True,
    'reflectance_level':'L1C harmonized TOA','model_ready':False,'model_run':False,'training':False,
    'limitations':['Quality processor imperfect; masks do not prove road passability','No matched pre L2A obtained for original date',
        'Phase correlation is diagnostic only, may be affected by texture and change; not independent ground control',
        'SRTM/ALOS terrain and full model-specific spectral inputs not prepared yet','Thresholds not tuned on model predictions']}
(OUT/'quality_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(OUT/'road_segment_quality.json').write_text(json.dumps(segment_records,indent=2),encoding='utf-8')
(OUT/'inventory_quality.json').write_text(json.dumps(slide_records,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2),flush=True)
