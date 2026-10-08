"""Stage VV-only multi-region features and provisional inventory candidates.

Do not treat inventory absence as a negative, or geometric visibility as safety.
No fitting, weights or optical predictor inputs.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import os
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, binary_erosion

ROOT = Path(__file__).resolve().parent
DOWN = Path(json.loads((ROOT/'latest_vv_download.json').read_text(encoding='utf-8'))['directory'])
download = json.loads((DOWN/'manifest.json').read_text(encoding='utf-8'))
design = json.loads((ROOT/'frozen_data_design.json').read_text(encoding='utf-8'))
assert hashlib.sha256((ROOT/'frozen_data_design.json').read_bytes()).hexdigest() == download['design_sha256']
REF = ROOT.parent/'bbunet/expanded_test/20261007T210924Z'
OUT = ROOT/'prepared_regions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True, exist_ok=False)
NAMES = ['VV_pre_filtered_db','VV_post_filtered_db','dVV_filtered_db','abs_dVV_filtered_db',
         'pre_local_std_5x5_db','post_local_std_5x5_db','delta_local_std_5x5_db',
         'elevation_m','slope_deg','aspect_sin','aspect_cos']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(dest, name, a, descriptions, profile, nodata):
    if a.ndim==2: a=a[None]
    cfg=dict(profile)
    cfg.update(driver='GTiff',count=len(a),dtype=str(a.dtype),nodata=nodata,
               compress='deflate',predictor=3 if a.dtype.kind=='f' else 2)
    p=dest/name
    with rasterio.open(p,'w',**cfg) as dst:
        dst.write(a)
        for i,d in enumerate(descriptions,1): dst.set_band_description(i,d)
        dst.update_tags(training_ready='false',prediction='false',
                        pre_date='2016-11-03',post_date='2016-11-15',input_sensor='Sentinel1_VV')
    with rasterio.open(p) as src:
        assert np.array_equal(src.read(),a) and src.transform==profile['transform'] and src.crs==profile['crs']
        assert list(src.descriptions)==descriptions and src.nodata==nodata
    return {'file':str(p),'sha256':sha(p),'bands':descriptions,'nodata':nodata,'readback':'PASS'}
def filtered_db(db, valid):
    power=np.power(10.,np.where(valid,db,0)/10)
    counts=uniform_filter(valid.astype('float64'),size=3,mode='constant')
    sums=uniform_filter(np.where(valid,power,0).astype('float64'),size=3,mode='constant')
    return 10*np.log10(np.maximum(np.divide(sums,counts,out=np.ones_like(sums),where=counts>0),1e-12))
def local_std(db, valid):
    counts=uniform_filter(valid.astype('float64'),size=5,mode='constant')
    mean=np.divide(uniform_filter(np.where(valid,db,0).astype('float64'),size=5,mode='constant'),
                   counts,out=np.zeros_like(counts),where=counts>0)
    second=np.divide(uniform_filter(np.where(valid,db**2,0).astype('float64'),size=5,mode='constant'),
                     counts,out=np.zeros_like(counts),where=counts>0)
    return np.sqrt(np.maximum(second-mean**2,0))
records=[]
for r in download['records']:
    rid=r['id']; p=Path(r['file']); assert sha(p)==r['sha256']
    dest=OUT/rid; dest.mkdir()
    with rasterio.open(p) as src:
        a=src.read(); profile=src.profile.copy(); grid=(src.shape,src.transform,src.crs)
    source_names=download['common']['band_order']
    band=dict(zip(source_names,a))
    valid=(band['valid_pre']==1)&(band['valid_post']==1)&np.isfinite(a[:3]).all(axis=0)&(a[:3]!=-9999).all(axis=0)
    terrain_valid=np.isfinite(a[7:10]).all(axis=0)&(a[7:10]!=-9999).all(axis=0)
    geometry=(band['geometry_visible_pre']==1)&(band['geometry_visible_post']==1)
    # Numerical mask excludes complete radius3 neighborhoods adjacent to missing data/AOI edges.
    feature_valid=binary_erosion(valid,np.ones((7,7)),border_value=0)&terrain_valid
    screened=feature_valid&geometry
    pre=filtered_db(band['VV_pre'],valid); post=filtered_db(band['VV_post'],valid)
    d=post-pre; prestd=local_std(pre,valid); poststd=local_std(post,valid)
    f=np.stack([pre,post,d,np.abs(d),prestd,poststd,poststd-prestd,band['elevation'],band['slope_deg'],
                np.sin(np.radians(band['aspect_deg'])),np.cos(np.radians(band['aspect_deg']))]).astype('float32')
    assert np.isfinite(f[:,feature_valid]).all(); f[:,~feature_valid]=-9999
    source_ref=REF/rid/'inventory_raster.tif'; source_road=REF/rid/'road100m_mask.tif'
    with rasterio.open(source_ref) as src:
        assert (src.shape,src.transform,src.crs)==grid; inventory=src.read(1)
    with rasterio.open(source_road) as src:
        assert (src.shape,src.transform,src.crs)==grid; road=src.read(1).astype(bool)
    assert np.isin(inventory,[0,1]).all()
    labels=np.full(grid[0],255,dtype='uint8'); labels[(inventory==1)&screened]=1
    assert 0 not in np.unique(labels)
    outputs=[write(dest,'VV_feature_candidates.tif',f,NAMES,profile,-9999),
             write(dest,'data_validity_mask.tif',valid.astype('uint8'),['pre_post_data_present'],profile,None),
             write(dest,'local_geometry_approx_mask.tif',geometry.astype('uint8'),['both_dates_active_local_geometry_approx'],profile,None),
             write(dest,'feature_numerical_validity_mask.tif',feature_valid.astype('uint8'),['numerical_feature_valid'],profile,None),
             write(dest,'screened_feature_mask.tif',screened.astype('uint8'),['numerical_valid_AND_local_geometry_NOT_certified_visibility'],profile,None),
             write(dest,'inventory_reference_only.tif',inventory.astype('uint8'),['inventory_candidate_NOT_verified_truth'],profile,None),
             write(dest,'candidate_labels_NOT_TRAIN_READY.tif',labels,['1_inventory_candidate_255_unknown_NO_verified_negatives'],profile,255),
             write(dest,'road100m_context_mask.tif',road.astype('uint8'),['within100m_SH1_current_OSM_NOT_blockage'],profile,None)]
    record={'id':rid,'raw_file':str(p),'raw_sha256':sha(p),'reference_file':str(source_ref),'reference_sha256':sha(source_ref),
            'road_file':str(source_road),'road_sha256':sha(source_road),'outputs':outputs,
            'data_valid_pixels':int(valid.sum()),'data_coverage_pct':float(valid.mean()*100),
            'feature_numerical_coverage_pct':float(feature_valid.mean()*100),
            'screened_feature_coverage_pct':float(screened.mean()*100),
            'road100m_pixels':int(road.sum()),'road100m_data_coverage_pct':float(valid[road].mean()*100) if road.any() else None,
            'road100m_screened_feature_coverage_pct':float(screened[road].mean()*100) if road.any() else None,
            'inventory_pixels':int(inventory.sum()),'candidate_label_pixels':int((labels==1).sum()),
            'candidate_label_pixels_near_road100m':int(((labels==1)&road).sum()),'verified_negative_pixels':0,
            'pre_post_raw_correlation_on_land_proxy':float(np.corrcoef(a[0,valid&(a[7]>1)],a[1,valid&(a[7]>1)])[0,1])}
    (dest/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    records.append(record)
    print(json.dumps({k:record[k] for k in ['id','data_coverage_pct','screened_feature_coverage_pct','candidate_label_pixels']},ensure_ascii=False),flush=True)

reference_report=json.loads((REF/'expanded_results.json').read_text(encoding='utf-8'))['reference_source']
report={'directory':str(OUT),'status':'8_VV_feature_packs_prepared_NOT_train_ready',
        'pre_date_utc':'2016-11-03','post_date_utc':'2016-11-15','relative_orbit':52,'pass':'ASCENDING',
        'feature_names':NAMES,'records':records,'region_count':8,'sampled_area_km2':327.68,
        'download_manifest_file':str(DOWN/'manifest.json'),'download_manifest_sha256':sha(DOWN/'manifest.json'),
        'inventory_source':reference_report,'inventory_interpretation':'Reused raw author Kaikora2016 polygons; not verified new/reactivated slide truth or road blockage',
        'verified_negative_pixels':0,'train_test_split_frozen':False,'training_ready':False,'training_performed':False,
        'feature_processing':'3x3 linear-power mean, then5x5 local dB std; radius3 boundary excluded; no radiometric terrain flattening',
        'limitations':['Local active angular geometry masks only; no passive ray tracing or geometric coregistration verification',
                       'SRTM about30m terrain; 10m grids do not resolve every small obstacle',
                       'Inventory absence is unknown; candidates require time/extent audit',
                       'Sampled SH1 subset only, not full project road network or historical topology',
                       'Prior sample regions and reference distributions known; not blind event-wide validation',
                       'Optical imagery/BBUnet outputs not used as RF features; only reused grid, raw reference, and road masks',
                       'Single-date pre is the current main baseline; retained pre median is a separately labelled alternative']}
(OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'latest_prepared_regions.json').write_text(json.dumps({'directory':str(OUT),'training_ready':False},ensure_ascii=False,indent=2),encoding='utf-8')

os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,4,figsize=(15,8),layout='constrained')
for ax,r in zip(axes.flat,records):
    with rasterio.open(r['raw_file']) as src: a=src.read()
    with rasterio.open(OUT/r['id']/'screened_feature_mask.tif') as src: mask=src.read(1)==1
    with rasterio.open(OUT/r['id']/'road100m_context_mask.tif') as src: road=src.read(1)==1
    ax.imshow(np.where(mask,a[2],np.nan),cmap='RdBu_r',vmin=-6,vmax=6)
    ax.contour(road,levels=[.5],colors=['black'],linewidths=.6)
    ax.set_title(f"{r['id']} | data {r['data_coverage_pct']:.2f}%\nNumerical + active geometry {r['screened_feature_coverage_pct']:.2f}%",fontsize=10)
    ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Sentinel-1 VV Nov3-Nov15 change (dB), NOT landslide predictions\nBlack: SH1 100m context; white: excluded. Active geometry screening is approximate.',fontsize=12)
fig.savefig(OUT/'VV_regions_QA.png',dpi=140);plt.close(fig)
print('PREPARED_DIRECTORY='+str(OUT),flush=True)
