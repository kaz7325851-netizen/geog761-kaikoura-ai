"""Prepare a clearly provisional VV-only data pack from audited local inputs.

Uses the actual three-date pre median, NOT a fabricated Nov3 single-date image.
No RF fitting, no verified negative labels, no full pre/post visibility claim.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, binary_erosion
from rasterio.features import rasterize
from shapely.geometry import shape, mapping
from shapely.ops import transform, unary_union
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'sar_lra/kaikoura_exploration'
RAW = BASE / 'downloads/20261005_182555/ASCENDING_REL52.tif'
META = BASE / 'earthengine_metadata.json'
GEOM_MANIFEST = BASE / 'label_cleaning/download_manifest.json'
INVENTORY = BASE / 'reference/pilot_inventory.geojson'
ROADS = ROOT.parent / 'sentinel1/study_route_lines.geojson'
OUT = ROOT / 'prepared_inputs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True, exist_ok=False)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, values, descriptions, profile, nodata):
    if values.ndim == 2:
        values = values[None]
    p = OUT / name
    cfg = dict(profile)
    cfg.update(driver='GTiff', count=len(values), dtype=str(values.dtype), nodata=nodata,
               compress='deflate', predictor=3 if values.dtype.kind == 'f' else 2)
    with rasterio.open(p, 'w', **cfg) as dst:
        dst.write(values)
        for i, desc in enumerate(descriptions, 1):
            dst.set_band_description(i, desc)
        dst.update_tags(training_ready='false', model_prediction='false',
                        pre_source='Sep16 Oct10 Nov3 median, NOT Nov3 single date')
    with rasterio.open(p) as src:
        assert src.transform == profile['transform'] and src.crs == profile['crs']
        assert np.array_equal(src.read(), values, equal_nan=True)
        assert list(src.descriptions) == descriptions
        assert src.nodata == nodata
    return {'file': str(p), 'sha256': sha(p), 'bands': descriptions,
            'nodata': nodata, 'reread_exact': True}

metadata = json.loads(META.read_text(encoding='utf-8'))
assert metadata['band_order'][7] == 'recentPreVV'
assert all('20161115' in sid for sid in metadata['post_scene_ids'])
assert sorted({sid.split('_')[4][:8] for sid in metadata['recent_vv_scene_ids']}) == ['20160916', '20161010', '20161103']
geo_manifest = json.loads(GEOM_MANIFEST.read_text(encoding='utf-8'))
geom_path = Path(geo_manifest['file'])
assert sha(geom_path) == geo_manifest['sha256']
with rasterio.open(RAW) as src:
    original = src.read()
    profile = src.profile.copy()
    grid = (src.shape, src.crs, src.transform)
    assert src.count == 10 and src.crs.to_epsg() == 2193
with rasterio.open(geom_path) as src:
    assert (src.shape, src.crs, src.transform) == grid
    geo = src.read()
pre, post = original[7], original[0]
valid = np.isfinite(pre) & np.isfinite(post) & (pre != -9999) & (post != -9999) & (original[9] == 1)
raw_delta = post - pre
raw_pack = np.stack([pre, post, raw_delta]).astype('float32')
raw_pack[:, ~valid] = -9999
assert np.allclose(raw_pack[2, valid], raw_pack[1, valid]-raw_pack[0, valid], atol=2e-5)

# Use only Nov15 geometry here. Aug23 pre geometry cannot certify recent preVV.
post_known, post_bad = [], []
for scene_index in [1, 2]:
    scene = geo_manifest['metadata']['scenes'][scene_index]
    assert '20161115' in scene['id']
    angle, alpha, layover_ok, shadow_ok, coverage = geo[3+5*scene_index:8+5*scene_index]
    known = (coverage == 1) & np.isfinite(alpha) & (alpha != -9999) & (angle != -9999)
    expected = np.degrees(np.arctan(np.tan(np.radians(geo[1])) * np.cos(np.radians(scene['heading_deg']-geo[2]))))
    assert np.max(np.abs(expected[known]-alpha[known])) < .001
    assert np.array_equal(layover_ok[known].astype(bool), alpha[known] < angle[known])
    assert np.array_equal(shadow_ok[known].astype(bool), alpha[known] > angle[known]-90)
    post_known.append(known)
    post_bad.append(known & ((layover_ok == 0) | (shadow_ok == 0)))
post_visible = (post_known[0] | post_known[1]) & ~(post_bad[0] | post_bad[1])
terrain_valid = np.isfinite(geo[:3]).all(axis=0) & (geo[:3] != -9999).all(axis=0)

def mean_filter_db(db, mask, size):
    power = np.where(mask, np.power(10., np.where(mask, db, 0)/10), 0)
    count = uniform_filter(mask.astype('float64'), size=size, mode='constant', cval=0)
    total = uniform_filter(power.astype('float64'), size=size, mode='constant', cval=0)
    ratio = np.divide(total, count, out=np.ones_like(total), where=count>0)
    return 10*np.log10(np.maximum(ratio, 1e-12))

def std_filter_db(db, mask, size):
    count = uniform_filter(mask.astype('float64'), size=size, mode='constant', cval=0)
    mean = np.divide(uniform_filter(np.where(mask, db, 0).astype('float64'), size=size, mode='constant'), count,
                     out=np.zeros_like(count), where=count>0)
    second = np.divide(uniform_filter(np.where(mask, db*db, 0).astype('float64'), size=size, mode='constant'), count,
                       out=np.zeros_like(count), where=count>0)
    return np.sqrt(np.maximum(second - mean*mean, 0))

filtered_pre = mean_filter_db(pre, valid, 3)
filtered_post = mean_filter_db(post, valid, 3)
delta = filtered_post - filtered_pre
pre_std = std_filter_db(filtered_pre, valid, 5)
post_std = std_filter_db(filtered_post, valid, 5)
# Radius3 covers the radius1 filter followed by radius2 texture neighborhood.
feature_valid = binary_erosion(valid, structure=np.ones((7,7)), border_value=0) & terrain_valid
names = ['VV_pre_median_filtered_db','VV_post_filtered_db','dVV_filtered_db',
         'abs_dVV_filtered_db','pre_local_std_5x5_db','post_local_std_5x5_db',
         'delta_local_std_5x5_db','elevation_m','slope_deg','aspect_sin','aspect_cos']
features = np.stack([filtered_pre, filtered_post, delta, np.abs(delta), pre_std, post_std,
                     post_std-pre_std, geo[0], geo[1],
                     np.sin(np.radians(geo[2])), np.cos(np.radians(geo[2]))]).astype('float32')
assert np.isfinite(features[:, feature_valid]).all()
features[:, ~feature_valid] = -9999

forward = Transformer.from_crs(4326,2193,always_xy=True).transform
inventory = json.loads(INVENTORY.read_text(encoding='utf-8'))
geometries = [transform(forward,shape(f['geometry'])) for f in inventory['features']]
ref = rasterize([(g,1) for g in geometries], out_shape=grid[0], transform=grid[2],
                fill=0, dtype='uint8', all_touched=False)
roads = json.loads(ROADS.read_text(encoding='utf-8'))
road_geom = unary_union([transform(forward,shape(f['geometry'])) for f in roads['features']])
near_road = rasterize([(road_geom.buffer(100),1)], out_shape=grid[0], transform=grid[2],
                     fill=0, dtype='uint8', all_touched=False)
candidate_labels = np.full(grid[0],255,dtype='uint8')
candidate_labels[(ref==1) & feature_valid & post_visible] = 1
assert not np.any(candidate_labels==0), 'Do not invent verified negative labels'
records = []
records.append(save('VV_preMedian_post_raw.tif',raw_pack,['VV_pre_median_db','VV_post_db','dVV_raw_db'],profile,-9999))
records.append(save('VV_feature_candidates.tif',features,names,profile,-9999))
records.append(save('data_validity_mask.tif',valid.astype('uint8'),['both_dates_data_present'],profile,None))
records.append(save('feature_numerical_validity_mask.tif',feature_valid.astype('uint8'),['numerical_features_valid_NOT_full_visibility'],profile,None))
records.append(save('post_geometry_approx_mask.tif',post_visible.astype('uint8'),['post_local_active_geometry_approx'],profile,None))
records.append(save('inventory_reference_only.tif',ref,['inventory_candidate_NOT_verified_truth'],profile,None))
records.append(save('candidate_labels_NOT_TRAIN_READY.tif',candidate_labels,['1_inventory_candidate_255_unknown_NO_verified_negatives'],profile,255))
records.append(save('road100m_context_mask.tif',near_road,['within100m_current_OSM_roads'],profile,None))

report = {'created_utc':datetime.now(timezone.utc).isoformat(), 'directory':str(OUT),
          'status':'local_provisional_pack_complete; new_single_date_exports_blocked_by_GEE_client_initialization',
          'pre_dates_utc':['2016-09-16','2016-10-10','2016-11-03'], 'pre_aggregation':'GEE dB-space median',
          'post_date_utc':'2016-11-15', 'orbit':52, 'pass':'ASCENDING', 'platform':'A',
          'shape':list(grid[0]),'crs':str(grid[1]),'transform':list(grid[2])[:6],
          'source_hashes':{str(p):sha(p) for p in [RAW,META,geom_path,GEOM_MANIFEST,INVENTORY,ROADS]},
          'outputs':records,'pixels':int(valid.size),'data_valid_pixels':int(valid.sum()),
          'feature_numerical_valid_pixels':int(feature_valid.sum()),
          'post_geometry_approx_visible_pixels':int((valid&post_visible).sum()),
          'candidate_inventory_pixels':int(ref.sum()),'candidate_label_pixels':int((candidate_labels==1).sum()),
          'verified_negative_pixels':0,'candidate_inventory_objects':len(geometries),
          'road100m_pixels':int(near_road.sum()),'road100m_data_valid_pixels':int((near_road.astype(bool)&valid).sum()),
          'speckle_processing':'3x3 box mean in linear power; experimental baseline, may blur small landslides',
          'texture':'5x5 local standard deviation on filtered dB; NOT GLCM; boundary radius3 excluded',
          'radiometric_terrain_flattening':False,'recent_pre_geometry_verified':False,
          'training_ready':False,'training_performed':False,'prediction_performed':False,
          'limitations':['Pre is actual3date median, NOT downloaded Nov3 single date',
                         'Post geometry active approximation only; recent pre geometry unavailable locally',
                         'No passive shadow/layover ray tracing; numerical coverage does not establish visibility',
                         'Inventory candidates require event/time/extent audit; outside inventory is unknown',
                         'This pilot has no independently verified negatives or spatial holdout',
                         '100m buffer denotes proximity, NOT a definition of blockage or a road label',
                         'Current OSM not audited2016 topology; no road-passability labels']}
(OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'latest_prepared_inputs.json').write_text(json.dumps({'directory':str(OUT),'training_ready':False},ensure_ascii=False,indent=2),encoding='utf-8')

os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,4,figsize=(16,4),layout='constrained')
axes[0].imshow(np.where(valid,pre,np.nan),cmap='gray',vmin=-25,vmax=0);axes[0].set_title('PreVV Sep16/Oct10/Nov3 median')
axes[1].imshow(np.where(valid,post,np.nan),cmap='gray',vmin=-25,vmax=0);axes[1].set_title('PostVV Nov15')
axes[2].imshow(np.where(valid,raw_delta,np.nan),cmap='RdBu_r',vmin=-6,vmax=6);axes[2].set_title('Post minus pre median (dB)')
for ax in axes[:3]:
    ax.contour(near_road,levels=[.5],colors=['cyan'],linewidths=.4)
classes=np.zeros(grid[0],dtype='uint8');classes[~valid]=1;classes[valid&~post_visible]=2;classes[candidate_labels==1]=3
from matplotlib.colors import ListedColormap
axes[3].imshow(classes,cmap=ListedColormap(['#dddddd','#222222','#aa77bb','#e1a72e']),vmin=0,vmax=3)
axes[3].set_title('Black: missing; purple: post geometry\nGold: inventory candidates, NOT truth')
for ax in axes:ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('VV-only provisional local input pack. Pre geometry and negative labels pending; NO trained model.',fontsize=11)
fig.savefig(OUT/'local_input_QA.png',dpi=140);plt.close(fig)
print(json.dumps({k:report[k] for k in ['directory','data_valid_pixels','feature_numerical_valid_pixels','post_geometry_approx_visible_pixels','candidate_label_pixels','verified_negative_pixels','training_ready']},ensure_ascii=False,indent=2))
