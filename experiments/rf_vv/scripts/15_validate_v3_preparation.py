"""Independent GDAL/pyogrio vector read and artifact validation."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
import rasterio,pyogrio
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box,mapping
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parent
OUT=Path(json.loads((ROOT/'latest_v3_reference_preparation.json').read_text(encoding='utf-8'))['directory'])
PREP=Path(json.loads((ROOT/'latest_prepared_regions.json').read_text(encoding='utf-8'))['directory'])
prepared=json.loads((PREP/'manifest.json').read_text(encoding='utf-8'))
report=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
reviews=json.loads((OUT/'coarse_visual_review_all54.json').read_text(encoding='utf-8'))
matrix=json.loads((OUT/'VV_feature_point_proposals_NOT_APPROVED.json').read_text(encoding='utf-8'))
split=json.loads((ROOT/'spatial_split_v1.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with rasterio.open(p) as s:return s.read(),(s.shape,s.transform,s.crs),s.nodata
assert sha(PREP/'manifest.json')==report['prepared_manifest_sha256']
assert sha(ROOT/'spatial_split_v1.json')==report['split_sha256']
assert len(reviews)==54 and len(set(s['id'] for s in reviews))==54
assert all(s['review_status']=='coarse_visual_screened_NOT_final_truth' and s['final_label'] is None for s in reviews)
vectorinfo={}
for k,v in report['source_metadata'].items():
    assert sha(v['file'])==v['sha256']
    info=pyogrio.read_info(v['file']);assert info['features']==v['records_total'] and info['crs']=='EPSG:2193'
    vectorinfo[k]={'gdal_count':info['features'],'crs':info['crs']}
verified=[]; diffs=[]
pointfile=ROOT/'official_reference_points/GNS_V2_source_centroids_2193.json'
pointdata=json.loads(pointfile.read_text(encoding='utf-8'))['features']
audit=Path(json.loads((ROOT/'latest_label_alignment_audit.json').read_text(encoding='utf-8'))['directory'])
warning_crosschecks=[]
for rec in report['regions']:
    rid=rec['id']; dest=OUT/rid
    baseline=next(r for r in prepared['records'] if r['id']==rid)
    assert sha(baseline['raw_file'])==rec['raw_sha256']==baseline['raw_sha256']
    for original in baseline['outputs']:assert sha(original['file'])==original['sha256']
    for output in rec['outputs']:assert sha(output['file'])==output['sha256']
    f,grid,_=read(PREP/rid/'VV_feature_candidates.tif'); h,w=grid[0];t=grid[1]
    bbox=(t.c,t.f-h*10,t.c+w*10,t.f)
    reread=[]
    for kind,filename in [('source','v3_source_reference.tif'),('debris','v3_debris_reference.tif')]:
        gd=pyogrio.read_dataframe(report['source_metadata'][kind]['file'],bbox=bbox)
        geoms=[g if g.is_valid else g.buffer(0) for g in gd.geometry if g is not None and not g.is_empty]
        shapes=[(mapping(g),1) for g in geoms]
        mask=rasterize(shapes,out_shape=(h,w),transform=t,dtype='uint8') if shapes else np.zeros((h,w),dtype='uint8')
        actual,g,n=read(dest/filename);assert g==grid and n is None
        different=int(np.count_nonzero(mask!=actual[0]));diffs.append({'id':rid,'kind':kind,'gdal_vs_pyshp_pixel_differences':different})
        assert different==0,(rid,kind,different)
        reread.extend(geoms)
    sub=rasterize([(mapping(g),1) for g in reread],out_shape=(h*4,w*4),transform=t*Affine.scale(.25,.25),dtype='uint8') if reread else np.zeros((h*4,w*4),dtype='uint8')
    fraction=sub.reshape(h,4,w,4).mean(axis=(1,3)).astype('float32')
    actual,g,n=read(dest/'v3_union_subpixel_fraction.tif');assert g==grid and np.array_equal(actual[0],fraction)
    labels,g,n=read(dest/'reference_positive_UNKNOWN_background.tif');assert g==grid and n==255 and np.isin(labels,[1,255]).all()
    screened,_,_=read(PREP/rid/'screened_feature_mask.tif')
    uncertain,_,_=read(dest/'v3_ambiguous_or_cross_role_mask.tif')
    assert np.array_equal(labels[0]==1,(fraction>=.75)&(screened[0]==1)&(uncertain[0]==0))
    weak,g,n=read(dest/'weak_labels_PROPOSAL_NOT_APPROVED.tif');assert g==grid and n==255 and np.isin(weak,[0,1,255]).all()
    negative,g,_=read(dest/'weak_negative_proposal_mask.tif');assert np.array_equal(weak[0]==0,negative[0]==1)
    assert np.array_equal(weak[0]==1,labels[0]==1)
    guard,_,_=read(dest/'v3_all_polygon_buffer100m.tif');assert not ((negative[0]==1)&(guard[0]==1)).any()
    proposal_domain,_,_=read(dest/'negative_review_candidates_V3_NOT_LABELS.tif')
    assert not ((negative[0]==1)&(proposal_domain[0]!=1)).any()
    assert int((negative[0]==1).sum())==rec['negative_proposal_pixels']
    for row in [r for r in matrix['records'] if r['region_id']==rid]:
        y,x=row['row'],row['col'];assert row['role']==split['assignments'][rid]
        assert row['reference_label_proposal']==weak[0,y,x]
        assert np.array_equal(np.array(row['features'],dtype='float32'),f[:,y,x])
        assert len(row['features'])==11 and np.isfinite(f[:,y,x]).all()
    verified.append({'id':rid,'checks':'PASS'})
    # Supplemental position check only: this does not turn centers into extent truth.
    oldaudit=json.loads((audit/rid/'audit.json').read_text(encoding='utf-8'))
    expanded=(bbox[0]-1100,bbox[1]-1100,bbox[2]+1100,bbox[3]+1100)
    gd=pyogrio.read_dataframe(report['source_metadata']['source']['file'],bbox=expanded)
    source_geoms=[g if g.is_valid else g.buffer(0) for g in gd.geometry if g is not None and not g.is_empty]
    tree=STRtree(source_geoms); checked=[]
    for warning in oldaudit['centroid_warnings']:
        f=pointdata[warning['point_index']];p=Point(f['geometry']['x'],f['geometry']['y'])
        distance=p.distance(source_geoms[int(tree.nearest(p))]) if source_geoms else None
        checked.append({'gns_v2_objectid':f['attributes']['objectid'],'distance_to_v3_source_m':distance,
                        'within100m_v3_source':distance is not None and distance<=100})
    warning_crosschecks.append({'id':rid,'old_distance_warning_points':len(checked),
                              'now_within100m_v3_source':sum(c['within100m_v3_source'] for c in checked),
                              'records':checked})
with np.load(OUT/'VV_feature_point_proposals_NOT_APPROVED.npz',allow_pickle=False) as z:
    assert len(z['X'])==len(matrix['records']) and z['X'].shape[1]==11
    assert np.array_equal(z['X'],np.array([r['features'] for r in matrix['records']],dtype='float32'))
    assert np.array_equal(z['y_reference_proposal'],[r['reference_label_proposal'] for r in matrix['records']])
    assert np.array_equal(z['role'],[r['role'] for r in matrix['records']])
    assert list(z['feature_names'])==matrix['feature_names']
object_roles={}
for row in matrix['records']:
    for oid in row['object_ids']:object_roles.setdefault(oid,set()).add(row['role'])
assert all(len(roles)==1 for roles in object_roles.values()),'Object leakage across fixed roles'
result={'status':'GDAL_independent_raster_fraction_and_artifact_checks_PASS_NOT_ground_truth_approval',
        'regions':verified,'vectorinfo':vectorinfo,'gdal_vs_pyshp':diffs,'all_original_VV_features_unchanged':True,
        'class_counts':dict(Counter(r['reference_label_proposal'] for r in matrix['records'])),
        'role_class_counts':{f'{role}_class{label}':sum(r['role']==role and r['reference_label_proposal']==label for r in matrix['records'])
                             for role in ['train','validation','test'] for label in [0,1]},
        'sampled_positive_reference_object_ids':len(object_roles),'cross_role_object_leakage':False,
        'optical_predictors_used':False,'training':False,'formal_training_ready':False,
        'geometry_note':'Pyshp emitted an orphan-ring warning; GDAL independently reproduced every region mask and4x4fraction without disagreement. Agreement does not certify true boundary accuracy.'}
result['warning_crosscheck_summary']={
    'old_warning_points':sum(r['old_distance_warning_points'] for r in warning_crosschecks),
    'now_within100m_v3_source':sum(r['now_within100m_v3_source'] for r in warning_crosschecks),
    'meaning':'Position/extent consistency check, NOT recall, model detection, or resolved observation dates'}
(OUT/'V2_center_old_warnings_vs_V3_sources.json').write_text(json.dumps(warning_crosschecks,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
