"""Stage authoritative event inventory and explicitly weak, reviewed label proposals.

VV-only predictor arrays. Optical images are independent label-review context.
No training, no road-safety labels, no automatic background-as-negative rule.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json, hashlib, os
import numpy as np
import rasterio, shapefile
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape, box, mapping, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
from pyproj import CRS

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'official_v3_acquisition/components'
AUDIT=Path(json.loads((ROOT/'latest_label_alignment_audit.json').read_text(encoding='utf-8'))['directory'])
PREP=Path(json.loads((ROOT/'latest_prepared_regions.json').read_text(encoding='utf-8'))['directory'])
prepared=json.loads((PREP/'manifest.json').read_text(encoding='utf-8'))
design=json.loads((ROOT/'frozen_data_design.json').read_text(encoding='utf-8'))
split=json.loads((ROOT/'spatial_split_v1.json').read_text(encoding='utf-8'))
samples=json.loads((AUDIT/'review_samples_UNREVIEWED.json').read_text(encoding='utf-8'))
download=json.loads((ROOT/'official_v3_acquisition/download_manifest.json').read_text(encoding='utf-8'))
OUT=ROOT/'v3_reference_preparation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True,exist_ok=False)
OPT=ROOT.parent/'bbunet/expanded_test/20261007T210924Z'
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):
    with rasterio.open(p) as s:return s.read(),s.profile.copy()
def save(p,a,profile,description,nodata=None):
    cfg=dict(profile);cfg.update(count=1,dtype=str(a.dtype),nodata=nodata,compress='deflate')
    with rasterio.open(p,'w',**cfg) as s:
        s.write(a,1);s.set_band_description(1,description)
        s.update_tags(training='false',formal_ground_truth='false',road_passability='false')
    reread,pr=read(p);assert np.array_equal(reread[0],a) and pr['transform']==profile['transform']
    return {'file':str(p),'sha256':sha(p),'meaning':description}
for r in download['records']:
    assert sha(r['file'])==r['sha256']
    if r['name']!='manifest-sha512.txt':assert r['publisher_sha512_verified']
sites={s['id']:box(*s['bounds']) for s in design['sites']}
whole=unary_union(list(sites.values())).buffer(1100)
roles={k:unary_union([sites[s] for s in sites if split['assignments'][s]==k]) for k in ['train','validation','test']}
vectors={};vector_metadata={};cross=[]
for kind,stem,idfield,expected in [('source','GNS_2016KaikouraEQ_LandslideSourceArea_Version3','Source_ID',31623),
                                   ('debris','GNS_2016KaikouraEQ_LandslideDebrisTrails_Version3','Debris_ID',26559)]:
    p=SOURCE/(stem+'.shp');assert CRS.from_wkt(p.with_suffix('.prj').read_text()).to_epsg()==2193
    reader=shapefile.Reader(str(p),encoding='utf-8');assert len(reader)==expected
    geometries=[];properties=[];repairs=0;ambiguous_total=0
    for sr in reader.iterShapeRecords():
        props=sr.record.as_dict()
        ambiguous=kind=='source' and (props.get('Reference')=='Cao et al. 2023' or
                     'incipient' in props.get('Note','').lower() or 'pre-existing' in props.get('Note','').lower())
        ambiguous_total+=int(ambiguous)
        if not box(*sr.shape.bbox).intersects(whole.envelope):continue
        g=shape(sr.shape.__geo_interface__)
        if not g.is_valid:g=g.buffer(0);repairs+=1
        if g.is_empty or not g.intersects(whole):continue
        touched_roles=[k for k,p in roles.items() if g.intersects(p)]
        quarantine=len(touched_roles)>1
        if quarantine:cross.append({'kind':kind,'id':props[idfield],'roles':touched_roles})
        props.update(kind=kind,object_id=int(props[idfield]),ambiguous_for_rapid_scar=ambiguous,
                     quarantine_cross_role=quarantine)
        geometries.append(g);properties.append(props)
    vectors[kind]={'geometries':geometries,'properties':properties,'tree':STRtree(geometries)}
    vector_metadata[kind]={'file':str(p),'sha256':sha(p),'records_total':len(reader),
                           'records_near_regions':len(geometries),'repaired_local_geometries':repairs,
                           'ambiguous_source_records_eventwide':ambiguous_total,'fields':[x[0] for x in reader.fields[1:]]}
    clipped=[{'type':'Feature','properties':props,'geometry':mapping(g.intersection(whole))}
             for g,props in zip(geometries,properties)]
    dump(OUT/f'v3_{kind}_clipped_2193.geojson',{'type':'FeatureCollection','crs':{'type':'name','properties':{'name':'EPSG:2193'}},'features':clipped})

# Explicit human visual-screen categories from the 14 context pages, not VV-score selection.
# "stable" means coarse vegetation appearance, never certified no landslide of any size.
stable={
 'SH1-EXP-01':['R04','R05'], 'SH1-EXP-02':['R01','R03','R04','R05'],
 'SH1-EXP-03':['R01','R02','R03','R04'], 'SH1-EXP-04':['R03','R04','R06'],
 'SH1-EXP-05':['R01','R04','R05','R06'], 'SH1-EXP-06':['R03','R04','R05'],
 'SH1-EXP-07':['R05','R06'], 'PILOT-ANCHOR':['R02','R04','R06']}
stable={f'{rid}-{s}' for rid,ids in stable.items() for s in ids}
pre_existing={'SH1-EXP-02-R06','SH1-EXP-07-R07'}
later_change={'SH1-EXP-04-R07','SH1-EXP-04-R08','SH1-EXP-05-R07',
              'SH1-EXP-06-R06','PILOT-ANCHOR-R07','PILOT-ANCHOR-R08'}
cloud={'SH1-EXP-01-R06'}
reviews=[];records=[];plotdata=[];pack=[]
for region_index,r in enumerate(prepared['records']):
    rid=r['id']; dest=OUT/rid;dest.mkdir(); region=sites[rid]
    raw,profile=read(r['raw_file']);shape2=raw.shape[1:];grid=profile['transform']
    feature,_=read(PREP/rid/'VV_feature_candidates.tif')
    screen,_=read(PREP/rid/'screened_feature_mask.tif');screen=screen[0]==1
    old,_=read(PREP/rid/'inventory_reference_only.tif');old=old[0]==1
    road,_=read(PREP/rid/'road100m_context_mask.tif');road=road[0]==1
    def raster(geoms,sub=1):
        values=[(mapping(g),1) for g in geoms if not g.is_empty]
        return rasterize(values,out_shape=(shape2[0]*sub,shape2[1]*sub),
                         transform=grid*Affine.scale(1/sub,1/sub),dtype='uint8').astype(bool) if values else np.zeros((shape2[0]*sub,shape2[1]*sub),bool)
    nearby={k:[int(i) for i in v['tree'].query(region.buffer(100))] for k,v in vectors.items()}
    direct={k:[i for i in nearby[k] if vectors[k]['geometries'][i].intersects(region)] for k in vectors}
    source_geoms=[vectors['source']['geometries'][i] for i in direct['source']]
    debris_geoms=[vectors['debris']['geometries'][i] for i in direct['debris']]
    srcmask=raster(source_geoms);debrmask=raster(debris_geoms);union=srcmask|debrmask
    frac=raster(source_geoms+debris_geoms,4).reshape(shape2[0],4,shape2[1],4).mean(axis=(1,3)).astype('float32')
    uncertain=[]
    for k in vectors:
        for i in nearby[k]:
            p=vectors[k]['properties'][i]
            if p['ambiguous_for_rapid_scar'] or p['quarantine_cross_role']:uncertain.append(vectors[k]['geometries'][i].buffer(20))
    unknown_mask=raster(uncertain)
    positive=(frac>=.75)&screen&~unknown_mask
    # Exclude ALL official polygons, including ambiguous source/debris records, from negatives.
    guard=raster([vectors[k]['geometries'][i].buffer(100) for k in vectors for i in nearby[k]])
    oldguard,_=read(AUDIT/rid/'inventory_buffer100m.tif')
    pointguard,_=read(AUDIT/rid/'gns_source_point_buffer100m.tif')
    negative_candidates=screen&(raw[7]>5)&~guard&~oldguard[0].astype(bool)&~pointguard[0].astype(bool)
    labels=np.full(shape2,255,dtype='uint8');labels[positive]=1
    negative_proposals=np.zeros(shape2,bool)
    q=[]
    for kind in ['pre','post']:
        scl,_=read(OPT/rid/f'{kind}_SCL.tif');q.append(np.isin(scl[0],[4,5]))
    for s in [s for s in samples if s['region_id']==rid]:
        y,x=s['row'],s['col'];context=(slice(y-1,y+2),slice(x-1,x+2))
        point=Point(s['x_nztm'],s['y_nztm'])
        matched={k:[vectors[k]['properties'][i]['object_id'] for i in nearby[k] if vectors[k]['geometries'][i].covers(point)] for k in vectors}
        if s['id'] in stable: observation='Vegetation/grass appearance persists at coarse optical scale'
        elif s['id'] in pre_existing:observation='Bare/disturbed area already visible in pre optical image; new/reactivation unresolved'
        elif s['id'] in later_change:observation='Clear later optical scar/debris change in mapped context; exact Nov15 timing unresolved'
        elif s['id'] in cloud:observation='Post optical context obscured by cloud/haze'
        else:observation='Mixed coastal/road-edge, shadow, bare-ground, or otherwise ambiguous coarse context'
        jointclear=bool(q[0][context].all() and q[1][context].all())
        eligible=s['id'] in stable and s['kind'].startswith('negative') and jointclear and bool(negative_candidates[context].all())
        if eligible:negative_proposals[context]=True
        reviews.append({**s,'review_status':'coarse_visual_screened_NOT_final_truth','observation':observation,
                        'matched_v3_source_ids':matched['source'],'matched_v3_debris_ids':matched['debris'],
                        'inside_v3_footprint':bool(union[y,x]),'joint_SCL_clear_3x3':jointclear,
                        'outside_all_100m_guards_3x3':bool(negative_candidates[context].all()),
                        'eligible_for_weak_negative_proposal':eligible,'final_label':None})
    # Separate clearly named proposal raster. Not replace formal labels with tentative zeros.
    weak=labels.copy();weak[negative_proposals]=0
    outputs=[]
    for filename,a,meaning,nodata in [
        ('v3_source_reference.tif',srcmask.astype('uint8'),'GNS_V3_event_source_footprints_not_per_date_truth',None),
        ('v3_debris_reference.tif',debrmask.astype('uint8'),'GNS_V3_event_debris_footprints_not_blockage',None),
        ('v3_union_subpixel_fraction.tif',frac,'4x4_subsample_coverage_approx_not_true_accuracy',None),
        ('v3_ambiguous_or_cross_role_mask.tif',unknown_mask.astype('uint8'),'incipient_existing_or_crossrole_uncertainty',None),
        ('v3_all_polygon_buffer100m.tif',guard.astype('uint8'),'heuristic_guard_not_complete_omission_proof',None),
        ('negative_review_candidates_V3_NOT_LABELS.tif',negative_candidates.astype('uint8'),'candidate_domain_only_not_class0',None),
        ('reference_positive_UNKNOWN_background.tif',labels,'1_official_event_reference_255_unknown_NO0',255),
        ('weak_negative_proposal_mask.tif',negative_proposals.astype('uint8'),'coarse_optical_screened_proposal_not_verified_no_slide',None),
        ('weak_labels_PROPOSAL_NOT_APPROVED.tif',weak,'0_coarse_optical_proposal_1_event_inventory_255_unknown_NOT_Nov15_truth',255)]:
        outputs.append(save(dest/filename,a,profile,meaning,nodata))
    # VV-only staged point matrix; explicitly proposed/weak labels, not approved training data.
    yy,xx=np.where(negative_proposals)
    selected=[(int(y),int(x),0) for y,x in zip(yy,xx)]
    rng=np.random.default_rng(761+region_index)
    candidates=np.argwhere(positive)
    limit=min(len(candidates),max(36,len(selected)*2))
    if limit:
        chosen=candidates[rng.choice(len(candidates),limit,replace=False)]
        selected += [(int(y),int(x),1) for y,x in chosen]
    for y,x,label in selected:
        assert np.isfinite(feature[:,y,x]).all() and (feature[:,y,x]!=-9999).all()
        ids=[f'{k}:{vectors[k]["properties"][i]["object_id"]}' for k in vectors for i in nearby[k] if vectors[k]['geometries'][i].covers(Point(grid.c+10*(x+.5),grid.f-10*(y+.5)))]
        pack.append({'region_id':rid,'role':split['assignments'][rid],'row':y,'col':x,
                     'reference_label_proposal':label,'object_ids':ids,'features':feature[:,y,x].tolist(),
                     'negative_review_group_ids':[s['id'] for s in samples if s['region_id']==rid and label==0 and abs(y-s['row'])<=1 and abs(x-s['col'])<=1]})
    rec={'id':rid,'role':split['assignments'][rid],'source_objects_intersecting':len(direct['source']),
         'debris_objects_intersecting':len(direct['debris']),'old_inventory_pixels':int(old.sum()),
         'v3_union_pixels':int(union.sum()),'v3_union_pixels_not_old':int((union&~old).sum()),
         'old_pixels_outside_v3':int((old&~union).sum()),'event_reference_positive_pixels':int(positive.sum()),
         'event_reference_positive_pixels_road100m':int((positive&road).sum()),
         'negative_proposal_pixels':int(negative_proposals.sum()),
         'coarse_review_negative_groups':sum(s['eligible_for_weak_negative_proposal'] for s in reviews if s['region_id']==rid),
         'verified_no_slide_or_road_safe_pixels':0,'training_ready':False,'outputs':outputs,
         'raw_sha256':sha(r['raw_file']),'feature_sha256':sha(PREP/rid/'VV_feature_candidates.tif')}
    dump(dest/'manifest.json',rec);records.append(rec);plotdata.append((rid,old,union,road,negative_proposals))
    print(json.dumps({k:rec[k] for k in ['id','v3_union_pixels','v3_union_pixels_not_old','negative_proposal_pixels']},ensure_ascii=False),flush=True)
dump(OUT/'coarse_visual_review_all54.json',reviews)
dump(OUT/'VV_feature_point_proposals_NOT_APPROVED.json',{'feature_names':prepared['feature_names'],
     'feature_sensor':'Sentinel1_VV_plus_terrain_NO_optical_predictors','status':'WEAK_PROPOSALS_NOT_APPROVED',
     'records':pack,'training':False,'sampling_rule':'All approved-for-proposal negative3x3 patches; random official positive pixels capped at twice negative pixels or36 perregion; seed761+regionindex',
     'warnings':['Artificial class ratio is not real prevalence','Pixels within each patch/object are correlated',
                 'Use frozen region split, no pixel random split','Proposals require explicit weak-reference interpretation']})
np.savez_compressed(OUT/'VV_feature_point_proposals_NOT_APPROVED.npz',
    X=np.array([s['features'] for s in pack],dtype='float32'),
    y_reference_proposal=np.array([s['reference_label_proposal'] for s in pack],dtype='uint8'),
    region=np.array([s['region_id'] for s in pack]),role=np.array([s['role'] for s in pack]),
    row=np.array([s['row'] for s in pack],dtype='int32'),col=np.array([s['col'] for s in pack],dtype='int32'),
    feature_names=np.array(prepared['feature_names']))
report={'directory':str(OUT),'status':'official_V3_staged_and54_visually_screened_weak_labels_PROPOSAL_only',
        'source_download_manifest_sha256':sha(ROOT/'official_v3_acquisition/download_manifest.json'),
        'prepared_manifest_sha256':sha(PREP/'manifest.json'),'split_sha256':sha(ROOT/'spatial_split_v1.json'),
        'source_metadata':vector_metadata,'quarantined_cross_role_polygons':cross,
        'regions':records,'reviewed_coarse_contexts':len(reviews),'weak_negative_proposal_groups':sum(s['eligible_for_weak_negative_proposal'] for s in reviews),
        'point_matrix_class_counts':dict(Counter(s['reference_label_proposal'] for s in pack)),
        'training_ready':False,'training':False,'truth_at_2016_11_15':False,'road_passability_truth':False,
        'policy':['Official inventory includes earthquake/aftershock footprints; exact Nov15 dates are unknown',
                  'No 20m uniform shrink;75% estimated pixel coverage is an experimental boundary rule',
                  'Existing/incipient and cross-role geometries isolated, not treated as negatives',
                  'Coarse negatives supported by reviewed clear optical vegetation and100m guards, still can miss small slides',
                  'No correlation/low-VV-change rule for negative labels',
                  'S2 only for label review; X contains only existing11VV/terrain features',
                  'Reference positives and coarse background do NOT mean roads blocked or safe',
                  'Prior AOIs known and test geography frozen; no blind or cross-event validation claim']}
dump(OUT/'manifest.json',report)
dump(ROOT/'latest_v3_reference_preparation.json',{'directory':str(OUT),'training_ready':False,'weak_reference_proposals_available':True})
fig,axes=plt.subplots(2,4,figsize=(15,8),layout='constrained')
for ax,(rid,old,union,road,neg) in zip(axes.flat,plotdata):
    rgb=np.zeros((*old.shape,3));rgb[old]=[1,.6,.1];rgb[union]=[.3,.6,1];rgb[old&union]=[.6,.2,.7];rgb[neg]=[0,1,.3]
    ax.imshow(rgb);ax.contour(road,levels=[.5],colors=['white'],linewidths=.5)
    ax.set_title(rid+' | '+split['assignments'][rid],fontsize=10);ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Inventory comparison, NOT predictions or road safety\nBlue=V3 only; orange=old only; purple=overlap; green=weak negative proposals; white=road100m',fontsize=11)
fig.savefig(OUT/'v3_old_reference_comparison.png',dpi=140);plt.close(fig)
print(json.dumps({'status':report['status'],'directory':str(OUT),'weak_negative_proposal_groups':report['weak_negative_proposal_groups'],
                  'point_matrix_class_counts':report['point_matrix_class_counts']},ensure_ascii=False),flush=True)
