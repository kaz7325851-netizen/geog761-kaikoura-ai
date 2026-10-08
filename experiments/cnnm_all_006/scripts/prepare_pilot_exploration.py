"""Freeze 3 road-centred pilot inputs before inference; exploratory only.
RGB rendering is NOT claimed to reproduce the unpublished author conversion.
No training, no threshold search, no presumed safe-road negative labels.
"""
from pathlib import Path
import json,hashlib
import numpy as np,rasterio
from rasterio.transform import Affine,rowcol
from rasterio.warp import reproject,Resampling
from rasterio.features import rasterize
from shapely.geometry import shape,box
from shapely.ops import transform,unary_union
from pyproj import Transformer
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'pilot_design'
DESIGN=BASE/'20261005T200713Z';DATA=BASE/'prepared_inputs/20261005T203524Z';QUAL=BASE/'imagery_checks/20261005T202623Z'
OUT=ROOT/'pilot_exploration';OUT.mkdir(exist_ok=True)
project=Transformer.from_crs(4326,2193,always_xy=True).transform
inventory=json.loads((DESIGN/'reference_inventory.geojson').read_text())['features']
features=[(f['properties'],transform(project,shape(f['geometry']))) for f in inventory]
inner=box(1667250+674,5321630+674,1673650-674,5328030-674)
eligible=[(p,g) for p,g in features if p['distance_to_sh1_m']<=100 and inner.contains(g.centroid)]
large=max([(p,g) for p,g in eligible if p['source_area_m2']>=10000],key=lambda x:x[0]['source_area_m2'])
small_candidates=[(p,g) for p,g in eligible if 100<=p['source_area_m2']<10000 and g.centroid.distance(large[1].centroid)>1350]
assert small_candidates
small=min(small_candidates,key=lambda x:abs(x[0]['source_area_m2']-2500))
roads=json.loads((DESIGN/'sh1_pilot_road.geojson').read_text())['features']
road=unary_union([transform(project,shape(f['geometry'])) for f in roads])
segments=json.loads((DESIGN/'road_evidence_segments.geojson').read_text())['features']
controls=[]
for f in segments:
    if f['properties']['role']!='inventory_sparse_control':continue
    geom=transform(project,shape(f['geometry']));center=geom.interpolate(.5,normalized=True)
    if inner.contains(center):controls.append((f['properties'],center))
control=max(controls,key=lambda x:min(x[1].distance(large[1].centroid),x[1].distance(small[1].centroid)))
selected=[('large_near_road',large[0],large[1].centroid,large[1]),
    ('small_near_road',small[0],small[1].centroid,small[1]),
    ('inventory_sparse_road_candidate',control[0],control[1],None)]
with rasterio.open(DATA/'pre_20161103NZ_TOA_12bands.tif') as s:
    pre=s.read([4,3,2]);source_aff=s.transform
with rasterio.open(DATA/'post_20161123NZ_TOA_12bands.tif') as s:post=s.read([4,3,2])
with rasterio.open(DATA/'ALOS_terrain_staged_10m.tif') as s:terr=s.read()
with rasterio.open(QUAL/'joint_quality_mask.tif') as s:clear=s.read(1)
world=Affine(6,0,1667250,0,-6,5328030)
def warp(a,aff,method):
    out=np.empty((224,224),a.dtype)
    reproject(a,out,src_transform=source_aff,src_crs='EPSG:2193',dst_transform=aff,dst_crs='EPSG:2193',resampling=method)
    return out
records=[]
for i,(role,props,pt,target) in enumerate(selected,1):
    rr,cc=rowcol(world,pt.x,pt.y);col,row=int(cc)-112,int(rr)-112
    aff=world*Affine.translation(col,row)
    valid=warp(clear,aff,Resampling.nearest)>0
    assert valid.mean()>=.95
    all_ref=rasterize([(g,1) for p,g in features],out_shape=(224,224),transform=aff,dtype='uint8').astype(bool)
    target_ref=rasterize([(target,1)],out_shape=(224,224),transform=aff,dtype='uint8').astype(bool) if target is not None else np.zeros((224,224),bool)
    roadmask=rasterize([(road.buffer(100),1)],out_shape=(224,224),transform=aff,dtype='uint8').astype(bool)
    # Preserve the currently staged terrain values; do not assert author's DEM/derivatives equivalence.
    hillshade=warp(terr[3],aff,Resampling.cubic);slope=warp(terr[1],aff,Resampling.cubic)
    hillshade=np.clip(hillshade,0,255);slope=np.clip(slope,0,90)
    msk=warp(terr[6],aff,Resampling.nearest)
    for upper in [3000,4000]:
        def rgb(raw):
            native=np.rint(np.clip(raw/upper,0,1)*255).astype('uint8')
            return np.stack([warp(b,aff,Resampling.cubic) for b in native],axis=-1)
        pr,po=rgb(pre),rgb(post)
        topo=np.stack([hillshade/127.5-1,slope/45-1],axis=-1).astype('float32')
        stem=f'pilot_{i:02d}_rgb0_{upper}';f=OUT/(stem+'.npz')
        np.savez_compressed(f,pre=pr.astype('float32')/127.5-1,post=po.astype('float32')/127.5-1,topo=topo,
            pre_rgb_uint8=pr,post_rgb_uint8=po,valid=valid,all_inventory=all_ref,target_inventory=target_ref,road100m=roadmask)
        records.append({'tile_id':stem,'sample_id':f'pilot_{i:02d}','role':role,'source_properties':props,
            'center_nztm':[float(pt.x),float(pt.y)],'grid':list(aff)[:6],'crs':'EPSG:2193',
            'rgb_rule':f'fixed DN clip [0,{upper}], round to uint8; same bounds for both dates and all RGB bands',
            'file':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'valid_fraction':float(valid.mean()),
            'reference_pixels':int(all_ref.sum()),'target_reference_pixels':int(target_ref.sum()),
            'road100m_copernicus_fill_fraction':float(np.mean(msk[roadmask]==48)),
            'not_safe_road_label':True})
manifest={'records':records,'shape':[224,224],'resolution_m':6,'variants_per_sample':2,
    'inputs':['pre_RGB','post_RGB','hillshade_and_slope'],'output_head':'prediction','threshold_fixed_before_inference':.5,
    'scoring_region':'central 112x112 of 224x224 context, following author notebook and avoiding padded edge outputs',
    'selection_frozen_before_inference':True,'training':False,'is_independent_accuracy_test':False,
    'limitations':['Source L1C/L2A level used for training not confirmed','Author raw-to-uint8 RGB rendering not documented; both rendering variants are exploratory',
       'Staged ALOS version and GEE terrain derivatives not verified as author equivalent','Training polygon overlap unresolved',
       'Inventory overlap is not confirmed road blockage; inventory-sparse is not a verified negative',
       'Overlapping windows are not independent samples','No complete precision/F1/IoU claim without cleaned ground truth']}
(OUT/'frozen_input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
