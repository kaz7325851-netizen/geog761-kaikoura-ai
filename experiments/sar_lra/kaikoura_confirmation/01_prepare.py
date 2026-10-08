"""Freeze label-stratified diagnostic sites before downloading new inputs/predicting."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
import shapefile,pyproj
from shapely.geometry import shape,box,mapping,Point
from shapely.ops import transform,unary_union
from shapely.strtree import STRtree
ROOT=Path(__file__).resolve().parent
SAR=ROOT.parent
ROOT.mkdir(exist_ok=True)
forward=pyproj.Transformer.from_crs(4326,2193,always_xy=True).transform
reverse=pyproj.Transformer.from_crs(2193,4326,always_xy=True).transform
roads_file=SAR.parent/'sentinel1/study_route_lines.geojson'
roads=unary_union([transform(forward,shape(f['geometry'])) for f in json.loads(roads_file.read_text(encoding='utf-8'))['features']])
source=SAR/'kaikoura_exploration/reference/Kaikora_2016.shp'
crs=pyproj.CRS.from_wkt(source.with_suffix('.prj').read_text())
conv=pyproj.Transformer.from_crs(crs,2193,always_xy=True).transform
inventory=[]; repairs=0
for shp in shapefile.Reader(str(source)).shapes():
    geom=shape(shp.__geo_interface__)
    if not geom.is_valid:
        geom=geom.buffer(0);repairs+=1
    inventory.append(transform(conv,geom))
tree=STRtree(inventory)
old=box(1645530,5299320,1651930,5305720)
def tile(point):
    west=int(point.x//10)*10-3200
    north=int(point.y//10)*10+3200
    return box(west,north-6400,west+6400,north)
def intersects_used(g,used):
    return any(g.intersection(u).area>0 for u in used)
large_candidates=[(i,g) for i,g in enumerate(inventory) if not g.is_empty and g.distance(roads)<10000 and not intersects_used(tile(g.centroid),[old])]
idx,large=max(large_candidates,key=lambda p:(p[1].area,-p[0]))
a=tile(large.centroid)
roadbuffer=roads.buffer(500)
stats=[];seen=set()
parts=list(roads.geoms) if hasattr(roads,'geoms') else [roads]
for part in parts:
    for distance in np.arange(0,part.length+1,2000):
        point=part.interpolate(float(distance));g=tile(point)
        key=tuple(g.bounds)
        if key in seen or intersects_used(g,[old,a]):continue
        seen.add(key)
        west,south,east,north=g.bounds
        aff=Affine(10,0,west,0,-10,north)
        local=[inventory[j].intersection(g) for j in tree.query(g,predicate='intersects')]
        label=rasterize([(s,1) for s in local if s.area>0],out_shape=(640,640),transform=aff,fill=0,dtype='uint8') if local else np.zeros((640,640),dtype='uint8')
        road=rasterize([(roadbuffer.intersection(g),1)],out_shape=(640,640),transform=aff,fill=0,dtype='uint8')
        fractions=[];near=[]
        for y in range(0,576,64):
            for x in range(0,576,64):
                fractions.append(label[y:y+64,x:x+64].mean());near.append(road[y+32,x+32]>0)
        fractions=np.array(fractions);near=np.array(near)
        stats.append({'bounds':list(g.bounds),'road_marginal_nonoverlap_windows':int(((fractions>.05)&(fractions<=.15)&near).sum()),
                      'all_raw_positive_nonoverlap_windows':int((fractions>.05).sum()),'raw_inventory_fraction':float(label.mean()),
                      'center_nztm':[point.x,point.y]})
assert stats
b_stat=max(stats,key=lambda s:(s['road_marginal_nonoverlap_windows'],-s['raw_inventory_fraction'],-s['bounds'][0]))
b=box(*b_stat['bounds'])
# Background contrast chosen without predictions, prioritizing far-inland road centre
# among minimum-inventory candidates (not guaranteed no landslide or safe).
eligible=[s for s in stats if not intersects_used(box(*s['bounds']),[old,a,b])]
c_stat=min(eligible,key=lambda s:(s['raw_inventory_fraction'],s['center_nztm'][0],s['bounds'][1]))
c=box(*c_stat['bounds'])
sites=[]
for sid,role,g,extra in [('A','large_inventory_slide',a,{'source_slide_index':idx,'source_slide_area_m2':large.area}),
                         ('B','road_near_marginal_slide',b,b_stat),('C','inventory_sparse_background_contrast',c,c_stat)]:
    west,south,east,north=g.bounds
    sites.append({'id':sid,'role':role,'bounds_nztm':list(g.bounds),'crs':'EPSG:2193',
                  'grid':[10,0,int(west),0,-10,int(north)],'dimensions':[640,640],
                  'bounds_wgs84':list(transform(reverse,g).bounds),'selection_details':extra})
assert all(not intersects_used(box(*s['bounds_nztm']),[old]+[box(*t['bounds_nztm']) for t in sites if t['id']!=s['id']]) for s in sites)
freeze={'prepared_utc':datetime.now(timezone.utc).isoformat(),'event':'Kaikoura2016','final_team_aoi':False,
 'sites':sites,'selection':'Label-stratified diagnostics, NOT random representative sampling; frozen before new imagery or scores',
 'site_rules':{'A':'Largest inventory polygon within10km of existing roads, excluding old pilot',
 'B':'Road-centred candidates every2km: maximum count of nonoverlap raw5-15% windows within500m roads; tie lowest inventory fraction',
 'C':'Minimum inventory fraction among remaining nonoverlap road-centred sites; tie most-western road centre'},
 'candidate_sites':stats,'source_inventory_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'source_roads_sha256':hashlib.sha256(roads_file.read_bytes()).hexdigest(),'geometry_repairs':repairs,
 'pre':['2016-08-23','2016-08-24'],'post':['2016-11-15','2016-11-16'],'relative_orbit':52,'pass':'ASCENDING',
 'checkpoint_sha256':'ba0599c713a69eb477b3ff540d008246b5b758a8bb9b14e3fc4d97f24f0ff21c',
 'model_threshold':.6,'positive_fraction_threshold':.05,'marginal_positive_max_fraction':.15,
 'dense_positive_min_fraction':.15,'size':64,'stride':32,'independent_grid_stride':64,
 'background_rule':'Raw inventory ZERO; no inventory within200m of window; >=95% DEM elevation>1m land proxy; >=95% geometry-visible',
 'background_limit':'Inventory-derived reference background only, NOT independently verified no-slide / safe road',
 'road_near_rule':'Window intersects road500m buffer (current OSM geometry, not verified2016 roads)',
 'visibility':'Vollrath2020 local active masks, SRTM30, buffer0; pre AND contributing post geometry',
 'geometry_control':'Additional old-pilot EPSG4326 nominal10m nearest-neighbour GEE export, matching author deployment CRS; same imagery and weights',
 'stop_rule':'If recall remains<=10% in every positive-support site, stop blanket regional expansion under current configuration; dense-only success implies coarse triage only',
 'selection_frozen_before_new_predictions':True,'training':False,'limitations':['Older82day single-date pre persists; no near-event dual-pol baseline available in prior audited region','This is a targeted diagnostic, not event-wide accuracy estimate']}
(ROOT/'frozen_design.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'test_sites.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':{'id':s['id'],'role':s['role'],'final_team_aoi':False},'geometry':mapping(transform(reverse,box(*s['bounds_nztm'])))} for s in sites]}),encoding='utf-8')
for s in sites:
    g=box(*s['bounds_nztm']);dest=ROOT/'reference'/s['id'];dest.mkdir(parents=True,exist_ok=True)
    geoms=[inventory[j].intersection(g) for j in tree.query(g,predicate='intersects')]
    # Include200m neighbouring labels so background distance is correct at tile edges.
    expanded=g.buffer(200)
    nearby=[inventory[j].intersection(expanded) for j in tree.query(expanded,predicate='intersects')]
    features=[{'type':'Feature','properties':{},'geometry':mapping(transform(reverse,k))} for k in nearby if k.area>0]
    (dest/'inventory_with_margin.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}),encoding='utf-8')
    aff=Affine(*s['grid'])
    for name,geomlist in [('raw_reference_mask',geoms),('inventory_buffer200_mask',[unary_union(nearby).buffer(200).intersection(g)]),('road500_mask',[roadbuffer.intersection(g)])]:
        pixels=rasterize([(k,1) for k in geomlist if not k.is_empty and k.area>0],out_shape=(640,640),transform=aff,fill=0,dtype='uint8') if any(not k.is_empty and k.area>0 for k in geomlist) else np.zeros((640,640),dtype='uint8')
        with rasterio.open(dest/(name+'.tif'),'w',driver='GTiff',width=640,height=640,count=1,crs='EPSG:2193',transform=aff,dtype='uint8',compress='deflate') as dst:dst.write(pixels,1)
print(json.dumps({'sites':sites,'candidate_count':len(stats),'selection_frozen':True},ensure_ascii=False,indent=2))
