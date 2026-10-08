"""Choose spatially separated SH1 test squares before reading any new scores.

Selection uses candidate geometry, road length, and selected product footprints
only, NOT inventory counts, cloud/prediction performance, or original eligibility.
"""
import json,hashlib,os
from pathlib import Path
from datetime import datetime,timezone
from shapely.geometry import box,shape,mapping
from shapely.ops import transform,unary_union
from pyproj import Transformer
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'pilot_design'/'20261005T200713Z'
OUT=ROOT/'expanded_test'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True,exist_ok=False)
project=Transformer.from_crs(4326,2193,always_xy=True).transform
geo=Transformer.from_crs(2193,4326,always_xy=True).transform
catalog_path=ROOT.parent/'sentinel2'/'availability'/'20261005T193858Z'/'CDSE_raw.json'
items=json.loads(catalog_path.read_text())['value']
names=[json.loads((ROOT/'data_acquisition'/'20261007T095702Z'/f'{date}_archive_verification.json').read_text())['catalog_product'] for date in ['pre','post']]
footprints=[transform(project,shape(next(x['GeoFootprint'] for x in items if x['Name'].removesuffix('.SAFE')==name.removesuffix('.SAFE')))) for name in names]
common=footprints[0].intersection(footprints[1])
old=json.loads((BASE/'frozen_pilot_design.json').read_text())
oldbox=box(*old['selected']['bounds_nztm'])
candidates=json.loads((BASE/'candidate_summary.json').read_text())
regions=[];skipped=[]
for c in sorted(candidates,key=lambda x:(x['center_nztm'][1],x['center_nztm'][0],x['candidate_id'])):
    bounds=c['bounds_nztm'];g=box(*bounds)
    coverage=common.intersection(g).area/g.area
    reason=None
    if c['sh1_length_km']<3:reason='road shorter than 3km'
    elif coverage<.999:reason='selected 59GQP product footprint coverage <99.9%'
    elif g.intersects(oldbox.buffer(500)):reason='within pilot separation buffer'
    elif any(g.intersects(box(*r['bounds_nztm']).buffer(500)) for r in regions):reason='within earlier selected region separation buffer'
    if reason:
        skipped.append({'candidate_id':c['candidate_id'],'reason':reason});continue
    regions.append({'region_id':f'SH1-EXP-{len(regions)+1:02d}','candidate_id':c['candidate_id'],
        'bounds_nztm':bounds,'bounds_wgs84':list(transform(geo,g).bounds),
        'center_nztm':c['center_nztm'],'sh1_length_km':c['sh1_length_km'],
        'selected_product_footprint_fraction':coverage,'role':'new_spatial_region',
        'transform':[10,0,bounds[0],0,-10,bounds[3]],'shape':[640,640]})
assert len(regions)>=4,'Insufficient fixed-spatial candidates; investigate rather than weaken rules'
anchor={'region_id':'PILOT-ANCHOR','candidate_id':'P035','bounds_nztm':old['selected']['bounds_nztm'],
        'bounds_wgs84':old['selected']['bounds_wgs84'],'center_nztm':old['selected']['center_nztm'],
        'sh1_length_km':old['selected']['sh1_length_km'],'role':'old_pilot_new_terrain_source_control',
        'transform':[10,0,oldbox.bounds[0],0,-10,oldbox.bounds[3]],'shape':[640,640]}
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'directory':str(OUT),'regions':regions,
    'anchor':anchor,'new_region_count':len(regions),'new_area_km2':len(regions)*40.96,
    'old_pilot_not_in_new_aggregate':True,'crs':'EPSG:2193','source_products':names,
    'dates_nz':['2016-11-03','2016-11-23'],'rules':{
        'selection':'all geometry-qualified candidates, south-to-north greedy, no inventory eligibility/count filters',
        'side_m':6400,'minimum_SH1_length_m':3000,'separation_buffer_m':500,'footprint_fraction_min':.999,
        'quality':'source valid, SENCLOUD12 clear both dates, SCL457 both dates; mask erosion sensitivity predeclared 10m',
        'threshold':.5,'patch':256,'stride':128,'aggregation':'mean sigmoid probabilities over overlapping patches',
        'large_source_area_m2':10000,'small_source_area_m2':[100,10000],
        'object_diagnostics':['any positive','>=10% evaluated area','>=50% evaluated area'],
        'unassessed_objects':'zero evaluated raster center pixels; separately reported, never counted as safe or missed',
        'no_positive_regions':'retain and report candidate-positive area; no recall when no positives',
        'background':'inventory absence is unverified; no false-positive rate or precision claim'},
    'recommendation_gate_predeclared':{
        'purpose':'Screening candidate recommendation only, not road-passability validation',
        'needs':'report every frozen new region, include exclusions and failure regions; no results-selected deletion',
        'provisional_quantitative_gate':'At least 4 regions with >=10 evaluated near-road objects, pooled >=70% near-road objects with >=10% area coverage and >=80% large near-road objects with >=10% coverage, with no such region below50%; otherwise not a robust recommendation based on detection alone',
        'not_sufficient_alone':'Unknown positives, cloud/geometry/inventory uncertainties still require human review and real road labels'},
    'sources':[{'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [BASE/'candidate_summary.json',BASE/'frozen_pilot_design.json',catalog_path]],
    'new_predictions_read_for_selection':False,'new_inventory_counts_used_for_selection':False,
    'training':False,'fine_tuning':False,'road_status_labels':False,'independent_events_test':False,
    'limitations':['Same earthquake and acquisition pair, not independent event generalization',
                   'Current OSM not verified2016; selection tests sampled SH1 only, not full project route',
                   'Spatially separated squares do not guarantee statistical independence or no checkpoint training overlap',
                   'ALOS PC terrain differs from first pilot GEE DSM; run anchor separately before interpretation',
                   'Author input formulas audited, not whole original processing pipeline reproduced']}
(OUT/'frozen_design.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(OUT/'skipped_candidates.json').write_text(json.dumps(skipped,indent=2),encoding='utf-8')
features=[{'type':'Feature','properties':{k:v for k,v in r.items() if k not in ['bounds_nztm','bounds_wgs84','transform']},'geometry':mapping(transform(geo,box(*r['bounds_nztm'])))} for r in regions+[anchor]]
(OUT/'test_regions.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}),encoding='utf-8')
print(json.dumps({'directory':str(OUT),'new_count':len(regions),'new_area_km2':manifest['new_area_km2'],'new_regions':[(r['region_id'],r['candidate_id']) for r in regions]},indent=2),flush=True)
