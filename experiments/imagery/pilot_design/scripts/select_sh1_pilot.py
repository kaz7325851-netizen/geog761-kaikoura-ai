"""Freeze a road-centred optical pilot using inventory and footprints, never scores.

All size/proximity cutoffs below are exploratory sampling rules, NOT road-status
truth, author model requirements, or operational safety thresholds.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import csv, hashlib, json, math
import numpy as np
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping, box
from shapely.ops import transform, unary_union, linemerge
from shapely.strtree import STRtree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
OUT = ROOT / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True, exist_ok=False)
to_metric = Transformer.from_crs(4326, 2193, always_xy=True).transform
to_geo = Transformer.from_crs(2193, 4326, always_xy=True).transform
road_path = DATA / 'sentinel1/study_route_lines.geojson'
catalog_path = DATA / 'sentinel2/availability/20261005T193858Z/CDSE_raw.json'
ref_path = DATA / 'sar_lra/kaikoura_exploration/reference/Kaikora_2016.shp'
road_features = json.loads(road_path.read_text())['features']
all_roads = unary_union([transform(to_metric, shape(f['geometry'])) for f in road_features])
road = unary_union([transform(to_metric, shape(f['geometry'])) for f in road_features if f['properties']['route'].startswith('SH1')])
merged = linemerge(road) if road.geom_type != 'LineString' else road
parts = list(merged.geoms) if merged.geom_type == 'MultiLineString' else [merged]
catalog = json.loads(catalog_path.read_text())['value']
dates = ['2016-11-02', '2016-11-22']
footprints, sources = {}, {}
for date in dates:
    items = [i for i in catalog if 'MSIL2A' in i['Name'] and i['ContentDate']['Start'][:10] == date]
    assert items
    sources[date] = items
    footprints[date] = transform(to_metric, unary_union([shape(i['GeoFootprint']) for i in items]))
common = footprints[dates[0]].intersection(footprints[dates[1]])
ref_crs = CRS.from_wkt(ref_path.with_suffix('.prj').read_text())
ref_to_metric = Transformer.from_crs(ref_crs, 2193, always_xy=True).transform
reader = shapefile.Reader(str(ref_path))
geoms, indices, repaired = [], [], 0
road_search_buffer = all_roads.buffer(5000)
for i, shp in enumerate(reader.iterShapes()):
    g = shape(shp.__geo_interface__)
    if not g.is_valid:
        g = g.buffer(0)
        repaired += 1
    g = transform(ref_to_metric, g)
    if not g.is_empty and g.intersects(road_search_buffer):
        indices.append(i)
        geoms.append(g)
tree = STRtree(geoms)
seen = set()
candidates = []

def line_parts(g):
    if g.is_empty: return []
    if g.geom_type == 'LineString': return [g]
    if hasattr(g,'geoms'): return [line for x in g.geoms for line in line_parts(x)]
    return []

def local_records(aoi, local_road):
    records = []
    for k in tree.query(aoi):
        g = geoms[int(k)]
        if not g.intersects(aoi): continue
        records.append({'source_index':indices[int(k)], 'geometry':g,
            'source_area_m2':g.area,'distance_to_sh1_m':g.distance(local_road),
            'clipped_area_m2':g.intersection(aoi).area})
    return records

def sample_segments(local_road):
    segs = []
    for line in line_parts(local_road):
        # Samples for evidence collection; not a validated topology/route network.
        for start in np.arange(0, max(0,line.length-199.999), 200):
            from shapely.ops import substring
            segment = substring(line,float(start),float(start+200))
            nearby = [int(k) for k in tree.query(segment.buffer(100)) if geoms[int(k)].distance(segment)<=100]
            farther = [int(k) for k in tree.query(segment.buffer(200)) if geoms[int(k)].distance(segment)<=200]
            large = sum(geoms[k].area >= 10000 for k in nearby)
            small = sum(100 <= geoms[k].area < 10000 for k in nearby)
            role = 'large_and_small_nearby' if large and small else 'large_nearby' if large else 'small_nearby' if small else 'inventory_sparse_control' if not farther else 'other_or_ambiguous'
            segs.append({'geometry':segment,'role':role,'nearby_large':large,'nearby_small':small,
                'nearby_inventory_indices':[indices[k] for k in nearby],
                'road_status':'unknown','independent_evidence_checked':False})
    return segs

for part_id, line in enumerate(parts):
    for distance in np.arange(0,line.length+1,2000):
        point = line.interpolate(float(distance))
        x,y = round(point.x/10)*10,round(point.y/10)*10
        if (x,y) in seen: continue
        seen.add((x,y))
        aoi = box(x-3200,y-3200,x+3200,y+3200)
        local_road = road.intersection(aoi)
        records = local_records(aoi,local_road)
        near = [r for r in records if r['distance_to_sh1_m']<=100]
        segments = sample_segments(local_road)
        c = {'candidate_id':f'P{len(candidates)+1:03d}','center_nztm':[x,y],
            'center_wgs84':list(transform(to_geo,point).coords)[0],
            'bounds_nztm':list(aoi.bounds),'sh1_length_km':local_road.length/1000,
            'common_footprint_fraction':min(1,common.intersection(aoi).area/aoi.area),
            'inventory_polygons_in_aoi':len(records),
            'large_polygons_near_sh1':sum(r['source_area_m2']>=10000 for r in near),
            'small_polygons_near_sh1':sum(100<=r['source_area_m2']<10000 for r in near),
            'very_small_polygons_near_sh1':sum(r['source_area_m2']<100 for r in near),
            'segment_role_counts':dict(Counter(s['role'] for s in segments))}
        c['eligible'] = (c['common_footprint_fraction']>=0.999 and c['sh1_length_km']>=3
            and c['large_polygons_near_sh1']>0 and c['small_polygons_near_sh1']>0
            and c['segment_role_counts'].get('inventory_sparse_control',0)>=3)
        candidates.append(c)

eligible = [c for c in candidates if c['eligible']]
assert eligible, 'No candidate satisfies frozen inventory/footprint criteria; do not relax silently'
# Deterministic ranking: balance large and small source polygons, then controls,
# then road length. Raw count means sample availability, NOT better model scores.
def rank(c):
    return (min(c['large_polygons_near_sh1'],c['small_polygons_near_sh1']),
        c['segment_role_counts'].get('inventory_sparse_control',0),c['sh1_length_km'],
        -int(c['candidate_id'][1:]))
selected = max(eligible,key=rank)
aoi = box(*selected['bounds_nztm'])
local_road = road.intersection(aoi)
records = local_records(aoi,local_road)
segments = sample_segments(local_road)
for i,s in enumerate(segments): s['segment_id'] = f'SH1-PILOT-{i+1:03d}'

def export(name,items):
    features=[]
    for props,g in items:
        features.append({'type':'Feature','properties':props,'geometry':mapping(transform(to_geo,g))})
    (OUT/name).write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False),encoding='utf-8')

export('pilot_aoi.geojson',[({'pilot_id':'SH1-OPTICAL-01','status':'optical pilot, NOT final team AOI','crs_for_bounds':'EPSG:2193','cloud_free_verified':False},aoi)])
export('sh1_pilot_road.geojson',[({'route':'SH1','historic_2016_geometry_verified':False},local_road)])
export('reference_inventory.geojson',[( {k:v for k,v in r.items() if k!='geometry'},r['geometry'].intersection(aoi)) for r in records])
export('road_evidence_segments.geojson',[( {k:v for k,v in s.items() if k!='geometry'},s['geometry']) for s in segments])
export('all_candidate_aois.geojson',[( {k:v for k,v in c.items() if k!='segment_role_counts'},box(*c['bounds_nztm'])) for c in candidates])
export('catalogue_common_footprint.geojson',[({'dates_utc':dates,'valid_pixels_verified':False},common)])
for date in dates:
    relevant=[i for i in sources[date] if transform(to_metric,shape(i['GeoFootprint'])).intersects(aoi)]
    selected[date]={'aoi_footprint_fraction':min(1,footprints[date].intersection(aoi).area/aoi.area),'products':[
        {'name':i['Name'],'id':i['Id'],'acquisition_utc':i['ContentDate']['Start'],
        'whole_product_cloud_percent':{v['Name']:v['Value'] for v in i['Attributes']}.get('cloudCover')} for i in relevant]}
selected['bounds_wgs84']=list(transform(to_geo,aoi).bounds)
selected['area_km2']=aoi.area/1e6
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'pilot_id':'SH1-OPTICAL-01',
    'selected':selected,'candidate_count':len(candidates),'eligible_count':len(eligible),
    'rules':{'square_side_m':6400,'sampling_spacing_m':2000,'road_proximity_m':100,
        'large_source_polygon_m2':'>=10000','small_source_polygon_m2':'100 to <10000',
        'control_no_inventory_within_m':200,'segment_length_m':200,
        'selection':'lexicographic balanced large/small count, control segments, SH1 length; no predictions read'},
    'evaluation_date_nz':'2016-11-23','evaluation_type':'Retrospective snapshot ~9 days after earthquake, NOT 24-48h response',
    'imagery_downloaded':False,'model_run':False,'labels_are_road_status':False,
    'training_overlap_checked':False,'cloud_free_pixels_checked':False,'geometry_repairs_full_inventory':repaired,
    'inputs':[{ 'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [road_path,catalog_path,ref_path,ref_path.with_suffix('.prj')]],
    'limitations':['Inventory-guided pilot, not random representative全域样本','Current OSM geometry, not verified 2016 road network',
        'Sparse inventory controls are not confirmed safe roads','No transport start/end or vehicle assumptions made',
        'Model training overlap still requires checking; not a frozen independent holdout yet',
        'If cloud mask fails, changes require new recorded design version rather than silent area edits']}
(OUT/'frozen_pilot_design.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'candidate_summary.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'road_evidence_segments.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['segment_id','role','nearby_large','nearby_small','road_status','independent_evidence_checked']
    writer=csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    writer.writerows({k:s[k] for k in fields} for s in segments)

# Static scientific figure, using sourced geometry; no tile or image download.
fig,axes=plt.subplots(1,2,figsize=(13,6.5),gridspec_kw={'width_ratios':[1,1.15]})
def plot_line(ax,g,**kwargs):
    for line in line_parts(g):
        x,y=line.xy; ax.plot(np.array(x)/1000,np.array(y)/1000,**kwargs)
def plot_polygon(ax,g,**kwargs):
    if g.geom_type=='Polygon':
        x,y=g.exterior.xy; ax.fill(np.array(x)/1000,np.array(y)/1000,**kwargs)
    elif hasattr(g,'geoms'):
        for part in g.geoms: plot_polygon(ax,part,**kwargs)
plot_line(axes[0],all_roads,color='#8b969e',lw=1)
plot_line(axes[0],road,color='#176b91',lw=1.7)
plot_polygon(axes[0],aoi,facecolor='none',edgecolor='#b24732',lw=2)
cx,cy=aoi.centroid.x/1000,aoi.centroid.y/1000
axes[0].annotate('SH1-OPTICAL-01',(cx,cy),xytext=(-60,30),textcoords='offset points',arrowprops={'arrowstyle':'->'},fontsize=10)
axes[0].set_title('Location within existing project roads')
for r in records: plot_polygon(axes[1],r['geometry'].intersection(aoi),facecolor='#b24732',edgecolor='none',alpha=.65)
plot_line(axes[1],local_road,color='#176b91',lw=2)
for s in segments:
    if s['role']=='inventory_sparse_control': plot_line(axes[1],s['geometry'],color='#40946c',lw=3)
plot_polygon(axes[1],aoi,facecolor='none',edgecolor='#666666',lw=1)
axes[1].set_xlim(aoi.bounds[0]/1000,aoi.bounds[2]/1000)
axes[1].set_ylim(aoi.bounds[1]/1000,aoi.bounds[3]/1000)
axes[1].set_title('Pilot: reference slides and SH1 (not predictions)')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
axes[1].legend(handles=[Line2D([],[],color='#176b91',label='SH1'),Patch(color='#b24732',label='Raw slide inventory'),Line2D([],[],color='#40946c',label='Inventory-sparse control candidates')],loc='lower left',fontsize=8)
for ax in axes:
    ax.set_aspect('equal'); ax.set_xlabel('NZTM easting (km)'); ax.set_ylabel('NZTM northing (km)'); ax.grid(alpha=.15)
fig.suptitle('Road-centred Sentinel-2 pilot selection | 6.4 x 6.4 km',fontsize=14)
fig.text(.5,.02,'Footprint coverage only; clouds, historical road status and training overlap not yet verified.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,.95])
fig.savefig(OUT/'pilot_selection.png',dpi=160)
plt.close(fig)

# Independently reread saved geometry to verify round-trip location/area.
saved=json.loads((OUT/'pilot_aoi.geojson').read_text())
roundtrip=transform(to_metric,shape(saved['features'][0]['geometry']))
area_error=abs(roundtrip.area-aoi.area)/aoi.area
assert area_error<1e-6
assert road.intersection(roundtrip).length>3000
assert all(s['road_status']=='unknown' for s in segments)
verification={'area_roundtrip_relative_error':area_error,'road_length_check_passed':True,
    'road_status_not_fabricated':True,'geometry_files_parse':True}
(OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps({'directory':str(OUT),'selected':selected,'candidate_count':len(candidates),'eligible_count':len(eligible),'verification':verification},ensure_ascii=True,indent=2),flush=True)
