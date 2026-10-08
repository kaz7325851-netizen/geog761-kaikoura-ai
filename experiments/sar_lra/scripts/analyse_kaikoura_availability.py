"""Cross-check public catalogues and footprint coverage; never infer valid pixels."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
import json
import csv
import re
import numpy as np
from zoneinfo import ZoneInfo
from shapely.geometry import shape,mapping
from shapely.ops import unary_union,transform as geom_transform
from pyproj import Transformer
ROOT=Path(__file__).resolve().parent
OUT=Path(json.loads((ROOT/'kaikoura_availability/latest_query.json').read_text())['directory'])
manifest=json.loads((OUT/'query_manifest.json').read_text())
assert all(q.get('pagination_complete') and q.get('http_status')==200 for q in manifest['requests'])
project=Transformer.from_crs(4326,2193,always_xy=True).transform
corridor=shape(json.loads((OUT/'provisional_corridor.geojson').read_text())['features'][0]['geometry'])
corridor_m=geom_transform(project,corridor)
EVENT=datetime.fromisoformat('2016-11-13T11:02:56+00:00')
# Reuse the already installed IANA zone file without installing software or
# assuming UTC+13 throughout August-November (August is NZST, UTC+12).
NZ_ZONE_FILE=ROOT.parent/'.venv/Lib/site-packages/tzdata/zoneinfo/Pacific/Auckland'
with NZ_ZONE_FILE.open('rb') as zone_file:
    NZ=ZoneInfo.from_file(zone_file,key='Pacific/Auckland')
PRE_START=EVENT-timedelta(days=60)
POST_END=EVENT+timedelta(days=12)
parse=lambda s:datetime.fromisoformat(s.replace('Z','+00:00'))
pol=lambda s:tuple(sorted(set(re.findall(r'VV|VH|HH|HV',s))))
def key(name):return '_'.join(name.replace('.SAFE','').split('_')[:8])
asf=json.loads((OUT/'ASF_raw.json').read_text())['features']
cdse=json.loads((OUT/'CDSE_raw.json').read_text())['value']
asf_by={key(f['properties']['sceneName']):f for f in asf}
cdse_by=defaultdict(list)
for rec in cdse:cdse_by[key(rec['Name'])].append(rec)
assert set(asf_by)==set(cdse_by),'Catalogues disagree on acquisition identifiers'
rows=[]
groups=defaultdict(list)
for k,f in sorted(asf_by.items()):
    p=f['properties']; recs=cdse_by[k]
    attr=[{a['Name']:a['Value'] for a in rec['Attributes']} for rec in recs]
    for a,rec in zip(attr,recs):
        assert p['flightDirection']==a['orbitDirection']
        assert p['pathNumber']==a['relativeOrbitNumber']
        assert pol(p['polarization'])==pol(a['polarisationChannels'])
        assert abs((parse(p['startTime'])-parse(rec['ContentDate']['Start'])).total_seconds())<1
    geom=shape(f['geometry'])
    if not geom.is_valid:geom=geom.buffer(0)
    geom_m=geom_transform(project,geom)
    fraction=geom_m.intersection(corridor_m).area/corridor_m.area
    dt=parse(p['startTime'])
    window='pre60' if PRE_START<=dt<EVENT else 'post12' if EVENT<=dt<POST_END else 'outside'
    row={'acquisition_slice_key':k,'pass':p['flightDirection'],'relative_orbit':p['pathNumber'],
         'platform':p['platform'],'absolute_orbit':p['orbit'],'time_utc':dt.isoformat(),
         'time_nz_local':dt.astimezone(NZ).strftime('%Y-%m-%d %H:%M:%S %Z'),
         'polarization':'+'.join(pol(p['polarization'])),'period':window,
         'footprint_corridor_fraction':fraction,'asf_scene':p['sceneName'],
         'asf_url':p['url'],'cdse_records':[r['Id'] for r in recs],
         'cdse_names':[r['Name'] for r in recs]}
    rows.append(row)
    acquisition=(p['flightDirection'],p['pathNumber'],p['platform'],p['orbit'])
    groups[acquisition].append((row,geom_m))
    # Both catalogue footprints may differ slightly due to reprocessing; not an exact pixel mask.

observations=[]
footprints=[]
for acquisition,slices in sorted(groups.items()):
    pass_,rel,platform,absolute=acquisition
    mosaic=unary_union([g for _,g in slices])
    first=min(parse(r['time_utc']) for r,_ in slices)
    pols={r['polarization'] for r,_ in slices}
    assert len(pols)==1
    polarization=pols.pop()
    window='pre60' if PRE_START<=first<EVENT else 'post12' if EVENT<=first<POST_END else 'outside'
    fraction=mosaic.intersection(corridor_m).area/corridor_m.area
    observation={'pass':pass_,'relative_orbit':rel,'platform':platform,'absolute_orbit':absolute,
        'first_slice_utc':first.isoformat(),'first_slice_nz_local':first.astimezone(NZ).isoformat(),
        'polarization':polarization,'period':window,'slices':len(slices),
        'footprint_corridor_fraction':fraction,'full_footprint_coverage_99_9pct':fraction>=.999,
        'delay_hours_if_post':(first-EVENT).total_seconds()/3600 if first>=EVENT else None,
        'days_before_if_pre':(EVENT-first).total_seconds()/86400 if first<EVENT else None}
    observations.append(observation)
    footprints.append({'type':'Feature','properties':observation,'geometry':mapping(geom_transform(
        Transformer.from_crs(2193,4326,always_xy=True).transform,mosaic))})

summaries=[]
for pass_,rel in sorted({(o['pass'],o['relative_orbit']) for o in observations}):
    subset=[o for o in observations if (o['pass'],o['relative_orbit'])==(pass_,rel)]
    pre=[o for o in subset if o['period']=='pre60']
    post=[o for o in subset if o['period']=='post12']
    predual=[o for o in pre if o['polarization']=='VH+VV']
    postdual=[o for o in post if o['polarization']=='VH+VV']
    olddual=[o for o in subset if o['polarization']=='VH+VV' and parse(o['first_slice_utc'])<EVENT]
    nearest=max(olddual,key=lambda o:o['first_slice_utc']) if olddual else None
    vvpre=[o for o in pre if 'VV' in o['polarization']]
    vvpost=[o for o in post if 'VV' in o['polarization']]
    summary={'pass':pass_,'relative_orbit':rel,'pre60_observations':len(pre),'pre60_dual_observations':len(predual),
        'post12_observations':len(post),'post12_dual_observations':len(postdual),
        'pre60_vv_dates':[o['first_slice_utc'] for o in vvpre],
        'post12_dates':[o['first_slice_utc'] for o in post],
        'post12_earliest_dual':min(postdual,key=lambda o:o['first_slice_utc']) if postdual else None,
        'nearest_pre_dual_within_search':nearest,
        'vv_only_stack_has_matching_track_pre_post':bool(vvpre and vvpost),
        'published_dual_60_12_stack_feasible':bool(predual and postdual),
        'pre60_full_footprint_observations':sum(o['full_footprint_coverage_99_9pct'] for o in pre),
        'post12_full_footprint_observations':sum(o['full_footprint_coverage_99_9pct'] for o in post)}
    summaries.append(summary)
report={'status':'Completed public-catalogue feasibility check, not inference',
    'event_utc_exact':EVENT.isoformat(),'event_time_source':'https://www.geonet.org.nz/earthquake/technical/2016p858000',
    'nz_time_zone':'Pacific/Auckland, IANA data; daylight saving applied by date',
    'query_manifest_event_time_was_rounded_to_minute':True,
    'exact_relative_windows':{'pre60_start_inclusive':PRE_START.isoformat(),'pre_end_exclusive':EVENT.isoformat(),
                              'post_start_inclusive':EVENT.isoformat(),'post12_end_exclusive':POST_END.isoformat()},
    'author_calendar_day_buffer_also_checked':{'pre':['2016-09-13','2016-11-12'],
                                              'post':['2016-11-14','2016-11-26'],
       'note':'Changing to these UTC-calendar windows does not add any pre-event VV+VH acquisitions'},
    'aoi_status':manifest['scope'],'aoi_area_km2':corridor_m.area/1e6,
    'search_range':manifest['window'],'asf_records':len(asf),'cdse_records':len(cdse),
    'common_unique_acquisition_slices':len(asf_by),'unique_overpasses':len(observations),
    'cdse_representations_per_slice':dict(Counter(len(v) for v in cdse_by.values())),
    'cross_catalogue_checks':'All 40 acquisition slice keys, pass, relative orbit, polarization and times agree (<1s)',
    'pre60_total_dual_overpasses':sum(o['period']=='pre60' and o['polarization']=='VH+VV' for o in observations),
    'current_dual_weights_compatible':any(s['published_dual_60_12_stack_feasible'] for s in summaries),
    'track_summaries':summaries,
    'limitations':['Provisional old corridor, not final group AOI',
        'Catalogues give product footprints, NOT usable pixel masks or shadow/layover-free coverage',
        'CDSE original and COG representations do not represent separate observation dates',
        'ASF/CDSE availability is NOT current Earth Engine ingestion availability',
        'No imagery, weights, road predictions or accuracy measurements produced in this check',
        'August dual-polarization baseline lies outside published 60-day configuration',
        'VV-only input requires a matching genuine VV-only checkpoint, not zero-filled VH']}
# Independently apply the author's calendar-day windows to prove the key result is stable.
author_pre=[o for o in observations if '2016-09-13'<=o['first_slice_utc'][:10]<'2016-11-12'
            and o['polarization']=='VH+VV']
assert not author_pre and report['pre60_total_dual_overpasses']==0
(OUT/'availability_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(OUT/'observations.json').write_text(json.dumps(observations,indent=2),encoding='utf-8')
(OUT/'slice_metadata.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(OUT/'observation_footprints.geojson').write_text(json.dumps({'type':'FeatureCollection','features':footprints}),encoding='utf-8')
fields=['pass','relative_orbit','platform','absolute_orbit','first_slice_utc','first_slice_nz_local',
        'polarization','period','slices','footprint_corridor_fraction','full_footprint_coverage_99_9pct',
        'delay_hours_if_post','days_before_if_pre']
with (OUT/'observations.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(observations)
print(json.dumps(report,indent=2),flush=True)
