"""Public read-only IW GRD catalogue queries; no credentials or raster downloads.

Use the OLD OSM corridor only as a provisional feasibility-check geometry, not
as a newly agreed team AOI. Save original metadata and request provenance.
"""
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union, transform as geom_transform
from pyproj import Transformer

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'kaikoura_availability'/datetime.now().strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True)
route_path=ROOT.parent/'sentinel1/study_route_lines.geojson'
routes=json.loads(route_path.read_text(encoding='utf-8'))
forward=Transformer.from_crs(4326,2193,always_xy=True).transform
inverse=Transformer.from_crs(2193,4326,always_xy=True).transform
lines=unary_union([shape(f['geometry']) for f in routes['features']])
corridor_metric=geom_transform(forward,lines).buffer(2000)
corridor=geom_transform(inverse,corridor_metric)
query_box=box(*corridor.bounds)
(OUT/'provisional_corridor.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[
    {'type':'Feature','properties':{'status':'Old corridor, feasibility check ONLY, not final team AOI',
      'buffer_m':2000,'crs_for_buffer':'EPSG:2193'},'geometry':mapping(corridor)}]}),encoding='utf-8')

START,END='2016-08-01T00:00:00Z','2016-12-01T00:00:00Z'
asf_params={'dataset':'SENTINEL-1','beamMode':'IW','processingLevel':'GRD_HD,GRD_HS',
    'intersectsWith':query_box.wkt,'start':START,'end':END,'maxResults':'1000','output':'geojson'}
asf_url='https://api.daac.asf.alaska.edu/services/search/param?'+urlencode(asf_params)
cdse_filter=("Collection/Name eq 'SENTINEL-1' and contains(Name,'IW_GRDH') and "
    f"ContentDate/Start ge {START} and ContentDate/Start lt {END} and "
    "OData.CSC.Intersects(area=geography'SRID=4326;"+query_box.wkt+"')")
cdse_url='https://catalogue.dataspace.copernicus.eu/odata/v1/Products?'+urlencode({
    '$filter':cdse_filter,'$top':1000,'$expand':'Attributes','$orderby':'ContentDate/Start asc','$count':'true'})

def fetch(source,url):
    record={'source':source,'url':url,'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        request=Request(url,headers={'User-Agent':'GEOG761-public-catalogue-feasibility-check/1.0'})
        with urlopen(request,timeout=45) as response:
            body=response.read()
            record.update(http_status=response.status,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
        data=json.loads(body)
        (OUT/(source+'_raw.json')).write_bytes(body)
        records=data.get('features',[]) if source=='ASF' else data.get('value',[])
        record['record_count']=len(records)
        record['pagination_complete']=len(records)<1000 and not data.get('@odata.nextLink')
        if '@odata.count' in data:record['reported_count']=data['@odata.count']
        print(json.dumps(record),flush=True)
        return record
    except Exception as exc:
        record.update(error_type=type(exc).__name__,error=str(exc))
        print(json.dumps(record),flush=True)
        return record

manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
  'event_utc':'2016-11-13T11:02:00Z','event_nzdt':'2016-11-14T00:02:00+13:00',
  'scope':'Old Oaro-Kekerengu SH1 and Kaikoura-Waiau Route70 corridor, NOT confirmed final team AOI',
  'source_route_file':str(route_path),'source_route_sha256':hashlib.sha256(route_path.read_bytes()).hexdigest(),
  'aoi_bounds':list(corridor.bounds),'aoi_area_km2':corridor_metric.area/1e6,
  'query_bbox':list(query_box.bounds),'window':[START,END],
  'catalogue_coverage_is_not_valid_pixel_or_layover_coverage':True,
  'browser':'Earth Engine browser connection timed out twice; did not access user credentials',
  'earth_engine_current_availability':'NOT re-queried in this run',
  'requests':[]}
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(fetch,'ASF',asf_url),pool.submit(fetch,'CDSE',cdse_url)]
    manifest['requests']=[f.result() for f in futures]
(OUT/'query_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(ROOT/'kaikoura_availability/latest_query.json').write_text(json.dumps({'directory':str(OUT)},indent=2),encoding='utf-8')
print('OUTPUT='+str(OUT),flush=True)
