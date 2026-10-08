"""Public metadata only: no raster downloads, exports or model inference."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen
from urllib.parse import urlencode
from collections import defaultdict
import json
from shapely.geometry import shape, box
from shapely.ops import unary_union, transform
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'availability' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True)
source = ROOT.parent / 'sar_lra/kaikoura_availability/20261005_170032/provisional_corridor.geojson'
corridor = unary_union([shape(f['geometry']) for f in json.loads(source.read_text())['features']])
project = Transformer.from_crs(4326, 2193, always_xy=True).transform
area = transform(project, corridor).area
query = "Collection/Name eq 'SENTINEL-2' and ContentDate/Start ge 2016-09-01T00:00:00Z and ContentDate/Start lt 2016-12-15T00:00:00Z and OData.CSC.Intersects(area=geography'SRID=4326;" + box(*corridor.bounds).wkt + "')"
url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?' + urlencode({'$filter':query, '$top':1000,'$expand':'Attributes','$orderby':'ContentDate/Start asc','$count':'true'})
with urlopen(url, timeout=45) as response:
    raw = response.read()
(OUT / 'CDSE_raw.json').write_bytes(raw)
data = json.loads(raw)
assert not data.get('@odata.nextLink'), 'Pagination incomplete'
rows, groups = [], defaultdict(list)
for item in data['value']:
    a = {v['Name']:v['Value'] for v in item.get('Attributes',[])}
    level = 'L2A' if 'MSIL2A' in item['Name'] else 'L1C' if 'MSIL1C' in item['Name'] else 'other'
    g = shape(item['GeoFootprint'])
    if not g.intersects(corridor): continue
    date = item['ContentDate']['Start'][:10]
    row = {'name':item['Name'],'id':item['Id'],'date':date,'level':level,'cloud_percent':a.get('cloudCover'),'attributes':a,'corridor_footprint_percent':100*transform(project,g.intersection(corridor)).area/area}
    rows.append(row)
    groups[(level,date)].append((g,row))
daily = []
for (level,date), entries in sorted(groups.items()):
    coverage = 100*transform(project,unary_union([g for g,r in entries]).intersection(corridor)).area/area
    daily.append({'level':level,'date':date,'footprint_percent':coverage,'products':len(entries),'cloud_values':sorted(set(r['cloud_percent'] for g,r in entries if r['cloud_percent'] is not None))})
summary = {'source_url':url,'scope':'Old provisional SH1 and Inland Kaikoura Road 2-km corridor; NOT final team AOI','bounds':corridor.bounds,'catalogue_records':len(data['value']),'intersecting_products':len(rows),'daily':daily,'limitations':['Footprint coverage only, NOT cloud-free or usable pixel coverage','Whole-product cloud percentage is not road-specific cloud percentage','No imagery downloaded or inference performed','Post-event imagery must be matched to dated road-status labels']}
(OUT / 'product_metadata.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(OUT / 'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'directory':str(OUT),**summary},indent=2),flush=True)
