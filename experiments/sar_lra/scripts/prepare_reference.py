"""Extract only Sumatra reference files safely and rasterize them onto exact input grid."""
from pathlib import Path
import zipfile
import hashlib
import json
import shapefile
from shapely.geometry import shape,mapping,box
from shapely.ops import transform as transform_geom
import pyproj
import rasterio
from rasterio.features import rasterize
import numpy as np
ROOT=Path(__file__).resolve().parent
target=ROOT/'data/reference'
target.mkdir(exist_ok=True)
with zipfile.ZipFile(ROOT/'data/landslide_inventories_polygons.zip') as archive:
    for name in archive.namelist():
        if not name.startswith('Sumatra_2022.'):continue
        dest=(target/name).resolve()
        if not dest.is_relative_to(target.resolve()):raise ValueError(name)
        dest.write_bytes(archive.read(name))
prj=(target/'Sumatra_2022.prj').read_text()
crs=pyproj.CRS.from_wkt(prj)
reader=shapefile.Reader(str(target/'Sumatra_2022.shp'))
geoms=[]
invalid=0
with rasterio.open(ROOT/'data/ASCENDING.tif') as src:
    transform=src.transform
    dst_crs=src.crs
    dims=src.shape
    bounds=src.bounds
    profile=src.profile.copy()
converter=pyproj.Transformer.from_crs(crs,dst_crs,always_xy=True).transform
for sh in reader.shapes():
    geom=shape(sh.__geo_interface__)
    if not geom.is_valid:
        invalid+=1
        geom=geom.buffer(0)
    geom=transform_geom(converter,geom)
    if geom.intersects(box(*bounds)):
        geoms.append(geom.intersection(box(*bounds)))
assert geoms,'Fixed pilot contains no reference polygons'
label=rasterize([(mapping(g),1) for g in geoms],out_shape=dims,transform=transform,
                fill=0,dtype='uint8',all_touched=False)
assert label.sum()>0
profile.update(count=1,dtype='uint8',compress='deflate',nodata=None)
profile.pop('photometric',None)
with rasterio.open(target/'reference_mask.tif','w',**profile) as dst:
    dst.write(label,1)
    dst.update_tags(description='Reference inventory rasterization, not model output',rasterization='pixel centres, all_touched=False')
features=[{'type':'Feature','properties':{'source':'SAR-LRA author Sumatra_2022 inventory'},'geometry':mapping(g)} for g in geoms]
(target/'pilot_inventory.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}),encoding='utf-8')
report={'source':'Author Zenodo record 15159492 Sumatra_2022.shp','source_crs':crs.to_string(),
        'output_crs':str(dst_crs),'total_source_features':len(reader),'intersecting_features':len(geoms),
        'geometry_repairs_total':invalid,'shape':list(dims),'reference_pixels':int(label.sum()),
        'reference_fraction':float(label.mean()),'all_touched':False,
        'source_shapefile_sha256':hashlib.sha256((target/'Sumatra_2022.shp').read_bytes()).hexdigest(),
        'caveats':['Original inventory, no pass-specific shadow/layover cleaning yet',
                   'Inventory absence treated as background for pilot scoring, not certified safe ground',
                   'Not independently audited against actual checkpoint training files; paper declares Sumatra unseen']}
(target/'reference_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
