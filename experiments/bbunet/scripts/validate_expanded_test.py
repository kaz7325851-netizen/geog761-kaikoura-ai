"""Reread predictions and original inventory/JP2, independently recompute diagnostics."""
import json,hashlib,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np,rasterio,shapefile
from rasterio.features import rasterize
from shapely.geometry import shape,box
from shapely.ops import transform,unary_union
from pyproj import Transformer,CRS
from scipy.ndimage import binary_erosion
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'expanded_test'/'20261007T210924Z'
report=json.loads((OUT/'expanded_results.json').read_text())
assert report['status']=='expanded_inference_complete_pending_independent_validation'
design=json.loads((OUT/'frozen_design.json').read_text())
assert hashlib.sha256((OUT/'frozen_design.json').read_bytes()).hexdigest()==report['design_sha256']
refpath=ROOT.parent/'sar_lra'/'kaikoura_exploration'/'reference'/'Kaikora_2016.shp'
project=Transformer.from_crs(CRS.from_wkt(refpath.with_suffix('.prj').read_text()),2193,always_xy=True).transform
reader=shapefile.Reader(str(refpath))
ids=set(x['source_index'] for r in report['regions'] for x in json.loads((OUT/r['region_id']/'inventory_objects.json').read_text()))
geoms={}
for i,s in enumerate(reader.iterShapes()):
    if i not in ids:continue
    g=transform(project,shape(s.__geo_interface__))
    geoms[i]=g if g.is_valid else g.buffer(0)
road=unary_union([transform(Transformer.from_crs(4326,2193,always_xy=True).transform,shape(f['geometry']))
    for f in json.loads((ROOT.parent/'sentinel1'/'study_route_lines.geojson').read_text())['features']
    if f['properties']['route'].startswith('SH1')])
bands=['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'];res=[60,10,10,10,20,20,20,10,20,60,20,20]
products={}
for key,name in zip(['pre','post'],design['source_products']):
    p=ROOT/(name+'.zip')
    with zipfile.ZipFile(p) as z:
        names=z.namelist();elems=list(ET.fromstring(z.read(next(n for n in names if n.endswith('/MTD_MSIL2A.xml')))).iter())
    offsets={int(e.attrib['band_id']):float(e.text) for e in elems if e.tag.split('}')[-1]=='BOA_ADD_OFFSET'}
    quant=float(next(e.text for e in elems if e.tag.split('}')[-1]=='BOA_QUANTIFICATION_VALUE'))
    bmap=dict(zip(['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'],range(13)))
    products[key]=(p,names,offsets,quant,bmap)
validation={'pass':True,'design_unchanged':True,'regions':[],'method':'Original inventory rerasterization, saved output counts, independent JP2 coordinate sampling, cloud/SCL and erosion reconstruction'}
rng=np.random.default_rng(1761)
def read(p):
    with rasterio.open(p) as d:return d.read(),d.profile.copy()
for result in report['regions']:
    rid=result['region_id'];d=OUT/rid
    for f in result['files']:assert hashlib.sha256(Path(f['file']).read_bytes()).hexdigest()==f['sha256']
    prob,profile=read(d/'probability_full.tif');prob=prob[0]
    ev=read(d/'evaluation_mask.tif')[0][0].astype(bool);er=read(d/'eroded_evaluation_mask.tif')[0][0].astype(bool)
    rd=read(d/'road100m_mask.tif')[0][0].astype(bool);ref=read(d/'inventory_raster.tif')[0][0].astype(bool)
    assert np.isfinite(prob).all() and 0<=prob.min()<=prob.max()<=1
    assert np.array_equal(er,binary_erosion(ev,structure=np.ones((3,3),bool),border_value=0))
    terrain=read(d/'terrain_physical.tif')[0];tv=np.isfinite(terrain).all(0)&(terrain!=-9999).all(0)
    expected=tv.copy();maxerr=0;sample_count=0;boundary_samples=[]
    for key in ['pre','post']:
        cl=read(d/f'{key}_SENCLOUD12.tif')[0][0];scl=read(d/f'{key}_SCL.tif')[0][0]
        expected&=(cl==0)&np.isin(scl,[4,5,7])
        boa=read(d/f'{key}_BOA.tif')[0];normalized=read(d/f'{key}_normalized.tif')[0]
        norm=np.clip(boa,0,1);norm[:,~((boa!=-9999).all(0))]=0
        assert np.array_equal(norm,normalized)
        p,names,offsets,quant,bmap=products[key]
        rows=rng.integers(0,640,25);cols=rng.integers(0,640,25)
        xs,ys=rasterio.transform.xy(profile['transform'],rows,cols)
        coords=list(zip(*Transformer.from_crs(2193,32759,always_xy=True).transform(xs,ys)))
        for b,resolution in zip(bands,res):
            member=next(n for n in names if n.endswith(f'_{b}_{resolution}m.jp2'))
            with rasterio.open('/vsizip/'+p.as_posix()+'/'+member) as src:
                dn=np.array([x[0] for x in src.sample(coords)],dtype='float32')
                with WarpedVRT(src,crs='EPSG:2193',transform=profile['transform'],width=640,height=640,
                    resampling=Resampling.nearest,src_nodata=0,nodata=0,dtype='uint16') as vrt:
                    vrt_dn=vrt.read(1)[rows,cols].astype('float32')
                for q in np.where(dn!=vrt_dn)[0]:
                    xc,yc=(~src.transform)*coords[q]
                    # GDAL's default approximate transformer has 0.125 source-pixel tolerance.
                    edge=min(xc%1,1-xc%1,yc%1,1-yc%1)
                    sr,sc=src.index(*coords[q]);window=src.read(1,window=((sr-1,sr+2),(sc-1,sc+2)))
                    assert vrt_dn[q] in window and edge<=.125,(rid,key,b,float(edge))
                    boundary_samples.append({'date':key,'band':b,'row':int(rows[q]),'col':int(cols[q]),
                        'direct_DN':float(dn[q]),'staged_DN':float(vrt_dn[q]),'source_boundary_distance_pixels':float(edge)})
            value=(dn+offsets[bmap[b]])/quant;value[(dn==0)|(dn==65535)]=-9999
            err=float(np.max(np.abs(value-boa[bands.index(b),rows,cols])));maxerr=max(maxerr,err)
            vrt_value=(vrt_dn+offsets[bmap[b]])/quant;vrt_value[(vrt_dn==0)|(vrt_dn==65535)]=-9999
            assert np.max(np.abs(vrt_value-boa[bands.index(b),rows,cols]))<2e-6,(rid,key,b,err)
            sample_count+=len(rows)
    assert np.array_equal(ev,expected)
    assert np.array_equal(read(d/'positive_screened.tif')[0][0],np.where(ev,prob>=.5,255).astype('uint8'))
    assert np.array_equal(read(d/'probability_screened.tif')[0][0],np.where(ev,prob,-9999))
    objs=json.loads((d/'inventory_objects.json').read_text());eobjs=json.loads((d/'erosion_inventory_objects.json').read_text());union=np.zeros(ev.shape,bool)
    for x,ex in zip(objs,eobjs):
        g=geoms[x['source_index']];pm=rasterize([(g.intersection(box(*result['bounds_nztm'])),1)],out_shape=ev.shape,transform=profile['transform'],dtype='uint8').astype(bool)
        union|=pm
        assert abs(g.area-x['source_area_m2'])<.01 and abs(g.distance(road)-x['distance_to_sh1_m'])<.01
        for m,obj in [(ev,x),(er,ex)]:
            assert int(pm.sum())==obj['rasterized_pixels']
            assert int((pm&m).sum())==obj['evaluated_pixels']
            assert int((pm&m&(prob>=.5)).sum())==obj['positive_pixels']
    assert np.array_equal(union,ref)
    for scope,key in [(np.ones(ev.shape,bool),'pixels'),(rd,'road_pixels')]:
        p=result[key];v=ev&scope;pos=v&(prob>=.5)
        assert p['evaluated_pixels']==int(v.sum()) and p['reference_pixels']==int((v&ref).sum())
        assert p['positive_within_reference']==int((pos&ref).sum())
        assert p['positive_outside_reference_UNKNOWN']==int((pos&~ref).sum())
    validation['regions'].append({'region_id':rid,'pass':True,'original_inventory_objects_rechecked':len(objs),
        'source_JP2_samples':sample_count,'max_direct_coordinate_reflectance_difference':maxerr,
        'GDAL_approximate_warp_boundary_samples':boundary_samples,'staged_reprojection_matches':True,
        'file_hashes_checked':len(result['files'])})
    print(f'{rid}: independent validation passed, {len(objs)} objects + {sample_count} JP2 samples',flush=True)
assert len(report['regions'])==8
report['status']='expanded_test_complete_independently_rechecked'
report['independent_output_validation']=validation
(OUT/'independent_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
(OUT/'expanded_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'pooled':report['pooled_unique_objects'],'gate':report['recommendation_gate'],'anchor':report['anchor_control']},indent=2),flush=True)
