"""Frozen multi-region BBUnet exploratory test; no training or road-status claims."""
import os,sys,json,hashlib,zipfile,importlib.util,time
from pathlib import Path
from argparse import Namespace
from xml.etree import ElementTree as ET
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'expanded_test'/'20261007T210924Z'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'/'matplotlib'))
sys.path.insert(0,str(ROOT/'runtime_smp033'))
import numpy as np
import torch,rasterio,shapefile
import segmentation_models_pytorch as smp
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.features import rasterize
from shapely.geometry import shape,box,mapping
from shapely.ops import transform,unary_union
from shapely.strtree import STRtree
from pyproj import Transformer,CRS
from scipy.ndimage import binary_erosion
from richdem_numpy_audit import terrain_derivatives,encode_terrain,run_tests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def write_json(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
design=json.loads((OUT/'frozen_design.json').read_text())
regions=design['regions']+[design['anchor']]
BANDS=['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']
NATIVE=[60,10,10,10,20,20,20,10,20,60,20,20]
NODATA=-9999.
torch.set_num_threads(2);torch.manual_seed(761)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
device='cuda' if torch.cuda.is_available() else 'cpu'
wp=ROOT/'weights'/'bbunet_r18.pt';cp=ROOT/'weights'/'mobilenetv2_l2a_all.pth'
assert sha(wp)=='a5f2b9ec72d920daf5254d1eddafa8ee3a0bebd0430078cb00b943a73f669152'
assert sha(cp)=='2f3221b0ce454679d171b1a2141730e5708397df302e06e4601f54ac5ec9f228'
with torch.serialization.safe_globals([Namespace]):ckpt=torch.load(wp,map_location='cpu',weights_only=True)
hp=vars(ckpt['hparams']);assert (hp['IMAGE_SIZE'],hp['STRIDE'],hp['THRESHOLD'])==(256,128,.5)
author=ROOT/'author_source_ce34d3ac'/'igarss-landslide-delineation-ce34d3ac02b80d4db7e542203b7e534649a46352'/'src'/'models'/'bbunet.py'
spec=importlib.util.spec_from_file_location('author_bbunet',author)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
model=mod.BBUnet(s2_in_channels=12,dem_in_channels=4,classes=1)
model.load_state_dict(ckpt['state_dict'],strict=True);model.eval().to(device)
cloud_model=smp.Unet(encoder_name='mobilenet_v2',encoder_weights=None,classes=4,in_channels=13)
cloud_model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True)
cloud_model.eval().to(device)
report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running','device':device,
 'design_sha256':sha(OUT/'frozen_design.json'),'weights_sha256':sha(wp),'cloud_weights_sha256':sha(cp),
 'strict_weight_loading':True,'author_model_sha256':sha(author),'training':False,'fine_tuning':False,
 'threshold':.5,'patch_size':256,'stride':128,'patches_per_region':16,
 'erosion_sensitivity':'one 3x3 square binary erosion, including AOI border; 10m cardinal/14.14m diagonal',
 'terrain_derivative_audit':run_tests(),'independent_event_accuracy':False,'road_passability':False,
 'metrics_meaning':'Conditional reference-inventory overlap, NOT verified accuracy. Inventory absence is not a negative label.',
 'limitations':design['limitations']+['PC DSM has no MSK/STK quality masks; terrain validity is numerical only',
 'Native RichDEM binary not executed; audited NumPy formula port on reprojected DSM',
 'No supervised adaptation, independent training-overlap audit or verified road-status labels'],
 'regions':[]}
write_json(OUT/'expanded_results.json',report)

# Verify all source archives before staging any new model predictions.
products={}
for key in ['pre','post']:
    audit=json.loads((ROOT/'data_acquisition'/'20261007T095702Z'/f'{key}_archive_verification.json').read_text())
    p=ROOT/(audit['catalog_product']+'.zip');assert sha(p)==audit['local_hashes']['sha256']
    with zipfile.ZipFile(p) as z:
        names=z.namelist();xml=ET.fromstring(z.read(next(n for n in names if n.endswith('/MTD_MSIL2A.xml'))))
    elems=list(xml.iter());quant=float(next(e.text for e in elems if e.tag.split('}')[-1]=='BOA_QUANTIFICATION_VALUE'))
    offsets={int(e.attrib['band_id']):float(e.text) for e in elems if e.tag.split('}')[-1]=='BOA_ADD_OFFSET'}
    ids=dict(zip(['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'],range(13)))
    products[key]={'archive':p,'names':names,'quant':quant,'offsets':[offsets[ids[b]] for b in BANDS],
                   'sha256':audit['local_hashes']['sha256']}
terrain_records=json.loads((OUT/'sources'/'terrain_acquisition.json').read_text())['records']
for r in terrain_records:assert sha(r['file'])==r['sha256']
project=Transformer.from_crs(4326,2193,always_xy=True).transform
roads_json=json.loads((ROOT.parent/'sentinel1'/'study_route_lines.geojson').read_text())
road=unary_union([transform(project,shape(f['geometry'])) for f in roads_json['features'] if str(f['properties'].get('route','')).startswith('SH1')])
assert not road.is_empty
refpath=ROOT.parent/'sar_lra'/'kaikoura_exploration'/'reference'/'Kaikora_2016.shp'
refproj=Transformer.from_crs(CRS.from_wkt(refpath.with_suffix('.prj').read_text()),2193,always_xy=True).transform
reader=shapefile.Reader(str(refpath));polys=[];properties=[];repaired=0
whole=unary_union([box(*r['bounds_nztm']) for r in regions]).buffer(1000)
for idx,s in enumerate(reader.iterShapes()):
    g=transform(refproj,shape(s.__geo_interface__))
    if not g.is_valid:g=g.buffer(0);repaired+=1
    if g.is_empty or not g.intersects(whole):continue
    polys.append(g);properties.append({'source_index':idx,'source_area_m2':g.area,'distance_to_sh1_m':g.distance(road)})
tree=STRtree(polys)
report['reference_source']={'shp_sha256':sha(refpath),'total_source_records':len(reader),'corridor_polygons':len(polys),'repaired_source_geometries':repaired}
write_json(OUT/'expanded_results.json',report)
print(f'SOURCE QA complete: {len(polys)} corridor reference objects; device={device}',flush=True)

def groups(objects):
    ans={}
    for name in ['all','near_road100m','large_near_road100m','small_near_road100m']:
        selected=[x for x in objects if (name=='all' or x['distance_to_sh1_m']<=100) and
            (name!='large_near_road100m' or x['source_area_m2']>=10000) and
            (name!='small_near_road100m' or 100<=x['source_area_m2']<10000)]
        ev=[x for x in selected if x['evaluated_pixels']>0]
        ans[name]={'source_objects':len(selected),'evaluated_objects':len(ev),'unassessed_objects':len(selected)-len(ev),
            'any_positive':sum(x['positive_pixels']>0 for x in ev),
            'ge10pct':sum(x['positive_pixels']/x['evaluated_pixels']>=.1 for x in ev),
            'ge50pct':sum(x['positive_pixels']/x['evaluated_pixels']>=.5 for x in ev)}
    return ans

for region in regions:
    start=time.monotonic();rid=region['region_id'];dest=OUT/rid;dest.mkdir(exist_ok=True)
    grid=Affine(*region['transform']);size=tuple(region['shape']);aoi=box(*region['bounds_nztm'])
    profile={'driver':'GTiff','height':size[0],'width':size[1],'crs':'EPSG:2193','transform':grid,'compress':'deflate','tiled':True}
    files=[]
    def save(name,arr,nodata=None,descriptions=None):
        if arr.ndim==2:arr=arr[None]
        p=dest/name
        with rasterio.open(p,'w',**profile,count=len(arr),dtype=str(arr.dtype),nodata=nodata) as d:
            d.write(arr)
            if descriptions:d.descriptions=tuple(descriptions)
            d.update_tags(warning='Exploratory landslide detection, NOT road passability')
        with rasterio.open(p) as d:assert np.array_equal(d.read(),arr) and d.transform==grid and d.crs.to_epsg()==2193
        files.append({'file':str(p),'sha256':sha(p),'readback':True})
    def mask(g):return rasterize([(g,1)],out_shape=size,transform=grid,dtype='uint8').astype(bool) if not g.is_empty else np.zeros(size,bool)
    roadmask=mask(road.intersection(aoi.buffer(100)).buffer(100))
    spectral={};clouds={};scls={};input_audit={}
    for key in ['pre','post']:
        product=products[key]
        def band(b,res):
            member=next(n for n in product['names'] if n.endswith(f'_{b}_{res}m.jp2'))
            with rasterio.open('/vsizip/'+product['archive'].as_posix()+'/'+member) as src:
                with WarpedVRT(src,crs='EPSG:2193',transform=grid,width=size[1],height=size[0],resampling=Resampling.nearest,src_nodata=0,nodata=0,dtype='uint16') as vrt:
                    return vrt.read(1)
        dn=np.stack([band(b,res) for b,res in zip(BANDS,NATIVE)])
        ok=((dn!=0)&(dn!=65535)).all(axis=0)
        boa=(dn.astype('float32')+np.asarray(product['offsets'],dtype='float32')[:,None,None])/product['quant']
        boa[(dn==0)|(dn==65535)]=NODATA
        scl=band('SCL',20).astype('uint8');aux=np.stack([band(b,20) for b in ['AOT','WVP']])
        cloudvalid=ok&(aux>0).all(axis=0)
        ca=np.maximum(boa[[i for i,b in enumerate(BANDS) if b!='B08']],0);ca[:,~ok]=1
        ax=aux.astype('float32')/65535;ax[aux==0]=1
        t=torch.from_numpy(np.concatenate([ca,ax])[None]).to(device)
        t=torch.nn.functional.pad(t,(16,16,16,16),mode='reflect')
        with torch.inference_mode():cl=cloud_model(t).argmax(1)[0,16:-16,16:-16].cpu().numpy().astype('uint8')
        cl[~cloudvalid]=255
        norm=np.clip(boa,0,1);norm[:,~ok]=0
        spectral[key]=norm;clouds[key]=cl;scls[key]=scl
        save(f'{key}_BOA.tif',boa,NODATA,BANDS);save(f'{key}_normalized.tif',norm,None,BANDS)
        save(f'{key}_SCL.tif',scl,0);save(f'{key}_SENCLOUD12.tif',cl,255)
        input_audit[key]={'source_sha256':product['sha256'],'source_valid_fraction':float(ok.mean()),
            'road_source_valid_fraction':float(ok[roadmask].mean()),'cloud_clear_fraction':float((cl==0).mean()),
            'road_cloud_clear_fraction':float((cl[roadmask]==0).mean()),
            'SCL_histogram':{str(i):int((scl==i).sum()) for i in range(12)}}
        print(f'{rid}: {key} L2A + cloud staging complete',flush=True)
    coarse=Affine(30,0,grid.c-90,0,-30,grid.f+90);dem=np.full((220,220),NODATA,dtype='float32')
    for tr in terrain_records:
        temp=np.full(dem.shape,NODATA,dtype='float32')
        with rasterio.open(tr['file']) as src:
            reproject(rasterio.band(src,1),temp,src_transform=src.transform,src_crs=src.crs,
                src_nodata=None,dst_transform=coarse,dst_crs='EPSG:2193',dst_nodata=NODATA,resampling=Resampling.nearest)
        dem[temp!=NODATA]=temp[temp!=NODATA]
    dem=np.rint(dem);sl,asp=terrain_derivatives(dem,30,30)
    physical=[]
    for arr,res in [(dem,Resampling.bilinear),(sl,Resampling.bilinear),(asp,Resampling.nearest)]:
        temp=np.full(size,NODATA,dtype='float32')
        reproject(arr,temp,src_transform=coarse,src_crs='EPSG:2193',src_nodata=NODATA,dst_transform=grid,
            dst_crs='EPSG:2193',dst_nodata=NODATA,resampling=res)
        physical.append(temp)
    physical=np.stack(physical);tv=np.isfinite(physical).all(axis=0)&(physical!=NODATA).all(axis=0)
    terrain=encode_terrain(*physical);terrain[:,~tv]=0
    save('terrain_physical.tif',physical,NODATA,['DSM_m','slope_deg','aspect_deg'])
    save('terrain_encoded.tif',terrain,None,['DSM_div3000','slope_div90','half_angle_sin','half_angle_cos'])
    valid=tv&(clouds['pre']==0)&(clouds['post']==0)&np.isin(scls['pre'],[4,5,7])&np.isin(scls['post'],[4,5,7])
    eroded=binary_erosion(valid,structure=np.ones((3,3),bool),iterations=1,border_value=0)
    total=np.zeros(size,'float64');sq=total.copy();count=np.zeros(size,'uint8');repeat=None
    for j,(r,c) in enumerate((r,c) for r in range(0,385,128) for c in range(0,385,128)):
        ts=[torch.from_numpy(a[:,r:r+256,c:c+256][None]).to(device) for a in [spectral['pre'],spectral['post'],terrain]]
        with torch.inference_mode():
            p=model(*ts).sigmoid()[0,0].cpu().numpy()
            if j==0:repeat=float(np.max(np.abs(p-model(*ts).sigmoid()[0,0].cpu().numpy())))
        assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
        total[r:r+256,c:c+256]+=p;sq[r:r+256,c:c+256]+=p.astype('float64')**2;count[r:r+256,c:c+256]+=1
    assert (count>0).all();prob=(total/count).astype('float32');pos=(prob>=.5)&valid
    std=np.sqrt(np.maximum(sq/count-(total/count)**2,0)).astype('float32')
    indexes=tree.query(aoi,predicate='intersects');objects=[];erosion_objects=[];union=np.zeros(size,bool)
    for i in indexes:
        i=int(i);pm=mask(polys[i].intersection(aoi));union|=pm
        item={**properties[i],'region_id':rid,'rasterized_pixels':int(pm.sum())}
        for m,collection in [(valid,objects),(eroded,erosion_objects)]:
            vm=pm&m;n=int(vm.sum());pn=int((vm&(prob>=.5)).sum())
            collection.append({**item,'evaluated_pixels':n,'positive_pixels':pn,'coverage_fraction':pn/n if n else None,
                'retained_fraction':n/item['rasterized_pixels'] if item['rasterized_pixels'] else None,
                'probability_max':float(prob[vm].max()) if n else None})
    def pixels(m,scope):
        vm=m&scope;rv=vm&union;pv=vm&(prob>=.5)
        return {'evaluated_pixels':int(vm.sum()),'reference_pixels':int(rv.sum()),'positive_pixels':int(pv.sum()),
            'positive_within_reference':int((pv&union).sum()),'positive_outside_reference_UNKNOWN':int((pv&~union).sum()),
            'reference_pixel_coverage':float((pv&union).sum()/rv.sum()) if rv.any() else None}
    result={**region,'seconds':time.monotonic()-start,'inputs':input_audit,
        'terrain_valid_fraction':float(tv.mean()),'quality_fraction_aoi':float(valid.mean()),
        'quality_fraction_road100m':float(valid[roadmask].mean()),'road100m_pixels':int(roadmask.sum()),
        'objects':groups(objects),'mask_erosion_objects':groups(erosion_objects),
        'pixels':pixels(valid,np.ones(size,bool)),'road_pixels':pixels(valid,roadmask),
        'mask_erosion_road_pixels':pixels(eroded,roadmask),'repeat_max_abs_error':repeat,
        'overlap_std_road_p50_p95_max':np.percentile(std[valid&roadmask],[50,95,100]).tolist() if (valid&roadmask).any() else None}
    for fn,a,nd in [('probability_full.tif',prob,None),('probability_screened.tif',np.where(valid,prob,NODATA).astype('float32'),NODATA),
        ('positive_screened.tif',np.where(valid,prob>=.5,255).astype('uint8'),255),('evaluation_mask.tif',valid.astype('uint8'),None),
        ('eroded_evaluation_mask.tif',eroded.astype('uint8'),None),('inventory_raster.tif',union.astype('uint8'),None),
        ('road100m_mask.tif',roadmask.astype('uint8'),None),('overlap_std.tif',std,None)]:save(fn,a,nd)
    result['files']=files
    write_json(dest/'inventory_objects.json',objects);write_json(dest/'erosion_inventory_objects.json',erosion_objects)
    write_json(dest/'region_results.json',result);report['regions'].append(result)
    write_json(OUT/'expanded_results.json',report)
    print(json.dumps({'region':rid,'seconds':result['seconds'],'road_objects':result['objects']['near_road100m'],
        'large_road_objects':result['objects']['large_near_road100m'],'quality_road':result['quality_fraction_road100m']}),flush=True)

# Aggregate source IDs, not duplicated polygon fragments. Anchor never enters new scores.
for source_file,label in [('inventory_objects.json','pooled_unique_objects'),('erosion_inventory_objects.json','pooled_eroded_unique_objects')]:
    pooled={}
    for r in design['regions']:
        for x in json.loads((OUT/r['region_id']/source_file).read_text()):
            i=x['source_index']
            if i not in pooled:pooled[i]={**x,'region_ids':[x['region_id']]}
            else:
                p=pooled[i];p['region_ids'].append(x['region_id'])
                for k in ['rasterized_pixels','evaluated_pixels','positive_pixels']:p[k]+=x[k]
    for x in pooled.values():
        x['coverage_fraction']=x['positive_pixels']/x['evaluated_pixels'] if x['evaluated_pixels'] else None
        x['retained_fraction']=x['evaluated_pixels']/x['rasterized_pixels'] if x['rasterized_pixels'] else None
    report[label]=groups(list(pooled.values()))
    write_json(OUT/(label+'.json'),list(pooled.values()))
new=report['regions'][:-1]
report['pooled_pixels']={k:sum(r['pixels'][k] for r in new) for k in ['evaluated_pixels','reference_pixels','positive_pixels','positive_within_reference','positive_outside_reference_UNKNOWN']}
report['pooled_road_pixels']={k:sum(r['road_pixels'][k] for r in new) for k in report['pooled_pixels']}
for key in ['pooled_pixels','pooled_road_pixels']:
    a=report[key];a['reference_pixel_coverage']=a['positive_within_reference']/a['reference_pixels'] if a['reference_pixels'] else None
support=[r for r in new if r['objects']['near_road100m']['evaluated_objects']>=10]
near=report['pooled_unique_objects']['near_road100m'];large=report['pooled_unique_objects']['large_near_road100m']
gate={'regions_with_ge10_road_objects':len(support),'near_ge10pct_fraction':near['ge10pct']/near['evaluated_objects'] if near['evaluated_objects'] else None,
    'large_ge10pct_fraction':large['ge10pct']/large['evaluated_objects'] if large['evaluated_objects'] else None,
    'supported_regions_below_half':[r['region_id'] for r in support if r['objects']['near_road100m']['ge10pct']/r['objects']['near_road100m']['evaluated_objects']<.5]}
gate['pass']=bool(len(support)>=4 and (gate['near_ge10pct_fraction'] or 0)>=.7 and (gate['large_ge10pct_fraction'] or 0)>=.8 and not gate['supported_regions_below_half'])
gate['meaning']='Locally predeclared exploratory screening gate, NOT author/course/operational standard or proof of road safety'
report['recommendation_gate']=gate
anchor=OUT/'PILOT-ANCHOR'
oldinputs=ROOT/'model_inputs'/'20261007T203029Z';oldresults=ROOT/'pilot_results'/'20261007T203251Z'
def read(p):
    with rasterio.open(p) as d:return d.read()
same={k:bool(np.array_equal(read(anchor/f'{k}_normalized.tif'),read(oldinputs/f'{k}_optical_12channels_normalized.tif'))) for k in ['pre','post']}
pv=read(anchor/'probability_full.tif')[0];op=read(oldresults/'Horn_numpy_probability_full_context.tif')[0]
cv=read(anchor/'evaluation_mask.tif')[0].astype(bool)&read(oldresults/'evaluation_mask.tif')[0].astype(bool)
report['anchor_control']={'same_normalized_optical':same,'common_evaluated_pixels':int(cv.sum()),
    'probability_MAE_common':float(np.abs(pv[cv]-op[cv]).mean()),'binary_disagreement_common':float(((pv[cv]>=.5)!=(op[cv]>=.5)).mean()),
    'meaning':'Same frozen pilot and verified same optical inputs; new PC terrain compared with prior GEE terrain. Not new-test score.'}
report['status']='expanded_inference_complete_pending_independent_validation'
write_json(OUT/'expanded_results.json',report)
fig,axs=plt.subplots(2,4,figsize=(18,10))
for ax,r in zip(axs.flat,report['regions']):
    d=OUT/r['region_id'];rgb=np.clip(read(d/'post_normalized.tif')[[3,2,1]].transpose(1,2,0)/.3,0,1)
    v=read(d/'evaluation_mask.tif')[0].astype(bool);p=read(d/'probability_full.tif')[0];ref=read(d/'inventory_raster.tif')[0];rd=read(d/'road100m_mask.tif')[0]
    ax.imshow(rgb);overlay=np.zeros((*v.shape,4));overlay[v&(p>=.5)]=[1,0,0,.75];ax.imshow(overlay)
    if ref.any():ax.contour(ref,levels=[.5],colors=['yellow'],linewidths=.35)
    if rd.any():ax.contour(rd,levels=[.5],colors=['cyan'],linewidths=.4)
    n=r['objects']['near_road100m'];ax.set_title(f"{r['region_id']} | road objects >=10%: {n['ge10pct']}/{n['evaluated_objects']}\nRoad quality retained: {r['quality_fraction_road100m']:.1%}",fontsize=11);ax.set_axis_off()
fig.suptitle('BBUnet expanded frozen test: 7 NEW spatial regions + original pilot CONTROL\nRed: screened positive | yellow: inventory | cyan: current SH1 context | NOT road passability',fontsize=14)
fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(OUT/'expanded_overview.png',dpi=160);plt.close(fig)
print('EXPANDED INFERENCE COMPLETE '+str(OUT),flush=True)
