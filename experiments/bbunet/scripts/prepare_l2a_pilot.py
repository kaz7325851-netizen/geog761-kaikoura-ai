"""Stage audited L2A on the frozen SH1 grid. No training or model inference."""
from pathlib import Path
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
import hashlib
import json
import os
import zipfile
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform as transform_geom, unary_union
from pyproj import Transformer
import cv2
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / 'pilot_design'
DESIGN = PILOT / '20261005T200713Z'
AUDITS = ROOT / 'data_acquisition' / '20261007T095702Z'
OLD = PILOT / 'prepared_inputs' / '20261005T203524Z'
AUTHOR = ROOT / 'author_source_ce34d3ac' / 'igarss-landslide-delineation-ce34d3ac02b80d4db7e542203b7e534649a46352'
OUT = ROOT / 'prepared_inputs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True, exist_ok=False)
BANDS = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']
NATIVE = dict(zip(BANDS, [60,10,10,10,20,20,20,10,20,60,20,20]))
GRID = Affine(10,0,1667250,0,-10,5328030)
CRS = 'EPSG:2193'
SIZE = (640,640)
NODATA = -9999.0
design = json.loads((DESIGN/'frozen_pilot_design.json').read_text(encoding='utf-8'))
assert design['selected']['bounds_nztm'] == [1667250.0,5321630.0,1673650.0,5328030.0]
project = Transformer.from_crs(4326,2193,always_xy=True).transform
road = unary_union([transform_geom(project,shape(f['geometry'])) for f in json.loads((DESIGN/'sh1_pilot_road.geojson').read_text())['features']])
inventory = unary_union([transform_geom(project,shape(f['geometry'])) for f in json.loads((DESIGN/'reference_inventory.geojson').read_text())['features']])
def mask(geom):
    return rasterize([(geom,1)],out_shape=SIZE,transform=GRID,dtype='uint8').astype(bool)
roadmask = mask(road.buffer(100))
inventorymask = mask(inventory.buffer(60))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

files = []
def save_tif(name, arr, descriptions, nodata, **tags):
    if arr.ndim == 2:
        arr = arr[None]
    path = OUT/name
    with rasterio.open(path,'w',driver='GTiff',height=SIZE[0],width=SIZE[1],
                       count=len(arr),dtype=arr.dtype,crs=CRS,transform=GRID,
                       nodata=nodata,compress='deflate',tiled=True) as dst:
        dst.write(arr)
        dst.descriptions = tuple(descriptions)
        dst.update_tags(stage='data_preparation_not_model_prediction', **tags)
    with rasterio.open(path) as src:
        assert src.crs.to_string() == CRS and src.transform == GRID
        assert src.shape == SIZE and src.descriptions == tuple(descriptions)
        assert np.array_equal(src.read(),arr)
    record = {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,
              'shape':list(arr.shape),'bands':descriptions,'nodata':nodata,
              'readback_verified':True}
    files.append(record)
    return record

records = {}
arrays = {}
for key,nzdate in [('pre','20161103'),('post','20161123')]:
    audit = json.loads((AUDITS/f'{key}_archive_verification.json').read_text(encoding='utf-8'))
    archive = ROOT/(audit['catalog_product']+'.zip')
    assert all(audit[x] for x in ['md5_matches','byte_count_matches','zip_crc_passed','safe_root_matches'])
    assert archive.stat().st_size == audit['bytes'] and sha(archive) == audit['local_hashes']['sha256']
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        product_xml_name = next(n for n in names if n.endswith('/MTD_MSIL2A.xml'))
        product_xml = z.read(product_xml_name)
        (OUT/f'{key}_MTD_MSIL2A.xml').write_bytes(product_xml)
        tile_xml_name = next(n for n in names if n.endswith('/MTD_TL.xml'))
        (OUT/f'{key}_MTD_TL.xml').write_bytes(z.read(tile_xml_name))
        tree = ET.fromstring(product_xml)
        elements = list(tree.iter())
        get_text = lambda tag: next(e.text for e in elements if e.tag.split('}')[-1] == tag)
        quant = float(get_text('BOA_QUANTIFICATION_VALUE'))
        offsets = {int(e.attrib['band_id']):float(e.text) for e in elements if e.tag.split('}')[-1]=='BOA_ADD_OFFSET'}
        spectral_ids = dict(zip(['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'],range(13)))
        assert get_text('PRODUCT_URI') == audit['catalog_product']
        member_records = []
        def read_member(band, resolution):
            matches = [n for n in names if n.endswith(f'_{band}_{resolution}m.jp2')]
            assert len(matches) == 1, matches
            member = matches[0]
            with rasterio.open('/vsizip/'+archive.as_posix()+'/'+member) as src:
                assert src.crs.to_epsg()==32759 and src.count==1
                with WarpedVRT(src,crs=CRS,transform=GRID,width=SIZE[1],height=SIZE[0],
                               resampling=Resampling.nearest,src_nodata=0,nodata=0,
                               dtype='uint16') as vrt:
                    a = vrt.read(1)
                member_records.append({'band':band,'member':member,'native_resolution_m':resolution,
                                       'source_crs':src.crs.to_string(),'source_shape':list(src.shape),
                                       'source_transform':list(src.transform)[:6], 'resampling':'nearest'})
            print(f'{key} {band} ready',flush=True)
            return a
        dn = np.stack([read_member(b,NATIVE[b]) for b in BANDS])
        valid_per_band = (dn!=0)&(dn!=65535)
        valid = np.all(valid_per_band,axis=0)
        offset_vector = np.array([offsets[spectral_ids[b]] for b in BANDS],dtype='float32')
        boa = (dn.astype('float32')+offset_vector[:,None,None])/quant
        boa[~valid_per_band] = NODATA
        scl = read_member('SCL',20).astype('uint8')
        aux = np.stack([read_member('AOT',20),read_member('WVP',20)])
    save_tif(f'{key}_{nzdate}NZ_L2A_original_DN_12bands.tif',dn,BANDS,0,
             role='audit_raw_DN_not_normalized',source_product=audit['catalog_product'])
    save_tif(f'{key}_{nzdate}NZ_L2A_BOA_12bands.tif',boa,BANDS,NODATA,
             role='physical_reflectance_not_final_model_tensor',
             formula='(valid_DN+metadata_BOA_ADD_OFFSET)/metadata_BOA_QUANTIFICATION_VALUE',
             source_product=audit['catalog_product'],resampling='nearest',
             normalization='none; valid negative reflectance retained')
    save_tif(f'{key}_SCL_20m_to10m_nearest.tif',scl,['SCL'],0,
             role='Sen2Cor_SCL_not_author_SENCLOUD12',source_resolution_m=20)
    save_tif(f'{key}_AOT_WVP_DN_20m_to10m_nearest.tif',aux,['AOT_DN','WVP_DN'],0,
             role='raw_auxiliary_DN_for_future_cloud_pipeline',
             AOT_QUANTIFICATION_VALUE=get_text('AOT_QUANTIFICATION_VALUE'),
             WVP_QUANTIFICATION_VALUE=get_text('WVP_QUANTIFICATION_VALUE'))
    save_tif(f'{key}_spectral_validity.tif',valid.astype('uint8'),['all12_DN_valid'],None)
    strict = valid & np.isin(scl,[4,5,7])
    arrays[key] = {'dn':dn,'boa':boa,'valid':valid,'strict':strict,'scl':scl}
    records[key] = {'product':audit['catalog_product'],'archive_sha256':audit['local_hashes']['sha256'],
                    'nz_date':nzdate,'quantification':quant,'offsets':offset_vector.tolist(),
                    'members':member_records,
                    'spectral_valid_aoi_fraction':float(valid.mean()),
                    'spectral_valid_road100m_fraction':float(valid[roadmask].mean()),
                    'strict_SCL_aoi_fraction':float(strict.mean()),
                    'strict_SCL_road100m_fraction':float(strict[roadmask].mean()),
                    'SCL_histogram_aoi':{str(i):int((scl==i).sum()) for i in range(12)},
                    'SCL_histogram_road100m':{str(i):int(((scl==i)&roadmask).sum()) for i in range(12)},
                    'boa_percentiles':{b:np.percentile(boa[i][valid_per_band[i]],[0,1,50,99,100]).tolist() for i,b in enumerate(BANDS)},
                    'negative_reflectance_fraction':{b:float((boa[i][valid_per_band[i]]<0).mean()) for i,b in enumerate(BANDS)}}

# Reuse original terrain staging as an explicitly provisional source, not RichDEM reproduction.
terrain_path = OLD/'ALOS_terrain_30m_audit.tif'
terrain_manifest = json.loads((OLD/'input_manifest.json').read_text(encoding='utf-8'))
expected_terrain_sha = next(r['sha256'] for r in terrain_manifest['records'] if r['key']=='terrain_audit')
assert sha(terrain_path) == expected_terrain_sha
terrain = []
with rasterio.open(terrain_path) as src:
    for band,resampling in [(1,Resampling.bilinear),(2,Resampling.bilinear),(3,Resampling.nearest)]:
        dest = np.full(SIZE,NODATA,dtype='float32')
        reproject(rasterio.band(src,band),dest,src_transform=src.transform,src_crs=src.crs,
                  src_nodata=src.nodata,dst_transform=GRID,dst_crs=CRS,dst_nodata=NODATA,
                  resampling=resampling)
        terrain.append(dest)
    msk = np.zeros(SIZE,dtype='uint8')
    reproject(rasterio.band(src,7),msk,src_transform=src.transform,src_crs=src.crs,
              dst_transform=GRID,dst_crs=CRS,resampling=Resampling.nearest)
terrain = np.stack(terrain)
terrain_valid = np.all(np.isfinite(terrain)&(terrain!=NODATA),axis=0)&(msk!=1)
save_tif('terrain_PROVISIONAL_DEM_slope_aspect.tif',terrain,['DSM_m','slope_deg','aspect_deg'],NODATA,
         role='provisional_existing_GEE_derivatives_not_author_RichDEM',
         source_sha256=expected_terrain_sha,DEM_slope_resampling='bilinear',aspect_resampling='nearest')
save_tif('terrain_source_MSK.tif',msk,['AW3D30_MSK'],None)
joint_spectral = arrays['pre']['valid'] & arrays['post']['valid'] & terrain_valid
joint_strict = arrays['pre']['strict'] & arrays['post']['strict'] & terrain_valid
save_tif('joint_spectral_terrain_validity.tif',joint_spectral.astype('uint8'),['joint_source_validity'],None)
save_tif('joint_SCL_conservative_land_quality.tif',joint_strict.astype('uint8'),['joint_SCL_4_5_7_and_source_valid'],None,
         rule='SCL in 4,5,7 in BOTH dates; excludes dark/uncertain2,shadow3,water6,cloud8/9,cirrus10,snow11',
         warning='Screening only; not author cloud mask, not proof of road passability')
save_tif('road_100m_context_mask.tif',roadmask.astype('uint8'),['current_OSM_SH1_100m_context'],None,
         warning='Current OSM road geometry; not verified 2016 network')

# Diagnostic only: no translations are applied to optical imagery.
registration = []
stable = joint_strict & ~inventorymask
gray = {k:np.clip(arrays[k]['boa'][3],0,1).astype('float32') for k in ['pre','post']}
for row in [0,192,384]:
    for col in [0,192,384]:
        w = (slice(row,row+256),slice(col,col+256))
        m = stable[w].astype('uint8')
        item = {'row':row,'col':col,'stable_fraction':float(m.mean())}
        if m.mean()<0.5:
            item['status']='skipped_less_than_half_stable_screened_land'
        else:
            try:
                cc,warp = cv2.findTransformECC(gray['pre'][w].copy(),gray['post'][w].copy(),
                   np.eye(2,3,dtype='float32'),cv2.MOTION_TRANSLATION,
                   (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-5),m,5)
                item.update(status='diagnostic_fit',correlation=float(cc),
                            dx_pixels=float(warp[0,2]),dy_pixels=float(warp[1,2]),
                            high_correlation_small_shift=bool(cc>=0.85 and np.linalg.norm(warp[:,2])<2))
            except cv2.error:
                item['status']='ECC_failed_to_converge'
        registration.append(item)

def lines(ax,geom,color,width):
    parts = list(geom.geoms) if hasattr(geom,'geoms') else [geom]
    for part in parts:
        if part.geom_type=='Polygon':
            xy=np.array(part.exterior.coords)
        elif part.geom_type=='LineString':
            xy=np.array(part.coords)
        else:
            continue
        ax.plot(xy[:,0],xy[:,1],color=color,lw=width)
extent = [1667250,1673650,5321630,5328030]
fig,axs=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
for j,k in enumerate(['pre','post']):
    rgb=np.clip(arrays[k]['boa'][[3,2,1]].transpose(1,2,0)/0.3,0,1)**(1/1.6)
    axs[0,j].imshow(rgb,extent=extent)
    axs[0,j].set_title(f'{k.upper()} L2A BOA RGB\nFixed display stretch; cyan=SH1, yellow=inventory',fontsize=10)
    lines(axs[0,j],road,'cyan',1)
    lines(axs[0,j],inventory,'yellow',0.4)
    cmap=ListedColormap(['black','red','#555555','#993366','#218c35','#ac864e','#217dcc','#cccccc','#bbbbff','white','#7d00e0','#00ffff'])
    axs[1,j].imshow(arrays[k]['scl'],extent=extent,cmap=cmap,norm=BoundaryNorm(np.arange(-.5,12.5),12))
    axs[1,j].set_title(f'{k.upper()} SCL screening\nDark/uncertain2, shadow3, water6 excluded',fontsize=10)
    lines(axs[1,j],road,'cyan',1)
axs[0,2].imshow(joint_strict,extent=extent,cmap=ListedColormap(['#444444','#59b65f']))
axs[0,2].set_title('Conservative SCL quality, both dates\nGreen=passes screening; NOT a prediction',fontsize=10)
lines(axs[0,2],road,'cyan',1)
axs[1,2].imshow(terrain[0],extent=extent,cmap='terrain',vmin=0,vmax=1200)
axs[1,2].set_title('Provisional ALOS DSM\nNOT author terrain reproduction',fontsize=10)
lines(axs[1,2],road,'cyan',1)
for ax in axs.ravel():
    ax.set_xlabel('NZTM easting (m)');ax.set_ylabel('NZTM northing (m)')
    ax.ticklabel_format(style='plain',axis='both',useOffset=False)
fig.suptitle('Frozen SH1 optical pilot | 2016-11-03 / 2016-11-23 NZ | DATA QA, NOT PREDICTIONS',fontsize=13)
fig.savefig(OUT/'L2A_quality_overview.png',dpi=140)
plt.close(fig)

source_paths = ['src/arg_parser.py','src/datasets.py','src/misc.py','src/cloud_detector.py','src/model_utils.py','processing/processing_japan.ipynb','configs/bbunet.yaml']
manifest = {
    'created_utc':datetime.now(timezone.utc).isoformat(),'output_directory':str(OUT),
    'status':'staging_complete_readback_verified_NOT_model_ready','pilot_id':design['pilot_id'],
    'design_sha256':sha(DESIGN/'frozen_pilot_design.json'),
    'crs':CRS,'transform':list(GRID)[:6],'shape':list(SIZE),'optical_band_order':BANDS,
    'grid_alignment_verified':True,'records':records,'files':files,
    'joint_source_valid_aoi_fraction':float(joint_spectral.mean()),
    'joint_source_valid_road100m_fraction':float(joint_spectral[roadmask].mean()),
    'joint_SCL_conservative_aoi_fraction':float(joint_strict.mean()),
    'joint_SCL_conservative_road100m_fraction':float(joint_strict[roadmask].mean()),
    'road_context_pixels':int(roadmask.sum()),
    'terrain':{'source':str(terrain_path),'sha256':expected_terrain_sha,'derivatives':'existing GEE, not RichDEM',
               'gapfilled_MSK48_aoi_fraction':float((msk==48).mean()),
               'gapfilled_MSK48_road100m_fraction':float((msk[roadmask]==48).mean()),
               'author_resampling_style':'DEM/slope bilinear, aspect nearest, but derivatives not reproduced'},
    'registration':{'method':'local red-band ECC translation, no shifts applied',
                    'window_size':256,'land_stable_rule':'joint conservative SCL outside inventory buffered60m',
                    'results':registration,'independent_geolocation_validation':False,
                    'warning':'Heuristic only; seasonal/radiometric changes or real terrain change may bias fit'},
    'author_code_commit':'ce34d3ac02b80d4db7e542203b7e534649a46352',
    'author_source_hashes':{p:sha(AUTHOR/p) for p in source_paths},
    'model_ready':False,'model_run':False,'training':False,
    'remaining':['Download and safely inspect actual bbunet_r18 checkpoint hyperparameters',
                 'Author terrain uses RichDEM; existing GEE derivatives and AW3D30 gap-fill differ',
                 'Author aspect encoding uses pi/360 rather than usual pi/180: preserve or explicitly test, never silently replace',
                 'Author cloud detector SENCLOUD12 not run; SCL rule here is not equivalent',
                 'Author baseline helper gates correction by acquisition date, not product baseline: never blindly apply to reprocessed 2016 N0500',
                 'Frozen NZTM grid differs from author native-UTM staging; spatial registration only screened heuristically',
                 'Current OSM roads and nearby inventory do not establish road blockage/safe-road labels',
                 'Cloud/shadow manual review and training-overlap/independent-test check still required']}
(OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
summary = {k:manifest[k] for k in ['output_directory','status','joint_source_valid_aoi_fraction',
                                'joint_source_valid_road100m_fraction','joint_SCL_conservative_aoi_fraction',
                                'joint_SCL_conservative_road100m_fraction','model_ready','model_run']}
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
