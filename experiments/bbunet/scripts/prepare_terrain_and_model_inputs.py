"""Generate versioned exploratory inputs with audited terrain formulas."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import rasterio
from rasterio.warp import reproject,Resampling
from richdem_numpy_audit import terrain_derivatives,scalar_reference,encode_terrain,run_tests
ROOT=Path(__file__).resolve().parent
STAGE=ROOT/'prepared_inputs'/'20261007T193918Z'
CLOUD=ROOT/'cloud_screen'/'20261007T200436Z'
SOURCE=ROOT.parent/'pilot_design'/'prepared_inputs'/'20261005T203524Z'/'ALOS_terrain_30m_audit.tif'
OUT=ROOT/'model_inputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True,exist_ok=False)
manifest=json.loads((STAGE/'input_manifest.json').read_text(encoding='utf-8'))
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==manifest['terrain']['sha256']
tests=run_tests()
with rasterio.open(STAGE/'terrain_PROVISIONAL_DEM_slope_aspect.tif') as ds:
    old=ds.read();profile=ds.profile.copy()
with rasterio.open(SOURCE) as ds:
    dem=ds.read(1);native_profile=ds.profile.copy();source_transform=ds.transform;source_crs=ds.crs
    source_valid=(dem!=ds.nodata)&np.isfinite(dem)
# Author notebook rounds DEM before deriving attributes. Preserve that step.
rounded=np.where(source_valid,np.rint(dem),-9999)
slope,aspect=terrain_derivatives(rounded,abs(source_transform.a),abs(source_transform.e))
ss,aa=scalar_reference(rounded,abs(source_transform.a),abs(source_transform.e))
assert np.allclose(slope,ss,atol=1e-4) and np.allclose(aspect,aa,atol=1e-4)
tests['real_source_cells_scalar_checked']=int(rounded.size)
tests['real_source_max_abs_error']=[float(np.max(np.abs(slope-ss))),float(np.max(np.abs(aspect-aa)))]

def save(name,array,descriptions,nd=-9999,base=profile):
    if array.ndim==2:array=array[None]
    p=base.copy();p.update(count=array.shape[0],dtype=str(array.dtype),nodata=nd,compress='deflate',photometric='minisblack')
    fn=OUT/name
    with rasterio.open(fn,'w',**p) as d:
        d.write(array);d.descriptions=tuple(descriptions)
        d.update_tags(meaning='Exploratory BBUnet inputs; NOT full author data reproduction')
    with rasterio.open(fn) as d:
        assert np.array_equal(d.read(),array) and d.transform==base['transform'] and d.crs==base['crs']
    return {'file':str(fn),'sha256':hashlib.sha256(fn.read_bytes()).hexdigest(),'shape':list(array.shape),'readback_verified':True}

native=np.stack([rounded,slope,aspect]).astype('float32')
native_record=save('terrain_30m_rounded_Horn_numpy.tif',native,['DSM_m_rounded','slope_deg','aspect_deg'],base=native_profile)
physical=np.full((3,640,640),-9999,dtype='float32')
for i,method in enumerate([Resampling.bilinear,Resampling.bilinear,Resampling.nearest]):
    reproject(native[i],physical[i],src_transform=source_transform,src_crs=source_crs,src_nodata=-9999,
              dst_transform=profile['transform'],dst_crs=profile['crs'],dst_nodata=-9999,resampling=method)
valid=(physical!=-9999).all(axis=0)&np.isfinite(physical).all(axis=0)
assert valid.all(),'Unexpected terrain gap: preserve and investigate rather than silently fill'
encoded=encode_terrain(*physical)
oldvalid=(old!=-9999).all(axis=0)&np.isfinite(old).all(axis=0)
assert oldvalid.all()
oldencoded=encode_terrain(*old)
assert np.isfinite(encoded).all()
records=[native_record,save('terrain_10m_physical_Horn_numpy.tif',physical,['DSM_m','slope_deg','aspect_deg']),
         save('terrain_4channels_author_encoding.tif',encoded,['DSM_div3000','slope_div90','aspect_sin_halfangle_01','aspect_cos_halfangle_01']),
         save('terrain_4channels_GEE_sensitivity.tif',oldencoded,['DSM_div3000','GEE_slope_div90','GEE_aspect_sin_halfangle_01','GEE_aspect_cos_halfangle_01'])]
for date in ['pre','post']:
    fn=f'{date}_{manifest["records"][date]["nz_date"]}NZ_L2A_BOA_12bands.tif'
    with rasterio.open(STAGE/fn) as ds:a=ds.read()
    with rasterio.open(STAGE/f'{date}_spectral_validity.tif') as ds:v=ds.read(1).astype(bool)
    a=np.clip(a,0,1);a[:,~v]=0
    records.append(save(f'{date}_optical_12channels_normalized.tif',a,manifest['optical_band_order'],nd=None))
with rasterio.open(STAGE/'joint_source_validity.tif') if (STAGE/'joint_source_validity.tif').exists() else rasterio.open(STAGE/'joint_spectral_terrain_validity.tif') as ds:
    joint_source=ds.read(1).astype(bool)
with rasterio.open(CLOUD/'joint_clear_only.tif') as ds:clear=ds.read(1).astype(bool)
with rasterio.open(STAGE/'joint_SCL_conservative_land_quality.tif') as ds:strict=ds.read(1).astype(bool)
evaluation=joint_source&clear&strict&valid
records.append(save('evaluation_conservative_mask.tif',evaluation.astype('uint8'),['source_valid_clear_SCL457_NOT_road_status'],nd=None))
report={'created_utc':datetime.now(timezone.utc).isoformat(),'directory':str(OUT),
        'input_stage':str(STAGE),'cloud_stage':str(CLOUD),'source_DSM':str(SOURCE),
        'source_DSM_sha256':manifest['terrain']['sha256'],'terrain_tests':tests,'files':records,
        'spectral_bands':manifest['optical_band_order'],'terrain_encoding':'DEM/3000,slope/90,(sin((aspect%360)*pi/360)+1)/2,(cos((aspect%360)*pi/360)+1)/2',
        'source_rounding':'np.rint to whole metres before Horn derivatives, as notebook',
        'derivative_calculation_grid':'existing GEE-resampled 30m NZTM DSM, not original author ALOS UTM grid',
        'RichDEM_native_binary_used':False,'full_author_pipeline_reproduced':False,
        'ready_for':'exploratory frozen-pilot inference, NOT strict reproduction or road passability',
        'terrain_valid_fraction':float(valid.mean()),'evaluation_mask_fraction':float(evaluation.mean()),
        'author_training_overlap_checked':False,'training':False,'model_run':False,
        'limitations':['Different original DSM version/gap-fill and projection remain',
                       'Terrain formulas audited against RichDEM source, but not compiled binary',
                       'Cloud detection and conservative land screen are predictions/screening not true labels',
                       'Source nodata optical padded with 0 for forward but excluded from evaluation; boundary predictions may be affected',
                       'Reference inventory not road blockage labels; background not verified negative']}
(OUT/'model_input_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'output':str(OUT),'tests':tests,'evaluation_fraction':float(evaluation.mean())},indent=2),flush=True)
