"""Change terrain derivatives ONLY; retain frozen prior RGB, masks and grids."""
from pathlib import Path
import json,hashlib
import numpy as np,rasterio
from rasterio.warp import reproject,Resampling
from rasterio.transform import Affine
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'pilot_exploration'
OUT=ROOT/'preprocessing_audit/terrain_aligned_pilot';OUT.mkdir(exist_ok=True)
TERR=ROOT/'preprocessing_audit/terrain'
DATA=ROOT.parent/'pilot_design/prepared_inputs/20261005T203524Z'
audit=json.loads((TERR/'terrain_algorithm_comparison.json').read_text())
for mode in ['slope','hillshade']:
    r=next(r for r in audit['records'] if r.get('algorithm')=='Horn' and r['mode']==mode and r['region']=='all_joint_valid')
    assert r['exact_equal_fraction']==1 and r['valid_mask_disagreement_pixels']==0
with rasterio.open(DATA/'ALOS_terrain_staged_10m.tif') as ds:
    profile=ds.profile.copy();profile.update(count=2,dtype='float32',nodata=-9999,compress='deflate')
    shape=(ds.height,ds.width);grid10=ds.transform;crs=ds.crs
layers=[]
for mode in ['hillshade','slope']:
    with rasterio.open(TERR/f'pilot_Horn_{mode}_30m.tif') as ds:
        src=ds.read(1);st=ds.transform;sn=ds.nodata
    dst=np.full(shape,-9999,dtype='float32')
    reproject(src,dst,src_transform=st,src_crs=crs,src_nodata=sn,
        dst_transform=grid10,dst_crs=crs,dst_nodata=-9999,resampling=Resampling.nearest)
    layers.append(dst)
with rasterio.open(OUT/'Horn_terrain_staged_10m.tif','w',**profile) as ds:
    ds.write(np.stack(layers));ds.set_band_description(1,'GDAL_Horn_hillshade_315_45');ds.set_band_description(2,'GDAL_Horn_slope_degrees')
manifest=json.loads((BASE/'frozen_input_manifest.json').read_text());records=[]
for prev in manifest['records']:
    p=Path(prev['file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==prev['sha256']
    with np.load(p) as z:arrays={k:z[k].copy() for k in z.files}
    warped=[]
    for a in layers:
        b=np.full((224,224),-9999,dtype='float32')
        reproject(a,b,src_transform=grid10,src_crs=crs,src_nodata=-9999,
            dst_transform=Affine(*prev['grid']),dst_crs=crs,dst_nodata=-9999,resampling=Resampling.cubic)
        assert np.all(b!=-9999) and np.isfinite(b).all()
        warped.append(b)
    hs=np.clip(warped[0],0,255);slope=np.clip(warped[1],0,90)
    arrays['topo']=np.stack([hs/127.5-1,slope/45-1],axis=-1).astype('float32')
    dest=OUT/(prev['tile_id']+'.npz');np.savez_compressed(dest,**arrays)
    record={**prev,'file':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'baseline_input_file':str(p),'baseline_input_sha256':prev['sha256'],
        'baseline_probability_file':str(BASE/(prev['tile_id']+'_raw_context_probability.npy')),
        'unchanged_except_topo':all(np.array_equal(arrays[k],np.load(p)[k]) for k in arrays if k!='topo'),
        'topo_mean_absolute_normalized_difference':np.mean(np.abs(arrays['topo']-np.load(p)['topo']),axis=(0,1)).tolist()}
    assert record['unchanged_except_topo'];records.append(record)
manifest.update({'records':records,'experiment':'derivative_algorithm_only_controlled_change',
    'author_sample_derivatives_numerically_reproduced':True,
    'terrain_process':'same pilot DEM -> GDAL3.0.2 Horn slope degrees and hillshade az315 alt45 scale1 z1 at30m -> nearest10m -> cubic6m -> same physical clipping as baseline',
    'RGB_generation_author_equivalent':False,'terrain_DEM_version_author_equivalent':False,
    'location_and_RGB_selection_unchanged':True,'full_author_preprocessing_reproduced':False})
(OUT/'frozen_input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'records':len(records),'only_topo_changed':all(r['unchanged_except_topo'] for r in records)},indent=2))
