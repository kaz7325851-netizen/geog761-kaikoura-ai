"""Controlled terrain comparison; no training or model threshold tuning."""
from pathlib import Path
import os,json,hashlib,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2';os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import numpy as np,rasterio
from rasterio.transform import Affine
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'preprocessing_audit/terrain_aligned_pilot'
audit=json.loads((ROOT/'checkpoint_static_audit.json').read_text())
assert hashlib.sha256((ROOT/'weights/M_ALL_006.hdf5').read_bytes()).hexdigest()==audit['checkpoint_sha256']
import tensorflow as tf,tf_keras
tf.get_logger().setLevel('ERROR');tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
model=tf_keras.models.load_model(ROOT/'weights/M_ALL_006.hdf5',compile=False)
manifest=json.loads((OUT/'frozen_input_manifest.json').read_text());records=[];outputs={};repeat=None
for r in manifest['records']:
    pth=Path(r['file']);assert hashlib.sha256(pth.read_bytes()).hexdigest()==r['sha256']
    z=np.load(pth);inputs=[z[k][None] for k in ['pre','post','topo']]
    assert all(np.isfinite(x).all() for x in inputs)
    start=time.monotonic();p=np.array(model(inputs,training=False)[0])[0,:,:,0];secs=time.monotonic()-start
    assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    if repeat is None:
        again=np.array(model(inputs,training=False)[0])[0,:,:,0]
        repeat={'sample':r['tile_id'],'max_absolute_probability_difference':float(np.max(np.abs(p-again))),
            'exactly_equal':bool(np.array_equal(p,again))}
    valid=z['valid'].copy();central=np.zeros((224,224),bool);central[56:168,56:168]=True;valid&=central
    positive=(p>=.5)&valid;target=z['target_inventory']&valid;ref=z['all_inventory']&valid
    road=z['road100m']&valid;baseline=np.load(r['baseline_probability_file']);oldpositive=baseline[valid]>=.5
    save=OUT/(r['tile_id']+'_probability.tif')
    with rasterio.open(save,'w',driver='GTiff',width=224,height=224,count=1,dtype='float32',crs=r['crs'],transform=Affine(*r['grid']),nodata=-9999,compress='deflate') as ds:
        ds.write(np.where(valid,p,-9999).astype('float32'),1);ds.set_band_description(1,'exploratory_landslide_probability')
        ds.update_tags(meaning='NOT road passability; unknown author RGB conversion; controlled GDAL Horn derivative comparison')
    np.save(OUT/(r['tile_id']+'_raw_context_probability.npy'),p);outputs[r['tile_id']]=p
    records.append({**r,'seconds':secs,'probability_file':str(save),'probability_sha256':hashlib.sha256(save.read_bytes()).hexdigest(),
        'scored_pixels':int(valid.sum()),'target_reference_pixels':int(target.sum()),
        'target_covered_pixels':int((target&positive).sum()),
        'target_inventory_overlap_fraction_not_validated_recall':float(np.mean(p[target]>=.5)) if target.any() else None,
        'predicted_positive_valid_pixels':int(positive.sum()),'predicted_positive_road100m_pixels':int((positive&road).sum()),
        'positive_outside_inventory_not_confirmed_FP':int((positive&~ref).sum()),
        'change_vs_old_terrain_binary_fraction':float(np.mean((p[valid]>=.5)!=oldpositive)),
        'change_vs_old_terrain_probability_MAE':float(np.mean(np.abs(p[valid]-baseline[valid]))),
        'prediction_unchanged_under_repeat':repeat['exactly_equal'] if len(records)==0 else None})
    print(json.dumps({'tile':r['tile_id'],'target_covered':records[-1]['target_inventory_overlap_fraction_not_validated_recall'],
        'terrain_binary_change':records[-1]['change_vs_old_terrain_binary_fraction']}),flush=True)
sensitivity=[]
for sample in sorted(set(r['sample_id'] for r in records)):
    group=[r for r in records if r['sample_id']==sample];a,b=[outputs[r['tile_id']] for r in group]
    valid=np.load(group[0]['file'])['valid'].copy();central=np.zeros((224,224),bool);central[56:168,56:168]=True;valid&=central
    aa=a[valid]>=.5;bb=b[valid]>=.5;union=np.sum(aa|bb)
    sensitivity.append({'sample_id':sample,'prediction_binary_disagreement_fraction':float(np.mean(aa!=bb)),
        'positive_mask_between_renderings_IoU':float(np.sum(aa&bb)/union) if union else None,
        'mean_absolute_probability_difference':float(np.mean(np.abs(a[valid]-b[valid])))})
summary={'records':records,'rendering_sensitivity':sensitivity,'repeatability_same_input_check':repeat,
    'checkpoint_sha256':audit['checkpoint_sha256'],'threshold':.5,'training':False,'fine_tuning':False,
    'is_independent_accuracy_test':False,'road_passability_inferred':False,
    'full_author_preprocessing_reproduced':False,'author_sample_terrain_derivatives_reproduced':True}
(OUT/'controlled_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'sensitivity':sensitivity,'repeat':repeat},indent=2),flush=True)
