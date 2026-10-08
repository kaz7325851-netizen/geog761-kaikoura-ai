"""Six small inference calls on 3 frozen locations x 2 RGB renderings."""
from pathlib import Path
import os,json,hashlib,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2';os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import numpy as np,rasterio
from rasterio.transform import Affine
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'pilot_exploration'
assert hashlib.sha256((ROOT/'weights/M_ALL_006.hdf5').read_bytes()).hexdigest()==json.loads((ROOT/'checkpoint_static_audit.json').read_text())['checkpoint_sha256']
import tensorflow as tf,tf_keras
tf.get_logger().setLevel('ERROR');tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
model=tf_keras.models.load_model(ROOT/'weights/M_ALL_006.hdf5',compile=False)
manifest=json.loads((OUT/'frozen_input_manifest.json').read_text());records=[];outputs={}
for l in manifest['records']:
    path=Path(l['file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==l['sha256']
    z=np.load(path);inputs=[z[k][None] for k in ['pre','post','topo']]
    assert all(a.shape[1:3]==(224,224) and np.isfinite(a).all() for a in inputs)
    t=time.monotonic();pred=model(inputs,training=False);p=np.array(pred[0])[0,:,:,0];secs=time.monotonic()-t
    assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    valid=z['valid'].copy();central=np.zeros((224,224),bool);central[56:168,56:168]=True;valid&=central
    positive=(p>=.5)&valid;target=z['target_inventory']&valid;allref=z['all_inventory']&valid
    rm=z['road100m']&valid;landvalid=valid # Do not silently mask non-inventory areas to improve scores.
    save=OUT/(l['tile_id']+'_probability.tif')
    with rasterio.open(save,'w',driver='GTiff',width=224,height=224,count=1,dtype='float32',crs=l['crs'],transform=Affine(*l['grid']),nodata=-9999,compress='deflate') as s:
        s.write(np.where(valid,p,-9999).astype('float32'),1);s.set_band_description(1,'exploratory_landslide_probability');s.update_tags(meaning='NOT road passability; exploratory RGB and terrain preprocessing',threshold=.5)
    outputs[l['tile_id']]=p
    np.save(OUT/(l['tile_id']+'_raw_context_probability.npy'),p)
    record={**l,'seconds':secs,'probability_file':str(save),'probability_sha256':hashlib.sha256(save.read_bytes()).hexdigest(),
       'probability_range':[float(p.min()),float(p.max())], 'scored_pixels':int(valid.sum()),
       'scored_target_reference_pixels':int(target.sum()),'scored_all_reference_pixels':int(allref.sum()),
       'predicted_positive_valid_pixels':int(positive.sum()),
       'predicted_positive_road100m_pixels':int((positive&rm).sum()),
       'target_inventory_predicted_positive_pixels':int((positive&target).sum()),
       'target_inventory_overlap_fraction_not_validated_recall':float((positive&target).sum()/target.sum()) if target.any() else None,
       'all_inventory_overlap_fraction_not_validated_recall':float((positive&allref).sum()/allref.sum()) if allref.any() else None,
       'predicted_positive_outside_inventory_pixels_not_confirmed_FP':int((positive&~allref).sum())}
    records.append(record);print(json.dumps(record),flush=True)
sensitivity=[]
for sample in sorted(set(l['sample_id'] for l in records)):
    group=[r for r in records if r['sample_id']==sample];a,b=[outputs[r['tile_id']] for r in group]
    valid=np.load(group[0]['file'])['valid'].copy();central=np.zeros((224,224),bool);central[56:168,56:168]=True;valid&=central
    aa=a[valid]>=.5;bb=b[valid]>=.5;union=np.sum(aa|bb)
    sensitivity.append({'sample_id':sample,'prediction_binary_disagreement_fraction':float(np.mean(aa!=bb)),
       'positive_mask_between_renderings_IoU':float(np.sum(aa&bb)/union) if union else None,
       'max_probability_difference':float(np.max(np.abs(a[valid]-b[valid]))),'probability_difference_mean':float(np.mean(np.abs(a[valid]-b[valid])))})
summary={'model':'M_ALL_006','checkpoint_sha256':json.loads((ROOT/'checkpoint_static_audit.json').read_text())['checkpoint_sha256'],
    'actual_inference_performed':True,'trained_from_scratch':False,'fine_tuned':False,'device':'CPU',
    'locations':3,'inference_variants':6,'threshold':.5,'scoring_region':'central 112x112; full 224x224 input provides context',
    'records':records,'rendering_sensitivity':sensitivity,
    'is_independent_accuracy_test':False,'road_passability_inferred':False,'limitations':manifest['limitations']}
(OUT/'exploratory_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'rendering_sensitivity':sensitivity},indent=2),flush=True)
