"""Actual checkpoint inference on frozen author inputs; not an accuracy test."""
from pathlib import Path
import os,json,hashlib,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2';os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'author_smoke'
P=ROOT/'weights/M_ALL_006.hdf5'
assert hashlib.sha256(P.read_bytes()).hexdigest()==json.loads((ROOT/'checkpoint_static_audit.json').read_text())['checkpoint_sha256']
import tensorflow as tf,tf_keras
tf.get_logger().setLevel('ERROR')
tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
model=tf_keras.models.load_model(P,compile=False)
manifest=json.loads((OUT/'sample_manifest.json').read_text())
records=[]
for l in manifest['records']:
    src=Path(l['input_file']);assert hashlib.sha256(src.read_bytes()).hexdigest()==l['input_sha256']
    z=np.load(src);inputs=[z[k][None] for k in ['pre','post','topo']]
    assert all(np.isfinite(a).all() for a in inputs)
    t=time.monotonic();result=model(inputs,training=False);p=np.array(result[0])[0,:,:,0];elapsed=time.monotonic()-t
    assert p.shape==(224,224) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    np.save(OUT/(l['tile_id']+'_probability.npy'),p)
    central=np.s_[56:168,56:168];v=z['valid'][central];actual=p[central][v];expected=z['expected_confidence'][central][v]/100
    delta=np.abs(actual-expected)
    record={**l,'seconds':elapsed,'probability_min':float(p.min()),'probability_max':float(p.max()),
        'central_positive_fraction_at_0_5':float(np.mean(actual>=.5)),
        'author_confidence_mae':float(delta.mean()),'author_confidence_max_abs_diff':float(delta.max()),
        'author_binary_agreement_at_0_5':float(np.mean((actual>=.5)==(expected>=.5))),
        'within_author_uint8_quantization_plus_1e_4_fraction':float(np.mean(delta<=.0101))}
    records.append(record);print(json.dumps(record),flush=True)
summary={'loaded_real_task_weights':True,'inference_performed':True,'training':False,
    'tensorflow':tf.__version__,'tf_keras':tf_keras.__version__,'device':'CPU','records':records,
    'implementation_agrees_with_author_quantized_output':all(r['within_author_uint8_quantization_plus_1e_4_fraction']>=.99 for r in records),
    'is_independent_accuracy_test':False,'reference_is_prediction_not_ground_truth':True}
(OUT/'smoke_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
