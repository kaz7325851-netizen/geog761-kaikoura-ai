"""Static HDF5 allowlist inspection before strict legacy Keras loading."""
from pathlib import Path
import json,hashlib,os,time,collections
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import h5py,numpy as np
ROOT=Path(__file__).resolve().parent
P=ROOT/'weights/M_ALL_006.hdf5'
sha=hashlib.sha256(P.read_bytes()).hexdigest()
assert sha=='24b81b52411917bb8b69dc87b994478a4ddbdce92c4f5c42e6b4061a763b6a3b'
with h5py.File(P,'r') as f:
    config=json.loads(f.attrs['model_config'])
    original_keras=str(f.attrs['keras_version'])
    dataset_count=0
    def visit(name,obj):
        global dataset_count
        if isinstance(obj,h5py.Dataset):
            assert obj.dtype.kind in 'fiub','Unexpected nonnumeric dataset'
            dataset_count+=1
    f['model_weights'].visititems(visit)
layers=config['config']['layers']
allowed={'InputLayer','ZeroPadding2D','Conv2D','BatchNormalization','Activation','Concatenate','MaxPooling2D','Add','Dropout','UpSampling2D','SpatialDropout2D','TensorFlowOpLayer'}
assert all(l['class_name'] in allowed for l in layers)
ops=[l for l in layers if l['class_name']=='TensorFlowOpLayer']
for l in ops:
    assert l['config']['node_def']['op']=='ResizeBilinear'
    assert l['config']['constants']=={'1':[224,224]}
def inspect(x):
    if isinstance(x,dict):
        assert not any(k in x for k in ['function','function_type','module','registered_name'])
        if 'class_name' in x:
            assert x['class_name'] in allowed|{'Model','L1L2','VarianceScaling','GlorotUniform','Zeros','Ones'}
        for v in x.values():inspect(v)
    elif isinstance(x,list):
        for v in x:inspect(v)
inspect(config)
(ROOT/'sources/checkpoint_model_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
audit={'checkpoint_sha256':sha,'checkpoint_bytes':P.stat().st_size,'original_keras_version':original_keras,
    'layer_types':dict(collections.Counter(l['class_name'] for l in layers)),
    'numeric_weight_datasets':dataset_count,'unsafe_lambda_present':False,'raw_ops_allowlisted':['ResizeBilinear'],
    'input_layers':config['config']['input_layers'],'output_layers':config['config']['output_layers']}
(ROOT/'checkpoint_static_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('Static HDF5 check passed; no serialized Lambda/custom function.',flush=True)
import tensorflow as tf
import tf_keras
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
t=time.monotonic()
model=tf_keras.models.load_model(P,compile=False)
info={**audit,'tensorflow_version':tf.__version__,'tf_keras_version':tf_keras.__version__,
    'loaded':True,'load_seconds':time.monotonic()-t,'parameters':model.count_params(),
    'input_shapes':[x.shape.as_list() for x in model.inputs],
    'output_shapes':[x.shape.as_list() for x in model.outputs],'output_names':model.output_names,
    'training':False,'model_inference_performed':False,'device':'CPU'}
(ROOT/'checkpoint_load_check.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
print(json.dumps(info,indent=2),flush=True)
