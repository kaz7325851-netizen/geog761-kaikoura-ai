"""Strict legacy-Keras topology loading of reviewed author model and tensor verification."""
from pathlib import Path
import os
os.environ['TF_USE_LEGACY_KERAS']='1'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import json
import numpy as np
import h5py
import tensorflow as tf
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'official/SAR-LRA Tool V1'))
from utils import CNN_CBAM,focal_loss

def load_model(pass_):
    filters=32 if pass_=='ASCENDING' else 64
    path=next((ROOT/'official/SAR-LRA Tool V1/model_weights').glob('*_'+pass_+'_*.hdf5'))
    tf.keras.backend.clear_session()
    model=CNN_CBAM(lr=.001,loss=focal_loss,filtersFirstLayer=filters,drop=.7,input_size=(64,64,4))
    model.load_weights(str(path),by_name=False,skip_mismatch=False)
    with h5py.File(path,'r') as h:
        saved=[]
        for name in h.attrs['layer_names']:
            group=h[name]
            for weightname in group.attrs['weight_names']:
                saved.append(np.asarray(group[weightname]))
    live=model.get_weights()
    assert len(saved)==len(live)
    assert all(np.array_equal(a,b) for a,b in zip(saved,live))
    return model,{'pass':pass_,'checkpoint':path.name,'filters':filters,'parameters':model.count_params(),
                  'tensor_count':len(saved),'every_weight_tensor_exactly_matches_hdf5':True,
                  'loading':'by_name=False,skip_mismatch=False; no architecture edits',
                  'training_performed':False}

if __name__=='__main__':
    checks=[]
    for pass_ in ['ASCENDING','DESCENDING']:
        model,details=load_model(pass_)
        # Only checks runtime output shape; NOT a real-data performance test.
        result=model(np.zeros((1,64,64,4),dtype=np.float32),training=False).numpy()
        assert result.shape==(1,1) and np.isfinite(result).all()
        details['synthetic_shape_check_only']=True
        checks.append(details)
        print(json.dumps(details),flush=True)
    (ROOT/'weight_load_checks.json').write_text(json.dumps({'tensorflow':tf.__version__,'keras':'tf_keras 2.16 legacy',
       'devices':[str(x) for x in tf.config.list_physical_devices()],'checks':checks},indent=2),encoding='utf-8')
