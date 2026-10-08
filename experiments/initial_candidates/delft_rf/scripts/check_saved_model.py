"""Load known sklearn/numpy types only; run actual saved RF on synthetic API inputs.

This checks classifier loading/inference, NOT imagery accuracy or end-to-end detection.
"""
from pathlib import Path
import importlib
import json
import hashlib
import time
import numpy as np
import sklearn
import joblib
from joblib.numpy_pickle import NumpyUnpickler

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'official/model/model_validation_2019_14_9'
ALLOWED={
    ('sklearn.ensemble.forest','RandomForestClassifier'),
    ('sklearn.tree.tree','DecisionTreeClassifier'),
    ('sklearn.tree._tree','Tree'),
    ('numpy','dtype'),('numpy','ndarray'),
    ('numpy.core.multiarray','_reconstruct'),('numpy.core.multiarray','scalar'),
    ('joblib.numpy_pickle','NumpyArrayWrapper'),
    ('sklearn.externals.joblib.numpy_pickle','NumpyArrayWrapper'),
}

class RestrictedUnpickler(NumpyUnpickler):
    def find_class(self,module,name):
        print('Serialized type:',module,name,flush=True)
        if (module,name) not in ALLOWED:
            raise ValueError('Unreviewed serialized global: '+module+'.'+name)
        return getattr(importlib.import_module(module),name)

def load_reviewed():
    with MODEL.open('rb') as handle:
        return RestrictedUnpickler(str(MODEL),handle,mmap_mode=None).load()

def main():
    try:
        model=load_reviewed()
        details={'status':'loaded','sklearn':sklearn.__version__,'numpy':np.__version__,'joblib':joblib.__version__,
                 'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                 'n_estimators':len(model.estimators_),'n_features':int(model.n_features_),
                 'classes':model.classes_.tolist(),'max_depth':model.max_depth,
                 'feature_importances':model.feature_importances_.tolist()}
        rng=np.random.RandomState(761)
        X=rng.uniform(-1,1,(4,model.n_features_)).astype(np.float32)
        start=time.perf_counter()
        pred=model.predict(X)
        proba=model.predict_proba(X)
        details['api_test_seconds']=time.perf_counter()-start
        details['synthetic_test_predictions']=pred.tolist()
        details['synthetic_test_probabilities']=proba.tolist()
        assert pred.shape==(4,) and proba.shape==(4,2) and np.allclose(proba.sum(1),1)
        details['warning']='Synthetic feature vectors verify the saved classifier executes; NOT real-image detection or accuracy evaluation.'
        (ROOT/'saved_model_check.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        print(json.dumps(details,indent=2))
    except Exception as exc:
        (ROOT/'saved_model_check.json').write_text(json.dumps({'status':'failed','type':type(exc).__name__,'error':str(exc)},indent=2),encoding='utf-8')
        raise

if __name__=='__main__':
    main()
