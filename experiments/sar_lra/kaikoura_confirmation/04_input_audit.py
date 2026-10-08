"""Audit existing arrays and frozen settings before new model evaluation."""
from pathlib import Path
import hashlib,json
import numpy as np
import rasterio
ROOT=Path(__file__).resolve().parent
SAR=ROOT.parent
old=SAR/'kaikoura_exploration'
sources={'sumatra':SAR/'data/ASCENDING.tif','kaikoura_old':Path(json.loads((old/'latest_input.json').read_text(encoding='utf-8'))['input_file'])}
# Resolve the actual recorded Sumatra path without searching large environments.
if not sources['sumatra'].exists():
    sources['sumatra']=next((SAR/'data').rglob('ASCENDING.tif'))
checks={}
for name,path in sources.items():
    with rasterio.open(path) as src:
        data=src.read();valid=data[4]==1
        assert np.isfinite(data[:4,valid]).all() and (data[:4,valid]!=-9999).all()
        entry={'file':str(path),'shape':list(src.shape),'crs':src.crs.to_string(),'transform':list(src.transform)[:6],
               'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'valid_pixels':int(valid.sum()),
               'channel_quantiles_p1_p25_p50_p75_p99':np.quantile(data[:4,valid],[.01,.25,.5,.75,.99],axis=1).T.tolist(),
               'expected_model_order':['postVV','postVH','diffVV','diffVH'],'input_normalization':'None, author deployment dB space',
               'reference_not_used_as_model_input':True}
        if name=='kaikoura_old':
            assert np.allclose(data[2,valid],data[0,valid]-data[5,valid],atol=2e-5)
            assert np.allclose(data[3,valid],data[1,valid]-data[6,valid],atol=2e-5)
            entry['difference_formula_verified']=True
    checks[name]=entry
checkpoint=next((SAR/'official/SAR-LRA Tool V1/model_weights').glob('*_ASCENDING_*.hdf5'))
design=json.loads((ROOT/'frozen_design.json').read_text(encoding='utf-8'))
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==design['checkpoint_sha256']
checks['checkpoint_sha256']=design['checkpoint_sha256']
checks['frozen_design_sha256']=hashlib.sha256((ROOT/'frozen_design.json').read_bytes()).hexdigest()
checks['status']='PASS for tested local consistency, not proof of full author training preprocessing reproduction'
checks['uncontrolled_differences']=['Region/terrain/landcover','Pre82day single date versus pre60day median','Post44h versus about10days','Geographic versus metric sampling grid; explicit control planned']
(ROOT/'input_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
