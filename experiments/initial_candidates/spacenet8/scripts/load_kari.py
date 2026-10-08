"""Limited weights-only loading of author serialized model, followed by strict state transfer."""
from pathlib import Path
import sys
import hashlib
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime_deps'))
sys.path.insert(0,str(ROOT/'official/01-ohhan777/code'))
from models import seg_hrnet_ocr as architecture
from models.config import update_config
from yacs.config import CfgNode

CUSTOM=['HighResolutionNet','HighResolutionModule','BasicBlock','Bottleneck',
        'SpatialGather_Module','SpatialOCR_Module','ObjectAttentionBlock2D']
STANDARD=[torch.nn.Conv2d,torch.nn.Dropout2d,torch.nn.SyncBatchNorm,torch.nn.ReLU,
          torch.nn.Sequential,torch.nn.ModuleList,torch.nn.MaxPool2d]
ALLOWED={cls.__module__+'.'+cls.__name__: cls for cls in STANDARD}
ALLOWED.update({'models.seg_hrnet_ocr.'+name:getattr(architecture,name) for name in CUSTOM})
ALLOWED.update({'yacs.config.CfgNode':CfgNode,'numpy.dtype':np.dtype,
                'numpy.core.multiarray.scalar':np._core.multiarray.scalar})
def load_model():
    path=ROOT/'weights/best_flood.pt'
    expected='c7f5ac3fe6da2e3d7536fc432aa8cb7efa6eb9f6f1e183eea091f901afc8df29'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Checkpoint differs from recorded download')
    globals_=torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
    if set(globals_) - set(ALLOWED):
        raise ValueError('Unreviewed checkpoint types: '+str(set(globals_)-set(ALLOWED)))
    safe=[(ALLOWED[name],name) for name in globals_]
    safe += [np.dtypes.Float64DType,np.dtypes.Float32DType]
    with torch.serialization.safe_globals(safe):
        checkpoint=torch.load(str(path),map_location='cpu',weights_only=True)
    saved=checkpoint['model']
    if type(saved) is not architecture.HighResolutionNet:
        raise TypeError(type(saved))
    config=update_config(str(ROOT/'official/01-ohhan777/code/models/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'))
    model=architecture.HighResolutionNet(config)
    keys=model.load_state_dict(saved.state_dict(),strict=True)
    details={'weights_only':True,'explicit_reviewed_safe_globals':globals_,
             'strict_state_transfer':True,'missing_keys':keys.missing_keys,'unexpected_keys':keys.unexpected_keys,
             'checkpoint_epoch':int(checkpoint.get('epoch',-1)),'checkpoint_date':checkpoint.get('date'),
             'parameter_count':sum(p.numel() for p in model.parameters()),
             'weights_source':json.loads((ROOT/'weight_source.json').read_text())}
    return model.eval(),details

if __name__=='__main__':
    model,details=load_model()
    (ROOT/'kari_load_check.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(json.dumps(details,indent=2))
