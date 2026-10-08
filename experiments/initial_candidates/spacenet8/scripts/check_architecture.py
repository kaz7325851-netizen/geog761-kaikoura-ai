"""RANDOM-INITIALIZED shape/GPU checks only. No predictions or accuracy reported."""
from pathlib import Path
import sys
import os
import json
import time
import traceback
ROOT=Path(__file__).resolve().parent
os.environ['TORCH_HOME']=str(ROOT.parent/'cache/torch')
sys.path.insert(0,str(ROOT/'official/baseline'))
import torch
from models.pytorch_zoo import abstract_model
from models.pytorch_zoo import unet

# Explicitly prevent ImageNet download: encoder weights alone are NOT task weights.
abstract_model.encoder_params['resnet34']['url']=None
torch.manual_seed(20261004)
device='cuda' if torch.cuda.is_available() else 'cpu'
summary={'status':'architecture_checks_running','repo_revision':'30ca30c2530c6a234d040f60a99a7366ff62d756',
         'torch':torch.__version__,'device':device,
         'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
         'trained_weights_loaded':False,'imagenet_weights_loaded':False,'retrained':False,
         'input':'synthetic tensors ONLY','results':[],
         'warning':'Random initialized architecture checks. No real-image predictions or accuracy evaluation. Timing/memory are feasibility diagnostics only.'}
for name, cls, classes in [('foundation_resnet34',unet.Resnet34_upsample,[1,8]),
                           ('flood_resnet34_siamese',unet.Resnet34_siamese_upsample,5)]:
    result={'model':name,'checks':[]}
    model=None
    try:
        model=cls(num_classes=classes,num_channels=3).eval().to(device)
        result['parameters']=sum(p.numel() for p in model.parameters())
        for size in [256,1300]:
            check={'input_shape':[1,3,size,size]}
            x=None
            outputs=None
            try:
                if device=='cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                x=torch.randn(1,3,size,size,device=device)
                if device=='cuda': torch.cuda.synchronize()
                start=time.perf_counter()
                with torch.inference_mode():
                    outputs=model(x,x) if 'siamese' in name else model(x)
                if device=='cuda': torch.cuda.synchronize()
                check['seconds']=time.perf_counter()-start
                tensors=outputs if isinstance(outputs,tuple) else [outputs]
                check['output_shapes']=[list(t.shape) for t in tensors]
                check['finite']=all(bool(torch.isfinite(t).all()) for t in tensors)
                assert check['finite']
                assert all(list(t.shape[-2:])==[size,size] for t in tensors)
                check['peak_cuda_allocated_mb']=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None
                check['status']='passed'
            except Exception as exc:
                check.update(status='failed',error=repr(exc))
            finally:
                del outputs,x
                if device=='cuda': torch.cuda.empty_cache()
            result['checks'].append(check)
            print(json.dumps({'model':name,**check}),flush=True)
    except Exception as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        del model
        if device=='cuda': torch.cuda.empty_cache()
    summary['results'].append(result)
summary['status']='architecture_checks_complete'
(ROOT/'architecture_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
