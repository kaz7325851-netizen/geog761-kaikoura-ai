"""Strictly load the task checkpoint into unmodified author architecture.

Uses task-local SMP 0.3.3; a synthetic forward is a software test, not an
assessment of landslide detection. No real study data are predicted here.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from argparse import Namespace
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime_smp033'))
import torch
import segmentation_models_pytorch as smp
torch.set_num_threads(2)
report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'smp_version': smp.__version__,
          'real_imagery_inference': False, 'training': False}
path = ROOT / 'author_source_ce34d3ac' / 'igarss-landslide-delineation-ce34d3ac02b80d4db7e542203b7e534649a46352' / 'src' / 'models' / 'bbunet.py'
try:
    # Import only the inspected BBUnet module, avoiding unrelated model imports.
    spec = importlib.util.spec_from_file_location('author_bbunet', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report['author_architecture_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    weight_path = ROOT / 'weights' / 'bbunet_r18.pt'
    with torch.serialization.safe_globals([Namespace]):
        ckpt = torch.load(weight_path, map_location='cpu', weights_only=True)
    hp = vars(ckpt['hparams'])
    assert hp['MODEL'] == 'bbunet' and hp['ENCODER'] == 'resnet18'
    model = mod.BBUnet(s2_in_channels=len(hp['BANDS']), dem_in_channels=4, classes=hp['N_CLASSES'])
    result = model.load_state_dict(ckpt['state_dict'], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    report['strict_task_state_load_pass'] = True
    report['tensor_count'] = len(ckpt['state_dict'])
    report['external_SSL_encoder_initialization_needed'] = False
    report['reason'] = 'Complete task state includes optical encoder and terrain encoder; direct author class construction avoids loading the training-only external initializer'
    model.eval()
    # Every initialized parameter is replaced by strict task-state loading above.
    torch.manual_seed(0)
    pre = torch.rand(1, len(hp['BANDS']), hp['IMAGE_SIZE'], hp['IMAGE_SIZE'])
    post = torch.rand_like(pre)
    terrain = torch.rand(1, 4, hp['IMAGE_SIZE'], hp['IMAGE_SIZE'])
    with torch.inference_mode():
        out = model(pre, post, terrain)
    assert list(out.shape) == [1, 1, 256, 256] and torch.isfinite(out).all()
    report['synthetic_software_forward'] = {'pass': True, 'device': 'cpu',
                                         'output_shape': list(out.shape),
                                         'not_accuracy_or_real_prediction': True}
    report['status'] = 'strict_loading_and_software_smoke_test_passed'
except Exception as e:
    report['status'] = 'failed'
    report['error'] = f'{type(e).__name__}: {e}'
    raise
finally:
    (ROOT / 'weights' / 'model_compatibility.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
