"""Frozen community Prithvi checkpoint pilot on 20 public L4S test patches."""
from pathlib import Path
import argparse
import csv
import gc
import importlib.metadata
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
for key, relative in [('HF_HOME', 'cache/huggingface'), ('TORCH_HOME', 'cache/torch'),
                      ('MPLCONFIGDIR', 'cache/matplotlib')]:
    os.environ[key] = str(BASE / relative)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['ALBUMENTATIONS_NO_CHECK_VERSION'] = '1'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

sys.path.insert(0, str(BASE / 'l4s_unet'))
from evaluate import counts, scores, read_pair, display_rgb

def inspect_checkpoint():
    path = ROOT / 'last.ckpt'
    if not path.is_file():
        raise SystemExit('Checkpoint download not complete.')
    source = json.loads((ROOT / 'checkpoint_source.json').read_text(encoding='utf-8'))
    if not source.get('verified'):
        raise SystemExit('Checkpoint provenance/hash not verified.')
    # Static pickle analysis first; do not automatically allow arbitrary classes.
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    print('Globals not in default weights-only allowlist:', unsafe, flush=True)
    if unsafe:
        (ROOT / 'checkpoint_unsafe_globals.json').write_text(json.dumps(unsafe, indent=2), encoding='utf-8')
        raise SystemExit('Review unsupported globals before attempting a safe load. Unsafe pickle loading is disabled.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = checkpoint['state_dict']
    details = {'top_level_keys': list(checkpoint), 'epoch': checkpoint.get('epoch'),
               'global_step': checkpoint.get('global_step'), 'lightning_version': checkpoint.get('pytorch-lightning_version'),
               'state_key_count': len(state), 'state_keys': list(state),
               'hyper_parameters': repr(checkpoint.get('hyper_parameters'))[:15000],
               'tensor_shapes': {k: list(v.shape) for k, v in state.items() if torch.is_tensor(v)}}
    (ROOT / 'checkpoint_inspection.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print({k: details[k] for k in ['epoch','global_step','lightning_version','state_key_count']}, flush=True)
    return checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    checkpoint = inspect_checkpoint()
    if args.inspect_only:
        return
    from terratorch.models import EncoderDecoderFactory
    from terratorch.datasets import Landslide4SenseNonGeo
    from terratorch.datamodules.landslide4sense import MEANS, STDS
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    config = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    kwargs = config['model']['init_args']['model_args'].copy()
    kwargs['backbone_pretrained'] = False  # Load full fine-tune below; no base download.
    bands = config['data']['init_args']['bands']
    assert bands == ['BLUE', 'GREEN', 'RED', 'NIR_BROAD', 'SWIR_1', 'SWIR_2']
    start = time.perf_counter()
    model = EncoderDecoderFactory().build_model(task='segmentation', **kwargs)
    state = checkpoint['state_dict']
    model_state = {k.removeprefix('model.'): v for k, v in state.items() if k.startswith('model.')}
    if not model_state:
        raise RuntimeError('No model.* weights found; inspect checkpoint instead of guessing.')
    loaded = model.load_state_dict(model_state, strict=True)
    print('Strict state loading passed:', loaded, flush=True)
    del state, model_state, checkpoint
    gc.collect()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    assert device.type == 'cuda', 'This pilot is configured for the verified NVIDIA GPU.'
    torch.set_num_threads(4)
    # Match uploaded config: bilinear image resize 224 and nearest mask resize.
    dataset = Landslide4SenseNonGeo(data_root=str(BASE / 'l4s_unet/data'), split='test', bands=bands,
                                  transform=A.Compose([A.Resize(224, 224), ToTensorV2()]))
    ids = json.loads((BASE / 'l4s_unet/manifest.json').read_text(encoding='utf-8'))['sample_ids'][:args.limit]
    positions = {int(p.stem.split('_')[-1]): j for j, p in enumerate(dataset.image_files)}
    mean = torch.tensor([MEANS[b] for b in bands], device=device).view(1, 6, 1, 1)
    std = torch.tensor([STDS[b] for b in bands], device=device).view(1, 6, 1, 1)
    def input_tensor(i):
        sample = dataset[positions[i]]
        tensor = sample['image'].unsqueeze(0).to(device)
        return (tensor - mean) / std, sample['mask'].numpy()
    def forward(tensor):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            logits = model(tensor).output
        if logits.shape != (1, 2, 224, 224) or not torch.isfinite(logits).all():
            raise ValueError(f'Invalid logits: {tuple(logits.shape)}')
        return logits.float()
    tensor, _ = input_tensor(ids[0])
    forward(tensor)
    torch.cuda.synchronize()
    load_and_warmup_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    destination = ROOT / 'results' / time.strftime('%Y%m%d_%H%M%S')
    destination.mkdir(parents=True)
    rows, total, total224 = [], np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64)
    for i in ids:
        image, label = read_pair(i)
        tensor, label224 = input_tensor(i)
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = forward(tensor)
        native_pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        pred = F.interpolate(logits, size=(128, 128), mode='bilinear', align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        c = counts(label, pred)
        total += c
        total224 += counts(label224, native_pred)
        rows.append({'sample_id': i, 'seconds': elapsed, 'tp': c[0], 'fp': c[1], 'fn': c[2], 'tn': c[3], **scores(*c)})
        with h5py.File(destination / f'prediction_{i}.h5', 'w') as handle:
            handle.create_dataset('mask', data=pred)
        error = np.zeros((128,128,3), np.float32)
        error[(label == 1) & (pred == 1)] = [0,1,0]
        error[(label == 0) & (pred == 1)] = [1,0,0]
        error[(label == 1) & (pred == 0)] = [0,0.5,1]
        fig, axes = plt.subplots(1,4,figsize=(12,3))
        for ax, pic, title in zip(axes,[display_rgb(image),label,pred,error],
                ['RGB (display only)','Reference label','Prithvi prediction','Green=TP / red=FP / blue=FN']):
            ax.imshow(pic, **({'cmap':'gray','vmin':0,'vmax':1} if pic.ndim==2 else {}))
            ax.set_title(title,fontsize=9)
            ax.axis('off')
        fig.suptitle(f'Public test sample {i} / community Prithvi fine-tune')
        fig.tight_layout()
        fig.savefig(destination / f'comparison_{i}.png',dpi=120)
        plt.close(fig)
        print(f'Sample {i}: F1={rows[-1]["f1"]}, {elapsed:.3f}s',flush=True)
    with (destination / 'per_sample.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    versions = {p: importlib.metadata.version(p) for p in ['torch','torchvision','terratorch','timm','albumentations','huggingface_hub']}
    report = {'model':'srivassid/prithvi-landslide', 'checkpoint_source':json.loads((ROOT/'checkpoint_source.json').read_text()),
              'strict_model_state_loading':True,'test_sample_ids':ids,'bands':bands,'input_resize':[224,224],
              'normalization':{'mean':mean.cpu().flatten().tolist(),'std':std.cpu().flatten().tolist()},
              'original_128_grid':{'confusion_tp_fp_fn_tn':total.tolist(),'metrics':scores(*total)},
              'native_224_grid':{'confusion_tp_fp_fn_tn':total224.tolist(),'metrics':scores(*total224)},
              'device':torch.cuda.get_device_name(0),'precision':'fp16 autocast; FP32 output interpolation',
              'versions':versions,'load_and_warmup_seconds':load_and_warmup_seconds,
              'mean_warm_prediction_seconds':float(np.mean([r['seconds'] for r in rows])),
              'peak_cuda_allocated_mb':torch.cuda.max_memory_allocated()/2**20,
              'limitations':['Small public benchmark pilot, not Kaikoura validation.',
                 'Training split exposure not independently audited: do not claim fully independent test accuracy.',
                 'Landslide segmentation is not road passability or beach landing suitability.',
                 'No test-label threshold tuning, training or augmentation was performed.']}
    (destination/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'latest_result.json').write_text(json.dumps({'directory':str(destination)},indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()
