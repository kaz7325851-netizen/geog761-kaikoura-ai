"""Validate public samples, or evaluate a provenance-checked official checkpoint.

This runner never trains a model and refuses inference without a checkpoint.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl-cache'))
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_pair(i):
    with h5py.File(ROOT / f'data/images/test/image_{i}.h5', 'r') as handle:
        image = handle['img'][:].astype(np.float32)
    with h5py.File(ROOT / f'data/annotations/test/mask_{i}.h5', 'r') as handle:
        label = handle['mask'][:]
    if image.shape != (128, 128, 14) or label.shape != (128, 128):
        raise ValueError(f'Unexpected dimensions for sample {i}')
    if not np.isfinite(image).all() or not np.isin(label, [0, 1]).all():
        raise ValueError(f'Invalid values for sample {i}')
    return image, label.astype(np.uint8)

def display_rgb(image):
    # B4/B3/B2 indices; per-channel percentile stretch is for display ONLY.
    rgb = image[:, :, [3, 2, 1]].copy()
    for c in range(3):
        low, high = np.percentile(rgb[:, :, c], [2, 98])
        rgb[:, :, c] = np.clip((rgb[:, :, c] - low) / max(high - low, 1e-6), 0, 1)
    return rgb

def scores(tp, fp, fn, tn):
    ratio = lambda a, b: float(a / b) if b else None
    return {'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
            'f1': ratio(2 * tp, 2 * tp + fp + fn), 'iou': ratio(tp, tp + fp + fn),
            'accuracy': ratio(tp + tn, tp + tn + fp + fn)}

def counts(label, pred):
    return [int(np.sum((label == a) & (pred == b)))
            for a, b in [(1, 1), (0, 1), (1, 0), (0, 0)]]

def check_data(ids):
    destination = ROOT / 'checks'
    destination.mkdir(exist_ok=True)
    rows = []
    for i in ids:
        image, label = read_pair(i)
        rows.append({'sample_id': i, 'image_shape': list(image.shape),
                     'landslide_pixels': int(label.sum()), 'landslide_fraction': float(label.mean())})
    report = {'sample_count': len(rows), 'total_pixels': len(rows) * 128 * 128,
              'landslide_pixels': sum(r['landslide_pixels'] for r in rows),
              'positive_patches': sum(r['landslide_pixels'] > 0 for r in rows),
              'all_shapes_and_values_valid': True, 'samples': rows,
              'note': 'Data validation only. No trained-model predictions or accuracy results.'}
    (destination / 'data_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    for ax, i in zip(axes, ids[:4]):
        image, label = read_pair(i)
        ax[0].imshow(display_rgb(image))
        ax[0].set_title(f'Test {i}: RGB display stretch')
        ax[1].imshow(label, vmin=0, vmax=1, cmap='gray')
        ax[1].set_title(f'Reference label: {int(label.sum())} landslide pixels')
        for a in ax:
            a.axis('off')
    fig.suptitle('Public benchmark samples / labels (NOT model predictions)')
    fig.tight_layout()
    fig.savefig(destination / 'samples_and_labels.png', dpi=120)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))

def runtime_check():
    import torch
    sys.path.insert(0, str(ROOT / 'official'))
    from model.Networks import unet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet(n_classes=2).to(device).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 14, 128, 128, device=device))
    assert tuple(output.shape) == (1, 2, 128, 128)
    report = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
              'cuda_available': torch.cuda.is_available(), 'device': str(device),
              'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'output_shape': list(output.shape),
              'note': 'Architecture smoke check with untrained weights only. NOT an inference/accuracy test.'}
    (ROOT / 'checks').mkdir(exist_ok=True)
    (ROOT / 'checks/runtime_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

def evaluate(ids, checkpoint, provenance):
    if not checkpoint or not checkpoint.is_file():
        raise SystemExit('No pretrained checkpoint. Cannot perform prediction or report accuracy.')
    if not provenance:
        raise SystemExit('Supply --checkpoint-source with verified source and training-split information.')
    import torch
    sys.path.insert(0, str(ROOT / 'official'))
    from model.Networks import unet
    from dataset.landslide_dataset import LandslideDataSet
    destination = ROOT / 'results' / time.strftime('%Y%m%d_%H%M%S')
    destination.mkdir(parents=True)
    list_file = destination / 'sample_list.txt'
    list_file.write_text('\n'.join(f'images/test/image_{i}.h5' for i in ids), encoding='utf-8')
    dataset = LandslideDataSet(str(ROOT / 'data') + '/', str(list_file), set='unlabeled')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet(n_classes=2)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    torch.backends.cudnn.benchmark = False
    synchronize = lambda: torch.cuda.synchronize() if device.type == 'cuda' else None
    x0 = torch.from_numpy(dataset[0][0]).unsqueeze(0).to(device)
    with torch.inference_mode():
        model(x0)  # Warm up. Excluded from timings.
    synchronize()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    rows, total = [], np.zeros(4, dtype=np.int64)
    for j, i in enumerate(ids):
        image, label = read_pair(i)
        tensor = torch.from_numpy(dataset[j][0]).unsqueeze(0).to(device)
        synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            pred = model(tensor).softmax(dim=1).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        synchronize()
        elapsed = time.perf_counter() - start
        c = counts(label, pred)
        total += c
        rows.append({'sample_id': i, 'seconds': elapsed, 'tp': c[0], 'fp': c[1], 'fn': c[2], 'tn': c[3], **scores(*c)})
        with h5py.File(destination / f'prediction_{i}.h5', 'w') as handle:
            handle.create_dataset('mask', data=pred)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        error = np.zeros((128, 128, 3), dtype=np.float32)
        error[(label == 1) & (pred == 1)] = [0, 1, 0]
        error[(label == 0) & (pred == 1)] = [1, 0, 0]
        error[(label == 1) & (pred == 0)] = [0, 0.5, 1]
        for ax, picture, title in zip(axes, [display_rgb(image), label, pred, error],
                ['RGB (display only)', 'Reference label', 'Model prediction', 'Green=TP / red=FP / blue=FN']):
            ax.imshow(picture, **({'vmin': 0, 'vmax': 1, 'cmap': 'gray'} if picture.ndim == 2 else {}))
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        fig.suptitle(f'Public test sample {i}')
        fig.tight_layout()
        fig.savefig(destination / f'comparison_{i}.png', dpi=120)
        plt.close(fig)
    with (destination / 'per_sample.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'sample_ids': ids, 'split': 'test', 'checkpoint_source': provenance,
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'device': str(device), 'torch': torch.__version__, 'confusion_tp_fp_fn_tn': total.tolist(),
              'aggregate_pixel_metrics': scores(*total), 'mean_warm_inference_seconds': float(np.mean([r['seconds'] for r in rows])),
              'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
              'limitations': ['20-patch pilot, not representative performance.',
                  'Test independence is conditional on verified checkpoint training provenance.',
                  'Detects landslides, not road passability, beach suitability or routes.',
                  'Does not validate Kaikoura geographic transfer.']}
    (destination / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Results: {destination}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--check-runtime', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-source')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['errors']:
        raise SystemExit('Incomplete download. Re-run prepare.py.')
    # Metric unit checks, including zero-denominator behavior.
    assert counts(np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0])) == [1, 1, 1, 1]
    assert scores(1, 1, 1, 1)['f1'] == 0.5
    assert scores(0, 0, 0, 10)['f1'] is None
    if args.check_data:
        check_data(manifest['sample_ids'])
    if args.check_runtime:
        runtime_check()
    if not args.check_data and not args.check_runtime:
        evaluate(manifest['sample_ids'], args.checkpoint, args.checkpoint_source)

if __name__ == '__main__':
    main()
