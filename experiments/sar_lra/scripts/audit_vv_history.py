"""Inspect author Git filenames across all available history, without fetching blobs."""
from pathlib import Path
import subprocess
import json
import zipfile
import hashlib
import io
import h5py

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'vv_only/source_audit'
REPO = OUT / 'author_history.git'

def git(*args):
    return subprocess.check_output(['git', '--git-dir=' + str(REPO), *args], encoding='utf-8')

names = sorted(set(x for x in git('log', '--all', '--format=', '--name-only').splitlines() if x))
weights = [x for x in names if Path(x).suffix.lower() in ('.hdf5', '.h5', '.pth', '.pt', '.ckpt', '.onnx', '.safetensors', '.keras')]
vv_weights = [x for x in weights if 'vv' in x.lower() and 'vv_vh' not in x.lower()]
archive_checks = []
with zipfile.ZipFile(OUT / 'Weights.zip') as archive:
    for item in archive.infolist():
        if not item.filename.endswith('.hdf5'):
            continue
        payload = archive.read(item)
        local = ROOT / 'official/SAR-LRA Tool V1/model_weights' / item.filename
        with h5py.File(io.BytesIO(payload), 'r') as handle:
            datasets = []
            handle.visititems(lambda name, obj: datasets.append({'name': name, 'shape': list(obj.shape)}) if isinstance(obj, h5py.Dataset) and len(obj.shape) == 4 else None)
        archive_checks.append({'file': item.filename, 'sha256': hashlib.sha256(payload).hexdigest(),
            'identical_to_existing_dual_checkpoint': local.exists() and payload == local.read_bytes(),
            'convolution_kernels': datasets})
report = {'head': git('rev-parse', 'HEAD').strip(), 'commit_count': int(git('rev-list', '--all', '--count')),
    'refs': git('show-ref').splitlines(), 'unique_historical_path_count': len(names),
    'historical_weight_paths': weights, 'vv_without_vh_weight_paths': vv_weights,
    'scope': 'All commits and refs in the author public repository clone, not all forks or private files',
    'archive_checks': archive_checks, 'checkpoint_found': bool(vv_weights), 'inference_performed': False}
(OUT / 'historical_paths.json').write_text(json.dumps(names, indent=2), encoding='utf-8')
(OUT / 'history_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
