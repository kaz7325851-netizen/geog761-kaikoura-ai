"""Fetch a pinned official baseline and 20 deterministic public test pairs."""
import concurrent.futures
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_SHA = '2df4c378cfc4a0feedf197826e769da520100d81'
DATA_SHA = '4b291891badf301b5c75c2153f0f8fe00eeb1435'
IDS = sorted(random.Random(20261004).sample(range(1, 801), 20))

def fetch(item):
    url, relative = item
    destination = ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'GEOG761-model-evaluation'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                if not payload:
                    raise ValueError('Empty response')
                destination.write_bytes(payload)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    payload = destination.read_bytes()
    print(f'{relative}: {len(payload)} bytes', flush=True)
    return {'path': relative, 'url': url, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}

def main():
    code_files = ['README.md', 'LICENSE', 'Predict.py', 'Train.py',
                  'model/Networks.py', 'dataset/landslide_dataset.py']
    jobs = [(f'https://raw.githubusercontent.com/iarai/Landslide4Sense-2022/{CODE_SHA}/{f}', f'official/{f}') for f in code_files]
    data_root = f'https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense/resolve/{DATA_SHA}'
    jobs.append((f'{data_root}/README.md', 'data/README.md'))
    for i in IDS:
        for category, stem in [('images', 'image'), ('annotations', 'mask')]:
            relative = f'{category}/test/{stem}_{i}.h5'
            jobs.append((f'{data_root}/{relative}', f'data/{relative}'))
    records, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:
                errors.append({'job': futures[future], 'error': str(exc)})
                print(f'FAILED: {errors[-1]}', flush=True)
    manifest = {'code_revision': CODE_SHA, 'data_revision': DATA_SHA, 'split': 'test',
                'seed': 20261004, 'sample_ids': IDS, 'files': records, 'errors': errors,
                'note': 'Small runnability pilot, not representative Kaikoura validation. No labels were used to select sample IDs.'}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
