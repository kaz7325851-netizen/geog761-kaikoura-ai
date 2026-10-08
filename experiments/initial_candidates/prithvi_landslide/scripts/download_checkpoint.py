"""Download the selected community checkpoint and verify HF LFS SHA256."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
SHA = '0dba9a736c5d6c03f8e70720f94bc25d1d1f4418'
EXPECTED_HASH = '22b3238867c04d62615a1ce18f46933d755f5abc7c8ca0eeff4ad1fd2127ccbe'
EXPECTED_SIZE = 3780602291
URL = f'https://huggingface.co/srivassid/prithvi-landslide/resolve/{SHA}/last.ckpt'

def main():
    destination = ROOT / 'last.ckpt'
    temporary = ROOT / 'last.ckpt.partial'
    if destination.exists():
        print('Checkpoint already present; verifying', flush=True)
        digest = hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest()
    else:
        start = time.monotonic()
        digest = hashlib.sha256()
        received = 0
        next_progress = 256 * 2**20
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as output:
            while chunk := response.read(2**20 * 4):
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if received >= next_progress:
                    print(f'{received / 2**30:.2f} / {EXPECTED_SIZE / 2**30:.2f} GiB, {time.monotonic()-start:.0f}s', flush=True)
                    next_progress += 256 * 2**20
        digest = digest.hexdigest()
        if received != EXPECTED_SIZE or digest != EXPECTED_HASH:
            raise RuntimeError('Checkpoint size/hash mismatch; do not load partial file.')
        temporary.rename(destination)
    if destination.stat().st_size != EXPECTED_SIZE or digest != EXPECTED_HASH:
        raise RuntimeError('Existing checkpoint failed integrity check.')
    report = {'repo': 'srivassid/prithvi-landslide', 'revision': SHA, 'url': URL,
              'size': destination.stat().st_size, 'sha256': digest, 'verified': True,
              'source_type': 'Community personal fine-tune, NOT official IBM/NASA downstream weights',
              'training': 'Model card says Landslide4Sense; config uses standard train/validation/test datamodule, training exposure not independently audited.'}
    (ROOT / 'checkpoint_source.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Checkpoint downloaded and hash verified', flush=True)

if __name__ == '__main__':
    main()
