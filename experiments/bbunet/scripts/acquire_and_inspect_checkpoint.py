"""Download the author's public task checkpoint; restricted-load inspection only."""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import requests

ROOT = Path(__file__).resolve().parent
FILE_ID = '1VIMgcq_5SyLWjo2jpMElvvWbkY9TbiLq'
SOURCE = 'https://drive.usercontent.google.com/download'
EXPECTED_SIZE = 140937574  # Author public Drive listing; not a published checksum.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inspect-only', action='store_true')
    args = ap.parse_args()
    dest = ROOT / 'weights' / 'bbunet_r18.pt'
    dest.parent.mkdir(exist_ok=True)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_repository': 'https://github.com/links-ads/igarss-landslide-delineation',
              'source_folder': 'https://drive.google.com/drive/folders/1351hEZeY2T67aGhD-ONNyLN8Cq0cAfzX',
              'file_id': FILE_ID, 'file': str(dest), 'model_inference': False,
              'published_checksum_available': False}
    try:
        if not dest.exists() and not args.inspect_only:
            partial = dest.with_suffix('.pt.partial')
            if partial.exists():
                raise RuntimeError('Existing partial file preserved; inspect it before restarting download')
            with requests.get(SOURCE, params={'id': FILE_ID, 'export': 'download', 'confirm': 't'},
                              stream=True, timeout=(20, 60)) as r:
                r.raise_for_status()
                report['content_type'] = r.headers.get('Content-Type')
                report['content_disposition'] = r.headers.get('Content-Disposition')
                if 'text/html' in r.headers.get('Content-Type', ''):
                    raise RuntimeError('Drive returned HTML, not a checkpoint; no HTML saved as weights')
                count = 0
                last = time.monotonic()
                with partial.open('xb') as f:
                    for block in r.iter_content(1024 * 1024):
                        if not block:
                            continue
                        f.write(block)
                        count += len(block)
                        if time.monotonic() - last > 10:
                            print(f'Downloaded {count:,} / expected {EXPECTED_SIZE:,} bytes', flush=True)
                            last = time.monotonic()
                if count != EXPECTED_SIZE:
                    raise RuntimeError(f'Unexpected length {count}; partial retained, no final checkpoint created')
                if not zipfile.is_zipfile(partial):
                    raise RuntimeError('Not a torch ZIP archive; partial retained')
                with zipfile.ZipFile(partial) as z:
                    bad = z.testzip()
                    if bad:
                        raise RuntimeError(f'Archive CRC failed: {bad}')
                partial.rename(dest)
                print(f'Download complete: {count:,} bytes', flush=True)
        if not dest.exists():
            raise FileNotFoundError(dest)
        report['bytes'] = dest.stat().st_size
        digest = hashlib.sha256()
        with dest.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        report['sha256'] = digest.hexdigest()
        with zipfile.ZipFile(dest) as z:
            assert z.testzip() is None
            report['zip_crc_pass'] = True
        import torch
        from argparse import Namespace
        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(dest)
        report['pickle_globals_not_in_default_allowlist'] = unsafe
        if set(unsafe) - {'argparse.Namespace'}:
            raise RuntimeError(f'Unknown serialized globals: {unsafe}; refusing unrestricted pickle load')
        with torch.serialization.safe_globals([Namespace]):
            ckpt = torch.load(dest, map_location='cpu', weights_only=True)
        assert isinstance(ckpt, dict) and 'hparams' in ckpt and 'state_dict' in ckpt
        hp = vars(ckpt['hparams']) if isinstance(ckpt['hparams'], Namespace) else ckpt['hparams']
        assert isinstance(hp, dict)
        state = ckpt['state_dict']
        assert isinstance(state, dict)
        assert all(isinstance(k, str) and isinstance(v, torch.Tensor) for k,v in state.items())
        report['top_level_keys'] = list(ckpt)
        report['hparams'] = hp
        report['tensor_count'] = len(state)
        report['tensor_elements'] = sum(v.numel() for v in state.values())
        report['tensor_shapes'] = {k: list(v.shape) for k,v in state.items()}
        report['floating_tensors_finite'] = all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point())
        report['restricted_load_pass'] = True
        report['status'] = 'downloaded_and_restricted_inspection_passed'
        print(json.dumps({k: report[k] for k in ['bytes','sha256','hparams','tensor_count','floating_tensors_finite']}, indent=2, default=str), flush=True)
    except Exception as e:
        report['status'] = 'failed_or_needs_followup'
        report['error'] = f'{type(e).__name__}: {e}'
        raise
    finally:
        (dest.parent / 'checkpoint_inspection.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

if __name__ == '__main__':
    main()
