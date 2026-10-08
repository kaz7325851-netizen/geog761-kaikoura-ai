"""Public SENCLOUD12 weight acquisition, no arbitrary pickle load."""
import hashlib
import json
import time
import zipfile
import socket
from pathlib import Path
from datetime import datetime, timezone
import requests
ROOT = Path(__file__).resolve().parent
URL = 'https://nextcloud.bgc-jena.mpg.de/s/qHKcyZpzHtXnzL2/download/mobilenetv2_l2a_all.pth'
dest = ROOT / 'weights' / 'mobilenetv2_l2a_all.pth'
report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source': URL,
          'referenced_by': 'author src/cloud_detector.py', 'model_run': False,
          'published_checksum_available': False}
try:
    if not dest.exists():
        hostname = 'nextcloud.bgc-jena.mpg.de'
        try:
            socket.getaddrinfo(hostname, 443)
        except socket.gaierror:
            # Scoped DNS fallback only; HTTPS still verifies the original host.
            dns = requests.get('https://dns.google/resolve', params={'name': hostname, 'type':'A'}, timeout=20)
            dns.raise_for_status()
            answer = dns.json()
            addresses = [a['data'] for a in answer.get('Answer',[]) if a['type']==1]
            if answer.get('Status') != 0 or not addresses:
                raise RuntimeError('Public DNS did not provide an A record')
            report['dns_fallback'] = {'service':'https://dns.google/resolve', 'hostname':hostname,
                                      'resolved_A':addresses, 'system_settings_changed':False,
                                      'TLS_verification_disabled':False}
            original_getaddrinfo = socket.getaddrinfo
            def scoped_getaddrinfo(host, port, *args, **kwargs):
                return original_getaddrinfo(addresses[0] if host==hostname else host, port, *args, **kwargs)
            socket.getaddrinfo = scoped_getaddrinfo
        partial = dest.with_suffix('.pth.partial')
        if partial.exists():
            raise RuntimeError('Partial download preserved; inspect before retry')
        with requests.get(URL, stream=True, timeout=(20,60)) as r:
            r.raise_for_status()
            report['content_type'] = r.headers.get('Content-Type')
            expected = r.headers.get('Content-Length')
            if 'text/html' in r.headers.get('Content-Type', ''):
                raise RuntimeError('HTML response instead of weights')
            count = 0
            last = time.monotonic()
            with partial.open('xb') as f:
                for block in r.iter_content(1024*1024):
                    if block:
                        f.write(block)
                        count += len(block)
                    if time.monotonic()-last>10:
                        print(f'Cloud weights downloaded {count:,} bytes', flush=True)
                        last = time.monotonic()
            if expected and count != int(expected):
                raise RuntimeError('Incomplete transfer; partial retained')
            if count < 1000000:
                raise RuntimeError('Unexpectedly small download; partial retained')
            partial.rename(dest)
    report['bytes'] = dest.stat().st_size
    report['sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
    if zipfile.is_zipfile(dest):
        with zipfile.ZipFile(dest) as z:
            assert z.testzip() is None
        report['zip_crc_pass'] = True
    import torch
    torch.set_num_threads(2)
    # Reject unsupported pickled globals; do not fall back to weights_only=False.
    state = torch.load(dest, map_location='cpu', weights_only=True)
    assert isinstance(state, dict) and all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items())
    assert all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point())
    report['tensor_count'] = len(state)
    report['tensor_shapes'] = {k:list(v.shape) for k,v in state.items()}
    report['restricted_load_pass'] = True
    report['status'] = 'downloaded_and_restricted_inspection_passed'
except Exception as e:
    report['status'] = 'failed_or_needs_followup'
    report['error'] = f'{type(e).__name__}: {e}'
    raise
finally:
    (ROOT / 'weights' / 'cloud_weight_inspection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='tensor_shapes'},indent=2),flush=True)
