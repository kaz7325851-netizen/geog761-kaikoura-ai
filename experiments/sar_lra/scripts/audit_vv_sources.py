"""Audit public author archives; download only the small published weights ZIP."""
from pathlib import Path
from datetime import datetime, timezone
import urllib.request
import json
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'vv_only' / 'source_audit'
OUT.mkdir(parents=True, exist_ok=True)
REPO = 'lorenzonava96/SAR-and-DL-for-Landslide-Rapid-Assessment'
report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'sources': {}, 'errors': []}

def get(url, name):
    request = urllib.request.Request(url, headers={'User-Agent': 'GEOG761-source-audit'})
    data = urllib.request.urlopen(request, timeout=60).read()
    (OUT / name).write_bytes(data)
    report['sources'][name] = {'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return json.loads(data)

for record_id in (15159492, 17328763, 14898556):
    try:
        record = get(f'https://zenodo.org/api/records/{record_id}', f'zenodo_{record_id}.json')
        print('ZENODO', record_id, [(x['key'], x['size']) for x in record.get('files', [])], flush=True)
        if record_id == 15159492:
            meta = next(x for x in record['files'] if x['key'] == 'Weights.zip')
            if meta['size'] > 60_000_000:
                raise ValueError('Unexpected archive size: refusing large download')
            dest = OUT / 'Weights.zip'
            if not dest.exists():
                with urllib.request.urlopen(meta['links']['self'], timeout=90) as response, dest.with_suffix('.partial').open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                dest.with_suffix('.partial').replace(dest)
            payload = dest.read_bytes()
            assert len(payload) == meta['size']
            assert 'md5:' + hashlib.md5(payload).hexdigest() == meta['checksum']
            with zipfile.ZipFile(dest) as archive:
                assert archive.testzip() is None
                members = [{'name': x.filename, 'bytes': x.file_size} for x in archive.infolist()]
            report['weights_archive'] = {'url': meta['links']['self'], 'bytes': len(payload), 'published_checksum': meta['checksum'], 'sha256': hashlib.sha256(payload).hexdigest(), 'crc_verified': True, 'members': members}
            print('WEIGHTS MEMBERS', members, flush=True)
    except Exception as exc:
        report['errors'].append({'source': f'zenodo_{record_id}', 'error': str(exc)})
        print('ERROR', record_id, str(exc), flush=True)

for endpoint in ('branches?per_page=100', 'tags?per_page=100', 'releases?per_page=100', 'commits?per_page=100'):
    try:
        result = get(f'https://api.github.com/repos/{REPO}/{endpoint}', 'github_' + endpoint.split('?')[0] + '.json')
        if endpoint.startswith('commits'):
            print('COMMITS', len(result), [(x['sha'][:8], x['commit']['message'][:100]) for x in result], flush=True)
        else:
            print(endpoint, result, flush=True)
    except Exception as exc:
        report['errors'].append({'source': endpoint, 'error': str(exc)})
        print('ERROR', endpoint, str(exc), flush=True)

(OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved', OUT / 'audit.json', flush=True)
