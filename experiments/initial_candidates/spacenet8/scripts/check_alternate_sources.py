"""Search public author/official indices only; no credentials or model execution."""
from pathlib import Path
import json
import urllib.request
import concurrent.futures
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
urls=[
 'https://spacenet-dataset.s3.amazonaws.com/?list-type=2&prefix=spacenet-model-weights/&delimiter=/&max-keys=1000',
 'https://huggingface.co/api/models?search=spacenet8&limit=30',
 'https://api.github.com/repos/motokimura/spacenet8_solution_5th-place/releases',
 'https://www.dropbox.com/s/3dg4qvd2c0pyj2k/models_sn8.tar.gz?dl=1',
 'https://motokimura-public-sn8.s3.amazonaws.com/?list-type=2&max-keys=1000',
]
def check(url):
    d={'url':url}
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'GEOG761-model-audit'}),timeout=25) as r:
            b=r.read(250000)
            d.update(status=r.status,content_type=r.headers.get('Content-Type'))
            if 'api' in url:
                d['data']=json.loads(b)
            elif b.startswith(b'<?xml'):
                root=ET.fromstring(b)
                d['keys']=[e.text for e in root.iter() if e.tag.endswith('Key') or e.tag.endswith('Prefix')]
            else:
                d['first_bytes_hex']=b[:8].hex()
                d['is_archive']=b.startswith(b'\x1f\x8b')
    except Exception as exc:
        d['error']=repr(exc)
    return d
results=list(concurrent.futures.ThreadPoolExecutor(max_workers=5).map(check,urls))
(ROOT/'alternate_source_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
for d in results:
    print(json.dumps(d,ensure_ascii=False))
