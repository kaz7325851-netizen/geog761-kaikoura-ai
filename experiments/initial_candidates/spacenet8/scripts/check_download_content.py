"""Check payloads, not just HTTP 200; never consume whole unknown archives."""
from pathlib import Path
import urllib.request
import json
import concurrent.futures
import re
ROOT=Path(__file__).resolve().parent
urls=[
 'https://ohhan.net/wordpress/wp-content/uploads/2022/08/best_flood.pt',
 'https://ohhan.net/wordpress/wp-content/uploads/2022/08/best_road.pt',
 'https://ohhan.net/wordpress/wp-content/uploads/2022/08/best_building.pt',
 'https://www.dropbox.com/s/o6io0342yb352q3/models_flood.tar.gz?dl=1',
 'https://dl.dropboxusercontent.com/s/o6io0342yb352q3/models_flood.tar.gz',
 'https://www.dropbox.com/s/2eoruek40n5hpfj/models_sn8_local.tar.gz?dl=1',
 'https://dl.dropboxusercontent.com/s/2eoruek40n5hpfj/models_sn8_local.tar.gz',
 'https://spacenet-dataset.s3.amazonaws.com/?list-type=2&prefix=spacenet-model-weights/spacenet-8/&max-keys=1000',
 'https://spacenet-dataset.s3.amazonaws.com/?list-type=2&prefix=spacenet/SN8_floods/&max-keys=15',
]
def check(url):
    d={'url':url}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'GEOG761-model-audit'})
        with urllib.request.urlopen(req,timeout=30) as r:
            payload=r.read(250000 if 's3.amazonaws' in url else 6000)
            d.update(status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type'),
                     content_length=r.headers.get('Content-Length'),first_bytes_hex=payload[:8].hex())
            if 's3.amazonaws' in url:
                d['listing']=payload.decode('utf-8',errors='replace')
            elif 'text/html' in str(d['content_type']):
                text=payload.decode('utf-8',errors='replace')
                d['html_title']=re.findall(r'<title>(.*?)</title>',text,re.S)
                d['html_excerpt']=text[-1200:]
            else:
                d['gzip_magic_verified']=payload.startswith(b'\x1f\x8b')
    except Exception as exc:
        d['error']=repr(exc)
    return d
results=list(concurrent.futures.ThreadPoolExecutor(max_workers=6).map(check,urls))
(ROOT/'payload_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
for d in results:
    print(json.dumps({k:v for k,v in d.items() if k!='html_excerpt'},ensure_ascii=False))
