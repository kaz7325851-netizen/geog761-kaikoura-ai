"""Download author-published KARI checkpoint; record source and local integrity."""
from pathlib import Path
import urllib.request
import hashlib
import json
import time
import zipfile
import concurrent.futures
ROOT=Path(__file__).resolve().parent
URL='https://ohhan.net/wordpress/wp-content/uploads/2022/08/best_flood.pt'
dest=ROOT/'weights/best_flood.pt'
dest.parent.mkdir(parents=True,exist_ok=True)
expected=513082249
if not dest.exists() or dest.stat().st_size!=expected:
    parts=ROOT/'weights/kari_parts'
    parts.mkdir(exist_ok=True)
    chunk=8*1024*1024
    start=time.monotonic()
    def fetch(index):
        first=index*chunk
        last=min(expected,first+chunk)-1
        target=parts/('part_%03d'%index)
        count=target.stat().st_size if target.exists() else 0
        if count==last-first+1: return target
        request=urllib.request.Request(URL,headers={'Range':'bytes=%d-%d'%(first+count,last)})
        with urllib.request.urlopen(request,timeout=60) as r,target.open('ab') as h:
            if r.status!=206 or r.headers.get('Content-Range')!='bytes %d-%d/%d'%(first+count,last,expected):
                raise ValueError('Unexpected range response')
            while True:
                block=r.read(256*1024)
                if not block: break
                h.write(block)
        if target.stat().st_size != last-first+1:
            raise ValueError('Range length mismatch')
        print('Finished chunk',index,'elapsed s',round(time.monotonic()-start,1),flush=True)
        return target
    count=(expected+chunk-1)//chunk
    # Keep partial parts for recovery; combine only complete verified byte ranges.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        paths=list(pool.map(fetch,range(count)))
    partial=dest.with_suffix('.pt.assembled')
    with partial.open('wb') as handle:
        for path in paths:
            with path.open('rb') as source:
                while True:
                    block=source.read(1024*1024)
                    if not block: break
                    handle.write(block)
    if partial.stat().st_size != expected:
        raise ValueError('Checkpoint length differs from author response')
    partial.replace(dest)
with zipfile.ZipFile(dest) as archive:
    bad=archive.testzip()
    if bad: raise ValueError('ZIP CRC failed: '+bad)
report={'url':URL,'author_repository':'https://github.com/SpaceNetChallenge/SpaceNet8/tree/main/01-ohhan777/code',
        'repo_revision':'30ca30c2530c6a234d040f60a99a7366ff62d756',
        'filename':dest.name,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'zip_crc_passed':True,'hash_note':'Locally recorded SHA256, no independently published author digest available.'}
(ROOT/'weight_source.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
