"""First three public test tiles, selected before viewing labels or predictions."""
from pathlib import Path
import urllib.request
import csv
import json
import hashlib
import concurrent.futures
ROOT=Path(__file__).resolve().parent
AOI='Louisiana-West_Test_Public'
BASE='https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/'+AOI+'/'
OUT=ROOT/'data'/AOI
OUT.mkdir(parents=True,exist_ok=True)
def download(relative):
    target=OUT/relative
    if not target.resolve().is_relative_to(OUT.resolve()): raise ValueError('Unsafe path')
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        if not target.exists():
            with urllib.request.urlopen(BASE+relative,timeout=45) as r,target.open('wb') as f:
                while True:
                    block=r.read(1024*1024)
                    if not block: break
                    f.write(block)
        d={'path':relative,'url':BASE+relative,'bytes':target.stat().st_size,
           'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    except Exception as exc:
        d={'path':relative,'url':BASE+relative,'error':repr(exc)}
    print(json.dumps(d),flush=True)
    return d
mapping_name=AOI+'_label_image_mapping.csv'
mapping=download(mapping_name)
rows=list(csv.DictReader((OUT/mapping_name).read_text().splitlines()))[:3]
files=[]
for row in rows:
    for key,value in row.items():
        value=value.strip()
        if not value: continue
        if '/' in value or '\\' in value: raise ValueError('Unexpected filename')
        if key=='label': files.append('annotations/'+value)
        elif key=='pre-event image': files.append('PRE-event/'+value)
        elif key=='post-event image 1': files.append('POST-event/'+value)
results=list(concurrent.futures.ThreadPoolExecutor(max_workers=6).map(download,files))
report={'dataset':AOI,'selection':'First 3 rows before looking at labels or model output',
        'samples':rows,'files':[mapping]+results,
        'post_choice':'post-event image 1 only, following first-place author test loader',
        'warning':'Small public test pilot. Checkpoint training exposure not independently audited.'}
(ROOT/'test_sample_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Test sample preparation complete',flush=True)
