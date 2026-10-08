"""Download one author dataset pair for INPUT checks only; not model evaluation."""
from pathlib import Path
import csv
import io
import json
import hashlib
import urllib.request
import concurrent.futures
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/Germany_Training_Public'
BASE='https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/Germany_Training_Public/'
OUT.mkdir(parents=True,exist_ok=True)
mapping_name='Germany_Training_Public_label_image_mapping.csv'
def download(relative):
    path=OUT/relative
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Invalid relative path')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(BASE+relative,timeout=45) as response, path.open('wb') as handle:
            while True:
                block=response.read(1024*1024)
                if not block: break
                handle.write(block)
    result={'relative_path':relative,'url':BASE+relative,'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps(result),flush=True)
    return result
mapping=download(mapping_name)
rows=list(csv.DictReader((OUT/mapping_name).read_text().splitlines()))
row=rows[0]
print('FIRST_MAPPING_ROW='+json.dumps(row),flush=True)
files=[]
for key,value in row.items():
    if not value.strip(): continue
    value=value.strip()
    if '/' in value or '\\' in value:
        raise ValueError('Unexpected mapping filename')
    if key=='label':
        files.append('annotations/'+value)
    elif key=='pre-event image':
        files.append('PRE-event/'+value)
    elif key.startswith('post-event image'):
        files.append('POST-event/'+value)
records=list(concurrent.futures.ThreadPoolExecutor(max_workers=4).map(download,files))
report={'dataset':'SpaceNet8 Germany_Training_Public','selection':'First mapping row, without looking at labels',
        'sample_mapping':row,'files':[mapping]+records,'warning':'Input-readiness sample from PUBLIC TRAINING data; not independent accuracy evaluation.'}
(ROOT/'sample_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
