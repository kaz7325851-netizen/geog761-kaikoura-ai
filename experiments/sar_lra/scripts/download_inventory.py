from pathlib import Path
import urllib.request
import json
import hashlib
import zipfile
ROOT=Path(__file__).resolve().parent
record=json.load(urllib.request.urlopen('https://zenodo.org/api/records/15159492',timeout=45))
meta=next(x for x in record['files'] if x['key']=='landslide_inventories_polygons.zip')
dest=ROOT/'data/landslide_inventories_polygons.zip'
dest.parent.mkdir(parents=True,exist_ok=True)
url=meta['links']['self']
if not dest.exists():
    with urllib.request.urlopen(url,timeout=90) as response, dest.with_suffix('.partial').open('wb') as output:
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            output.write(chunk)
    dest.with_suffix('.partial').replace(dest)
payload=dest.read_bytes()
assert len(payload)==meta['size']
assert 'md5:'+hashlib.md5(payload).hexdigest()==meta['checksum']
with zipfile.ZipFile(dest) as archive:
    bad=archive.testzip()
    if bad:raise ValueError(bad)
    names=archive.namelist()
    print('\n'.join(names),flush=True)
report={'zenodo_record':15159492,'url':url,'bytes':len(payload),'published_checksum':meta['checksum'],
        'md5_passed':True,'sha256':hashlib.sha256(payload).hexdigest(),'zip_crc_passed':True,'members':names}
(ROOT/'inventory_source.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
