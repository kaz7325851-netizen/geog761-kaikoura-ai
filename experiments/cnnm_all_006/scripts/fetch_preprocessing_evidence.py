"""Retrieve small public author metadata; no login or external messages."""
from pathlib import Path
import urllib.request,json,hashlib
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'preprocessing_audit/public_sources'
OUT.mkdir(parents=True,exist_ok=True)
commit='2268ee5aab022454a21344c3601069afd6054690'
base=f'https://raw.githubusercontent.com/nprksh/landslide-mapping-with-cnn/{commit}/'
tree=json.loads((ROOT/'sources/repository_tree.json').read_text())['tree']
urls=[(x['path'].replace('/','__'),base+x['path']) for x in tree if x['path'].endswith('.aux.xml')]
urls += [('issues.json','https://api.github.com/repos/nprksh/landslide-mapping-with-cnn/issues?state=all&per_page=100'),
    ('commits.json','https://api.github.com/repos/nprksh/landslide-mapping-with-cnn/commits?per_page=100')]
records=[]
for name,url in urls:
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'GEOG761-preprocessing-audit'})
        with urllib.request.urlopen(req,timeout=20) as resp:
            data=resp.read(2_000_001)
        assert len(data)<=2_000_000
        (OUT/name).write_bytes(data)
        records.append({'file':name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'success':True})
    except Exception as e:
        records.append({'file':name,'url':url,'success':False,'error':str(e)})
    print(json.dumps(records[-1]),flush=True)
(OUT/'download_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
