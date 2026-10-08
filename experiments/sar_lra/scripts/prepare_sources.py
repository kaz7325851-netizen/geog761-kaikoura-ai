"""Freeze and download only public author code and two published model weights."""
from pathlib import Path
import urllib.request
import json
import hashlib
import concurrent.futures
ROOT=Path(__file__).resolve().parent
REPO='lorenzonava96/SAR-and-DL-for-Landslide-Rapid-Assessment'
def get_json(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'GEOG761-model-test'}),timeout=40))
commit=get_json('https://api.github.com/repos/'+REPO+'/commits/main')['sha']
tree=get_json('https://api.github.com/repos/'+REPO+'/git/trees/'+commit+'?recursive=1')['tree']
(ROOT/'author_tree.json').write_text(json.dumps({'commit':commit,'tree':tree},indent=2),encoding='utf-8')
selected=[x for x in tree if x['type']=='blob' and (x['path'] in ['README.md','LICENSE'] or
    (x['path'].startswith(('SAR-LRA Tool V1/','SAR-LRA Tool V2/','SAR-LRA Tool V1 for Google Colaboratory/'))
     and x['path'].endswith(('.ipynb','.py','.hdf5','.md'))))]
def download(x):
    rel=Path(x['path'])
    dest=ROOT/'official'/rel
    if not dest.resolve().is_relative_to((ROOT/'official').resolve()):raise ValueError(rel)
    dest.parent.mkdir(parents=True,exist_ok=True)
    url='https://raw.githubusercontent.com/'+REPO+'/'+commit+'/'+urllib.parse.quote(x['path'])
    if not dest.exists():
        payload=urllib.request.urlopen(url,timeout=90).read()
        if len(payload)!=x['size']:raise ValueError('Size mismatch '+x['path'])
        dest.write_bytes(payload)
    payload=dest.read_bytes()
    git_sha=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if git_sha!=x['sha']:raise ValueError('Git blob mismatch '+x['path'])
    print('Verified',x['path'],len(payload),flush=True)
    return {'path':x['path'],'url':url,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'git_blob_sha1':git_sha}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(download,selected))
(ROOT/'source_manifest.json').write_text(json.dumps({'repo':REPO,'revision':commit,'files':files},indent=2),encoding='utf-8')
print('REVISION='+commit,flush=True)
