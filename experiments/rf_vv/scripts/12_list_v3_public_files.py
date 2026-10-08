"""Use the public listing endpoint declared by DesignSafe's official page."""
from pathlib import Path
from urllib.parse import quote
import requests, json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'official_v3_acquisition'
path='/published-data/PRJ-5827'
url='https://www.designsafe-ci.org/api/datafiles/tapis/public/listing/designsafe.storage.published/'+quote(path,safe='')
r=requests.get(url,params={'offset':0,'limit':100,'doi':'10.17603/ds2-rpyg-0h84'},timeout=30)
print('STATUS',r.status_code,flush=True)
r.raise_for_status()
data=r.json()
(OUT/'DesignSafe_root_listing.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
assert data.get('reachedEnd'), 'Unexpected root pagination'
latest=next(x for x in data['listing'] if x['name'].endswith('--V3'))
queue=[latest['path']]; files=[]; listings=[]
while queue:
    path=queue.pop(0)
    url='https://www.designsafe-ci.org/api/datafiles/tapis/public/listing/designsafe.storage.published/'+quote(path,safe='')
    response=requests.get(url,params={'offset':0,'limit':100,'doi':'10.17603/ds2-rpyg-0h84'},timeout=30)
    response.raise_for_status(); content=response.json()
    assert content.get('reachedEnd'), 'Unexpected pagination; do not silently truncate'
    listings.append({'url':response.url,'path':path,'listing':content['listing']})
    for item in content['listing']:
        if item['type']=='dir':queue.append(item['path'])
        else:files.append(item)
    print(json.dumps({'path':path,'items':[{k:i.get(k) for k in ['name','type','length','path']} for i in content['listing']]},ensure_ascii=False),flush=True)
manifest={'publication':'https://doi.org/10.17603/ds2-rpyg-0h84','root':latest['path'],
          'license':'CC-BY-4.0, as declared by DesignSafe public publication page JSON-LD',
          'listings':listings,'files':files}
(OUT/'DesignSafe_V3_files.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('FILES',len(files),flush=True)
