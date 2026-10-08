"""Save public official metadata/pages; no credentials or access-control changes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import requests, json, re, hashlib

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'official_v3_acquisition'
OUT.mkdir(exist_ok=True)
SOURCES={
 'CKAN_metadata.json':'https://catalogue.data.govt.nz/api/3/action/package_show?id=kaikura-2016-landslide-inventory-version-3',
 'DesignSafe_public_page.html':'https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-5827/',
 'DesignSafe_public_imports.js':'https://www.designsafe-ci.org/static/react-assets/assets/imports-Cd1JlHR4.js',
 'GNS_public_page.html':'https://data.gns.cri.nz/landslides/wms.html',
 'GNS_public_main.js':'https://data.gns.cri.nz/landslides/static/js/main.e8e25078.js',
}
def fetch(item):
    name,url=item
    try:
        r=requests.get(url,timeout=22);r.raise_for_status()
        dest=OUT/name;dest.write_bytes(r.content)
        result={'file':str(dest),'url':url,'final_url':r.url,'status':r.status_code,
                'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest()}
    except requests.RequestException as exc:
        result={'url':url,'status':'request_failed','error':str(exc)}
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result
with ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(fetch,SOURCES.items()))
(OUT/'public_source_manifest.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'records':results},indent=2),encoding='utf-8')
for name in ['DesignSafe_public_imports.js','GNS_public_main.js','DesignSafe_public_page.html']:
    p=OUT/name
    if not p.exists():continue
    t=p.read_text(encoding='utf-8')
    pattern=r'file-listing|files/list|files/download|api/datafiles|api/files|/api/data|public/list|downloadDataset|GetFeature|geoserver|wfs|landslide_v3|kaikoura|PRJ-5827|project-publications'
    matches=[{'term':m.group(),'context':t[max(0,m.start()-130):m.start()+220]} for m in re.finditer(pattern,t,re.I)]
    (OUT/(p.stem+'_endpoint_contexts.json')).write_text(json.dumps(matches,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'source':name,'contexts':matches[:40]},ensure_ascii=False),flush=True)
