"""Official anonymous DesignSafe download flow, selecting GIS + HTML metadata.

The public PUT returns a temporary download link, not a modification to source data.
No user credentials are read or used. Extract whitelisted basenames only.
"""
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from datetime import datetime, timezone
import requests, json, hashlib, zipfile, re

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'official_v3_acquisition'
listing=json.loads((OUT/'DesignSafe_V3_files.json').read_text(encoding='utf-8'))
selected=[f for f in listing['files'] if '/Shapefiles/' in f['path'] or '/Metadata/' in f['path'] or f['name']=='manifest-sha512.txt']
assert len(selected)==19 and len(set(f['name'] for f in selected))==19
session=requests.Session()
landing='https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-5827/'
response=session.get(landing,timeout=30);response.raise_for_status()
endpoint='https://www.designsafe-ci.org/api/datafiles/tapis/public/download/designsafe.storage.published/?doi=10.17603/ds2-rpyg-0h84'
headers={'Referer':landing}
# Standard CSRF token from our own anonymous session only, if the page supplies it.
if session.cookies.get('csrftoken'):headers['X-CSRFToken']=session.cookies.get('csrftoken')
response=session.put(endpoint,json={'paths':[f['path'] for f in selected]},headers=headers,timeout=35)
print('PUBLIC_DOWNLOAD_LINK_STATUS',response.status_code,flush=True)
response.raise_for_status(); ticket=response.json()
href=ticket['href']; assert urlparse(href).scheme=='https'
archive=OUT/'GNS_V3_GIS_metadata_official.zip'
with session.get(href,stream=True,timeout=(20,35)) as stream:
    stream.raise_for_status(); length=stream.headers.get('Content-Length'); total=0
    with archive.open('wb') as dest:
        for chunk in stream.iter_content(1024*1024):
            if chunk:dest.write(chunk);total+=len(chunk)
    if length:assert total==int(length)
print('ARCHIVE_DOWNLOADED_BYTES',total,flush=True)
components=OUT/'components';components.mkdir(exist_ok=True)
records=[]
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for f in selected:
        matches=[n for n in z.namelist() if PurePosixPath(n).name==f['name'] and not n.endswith('/')]
        assert len(matches)==1,(f['name'],matches)
        b=z.read(matches[0]); assert len(b)==f['length'],(f['name'],len(b),f['length'])
        dest=components/f['name']; assert dest.resolve().parent==components.resolve()
        dest.write_bytes(b)
        records.append({'file':str(dest),'name':f['name'],'source_path':f['path'],
                        'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
                        'sha512':hashlib.sha512(b).hexdigest(),'publisher_sha512_verified':False})
manifest=(components/'manifest-sha512.txt').read_text(encoding='utf-8')
hashes=[]
for line in manifest.splitlines():
    m=re.match(r'^([0-9a-fA-F]{128})\s+\*?(.+)$',line)
    if m:hashes.append((m[1].lower(),PurePosixPath(m[2]).name))
assert hashes, 'Publisher checksum format not recognized'
for r in records:
    if r['name']=='manifest-sha512.txt':continue
    matched=[h for h,n in hashes if n==r['name']]
    assert len(matched)==1 and matched[0]==r['sha512'],r['name']
    r['publisher_sha512_verified']=True
report={'status':'official_V3_GIS_downloaded_publisher_checksums_passed',
        'downloaded_utc':datetime.now(timezone.utc).isoformat(),'publication':listing['publication'],
        'license':listing['license'],'selected_file_count':len(records),'download_url_host':urlparse(href).hostname,
        'archive':str(archive),'archive_bytes':total,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'records':records,'source_listing_file':str(OUT/'DesignSafe_V3_files.json'),
        'training':False,'notes':['Only official Shapefile components, HTML metadata, publisher manifest downloaded',
                                'No CSV or DOCX content processed','Source files not modified']}
(OUT/'download_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':report['status'],'files':len(records),'bytes':total},ensure_ascii=False),flush=True)
