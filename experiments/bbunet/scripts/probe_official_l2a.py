"""Verify exact public catalog products and anonymous download availability.
Never searches for credentials, submits login forms, or logs access tokens.
"""
from pathlib import Path
import urllib.request,urllib.error,json,datetime
ROOT=Path(__file__).resolve().parent
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT=ROOT/'data_acquisition'/stamp;OUT.mkdir(parents=True,exist_ok=True)
PRODUCTS=[('pre','ead1cbd6-10ec-4333-bfe5-980e1c72add5','S2A_MSIL2A_20161102T221602_N0500_R129_T59GQP_20230922T041941.SAFE'),
    ('post','7a1a0f8a-76ef-4d77-a4d2-b3fb9befde98','S2A_MSIL2A_20161122T221602_N0500_R129_T59GQP_20230923T203359.SAFE')]
records=[]
for role,pid,expected in PRODUCTS:
    record={'role':role,'id':pid,'expected_name':expected}
    url=f'https://catalogue.dataspace.copernicus.eu/odata/v1/Products({pid})'
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'GEOG761-L2A-check'})
        with urllib.request.urlopen(req,timeout=25) as r:data=json.load(r)
        (OUT/f'{role}_catalog.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        assert data['Name']==expected
        record.update({'catalog_verified':True,'online':data.get('Online'),'bytes':data.get('ContentLength'),
            'name':data['Name'],'checksum':data.get('Checksum'),'s3_path':data.get('S3Path')})
    except Exception as e:record.update({'catalog_verified':False,'catalog_error':str(e)})
    dl=f'https://download.dataspace.copernicus.eu/odata/v1/Products({pid})/$value'
    try:
        req=urllib.request.Request(dl,headers={'User-Agent':'GEOG761-L2A-check','Range':'bytes=0-63'})
        with urllib.request.urlopen(req,timeout=25) as r:
            record['anonymous_download_status']=r.status
            magic=r.read(64)
            record['first_bytes_hex']=magic.hex()
            record['downloaded_product']=False
    except urllib.error.HTTPError as e:record['anonymous_download_status']=e.code
    except Exception as e:record['download_error']=str(e)
    records.append(record);print(json.dumps(record),flush=True)
summary={'created_utc':stamp,'records':records,'output_directory':str(OUT),'full_product_downloaded':False,
    'credentials_used':False,'training':False}
(OUT/'acquisition_status.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'output_directory':str(OUT)}),flush=True)
