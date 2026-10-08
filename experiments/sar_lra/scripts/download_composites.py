from pathlib import Path
import urllib.request
import hashlib
import json
import concurrent.futures
ROOT=Path(__file__).resolve().parent
URLS={
 'ASCENDING': '<GENERATE_FRESH_EARTH_ENGINE_URL>',
 'DESCENDING': '<GENERATE_FRESH_EARTH_ENGINE_URL>'
}
SCENES={
 'ASCENDING':{'relative_orbit':142,'pre_dates':['2022-01-06','2022-01-18','2022-01-30','2022-02-11','2022-02-23'],'post_dates':['2022-03-07']},
 'DESCENDING':{'relative_orbit':164,'pre_dates':['2021-12-26','2022-01-07','2022-01-19','2022-01-31','2022-02-12'],'post_dates':['2022-03-08']}
}
def download(pass_):
    dest=ROOT/'data'/(pass_+'.tif')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(URLS[pass_],timeout=120) as response:
        data=response.read()
    if not data.startswith((b'II*\x00',b'MM\x00*')):raise ValueError('Not TIFF')
    dest.write_bytes(data)
    print(pass_,len(data),'bytes downloaded',flush=True)
    return {'pass':pass_,'file':dest.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),**SCENES[pass_]}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:files=list(pool.map(download,URLS))
report={'event':'2022-02-25 Sumatra earthquake','aoi_bounds':[99.985,.060,100.045,.120],
        'selection':'Fixed author-area subrectangle before model predictions; no outcome-based selection',
        'collection':'COPERNICUS/S1_GRD','pre_window':['2021-12-26','2022-02-24'],
        'post_window':['2022-02-26','2022-03-10'],'filterDate_end_exclusive':True,
        'source_script':'02_sumatra_composites.js','scale_m':10,'crs':'EPSG:4326',
        'band_order':['postVV','postVH','diffVV','diffVH','valid'],
        'processing':'GEE dB-space median per pass/relative orbit; post minus pre; no additional normalisation or speckle filter',
        'quality':'valid band is acquisition coverage only, NOT SAR shadow/layover mask; those masks not yet applied',
        'files':files}
(ROOT/'data_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
