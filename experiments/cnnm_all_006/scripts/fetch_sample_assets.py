"""Read-only public author sample acquisition. Downloaded Python is not executed."""
from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor
import json,hashlib
ROOT=Path(__file__).resolve().parent
COMMIT='2268ee5aab022454a21344c3601069afd6054690'
BASE='https://raw.githubusercontent.com/nprksh/landslide-mapping-with-cnn/'+COMMIT+'/'
paths=['Utils/tileHelpers.py','Utils/gdalCommonUtils.py','Utils/IO.py',
'RawData/AOI_E_S2-S2/DEM/DEM_030.tif',
'RawData/AOI_E_S2-S2/DEM/DEM_030_HILLSHADE.tif',
'RawData/AOI_E_S2-S2/DEM/DEM_030_SLOPE.tif',
'RawData/AOI_E_S2-S2/fromOptical/JIUZ_POST_S2_RGB_010_UINT8.tif',
'RawData/AOI_E_S2-S2/fromOptical/JIUZ_PRE_S2_RGB_010_UINT8.tif',
'RawData/AOI_E_S2-S2/noDataMask/SNOW_CLOUD_MASK_010.tif',
'RawData/AOI_E_S2-S2/testBoundary/Test_006.tif',
'Mapping_results/Predict_LS_Conf.tif']
def fetch(path):
    dest=ROOT/'sources'/path;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with urlopen(Request(BASE+path,headers={'User-Agent':'GEOG761-research'}),timeout=60) as resp,dest.open('wb') as f:
            n=0
            while chunk:=resp.read(1024*1024):
                n+=len(chunk);assert n<=20_000_000;f.write(chunk)
    result={'path':path,'file':str(dest),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    print(json.dumps(result),flush=True);return result
with ThreadPoolExecutor(max_workers=2) as pool: records=list(pool.map(fetch,paths))
(ROOT/'sources/sample_download_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
dest=ROOT/'sources/paper_fulltext.xml'
try:
    with urlopen('https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8102623/fullTextXML',timeout=30) as resp:
        content=resp.read(5_000_000)
    assert b'<article' in content
    dest.write_bytes(content)
    print('Author paper XML downloaded; bytes '+str(len(content)),flush=True)
except Exception as e:print('Paper XML unavailable: '+str(e)[:200],flush=True)
