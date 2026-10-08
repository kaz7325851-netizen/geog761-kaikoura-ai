"""Download pinned public author artifacts; do not execute downloaded code."""
from pathlib import Path
from urllib.request import Request,urlopen
from datetime import datetime,timezone
import hashlib,json,time

ROOT=Path(__file__).resolve().parent
COMMIT='2268ee5aab022454a21344c3601069afd6054690'
BASE='https://raw.githubusercontent.com/nprksh/landslide-mapping-with-cnn/'+COMMIT+'/'
OUT=ROOT/'sources';OUT.mkdir(parents=True,exist_ok=True)
def fetch(url,dest,expected=None,limit=300_000_000):
    if dest.exists() and expected and hashlib.sha256(dest.read_bytes()).hexdigest()==expected:
        print('Verified cached '+dest.name,flush=True);return
    print('Downloading '+dest.name,flush=True)
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'GEOG761-research'}),timeout=60) as response, dest.open('wb') as f:
                total=0;last=time.monotonic();digest=hashlib.sha256()
                while chunk:=response.read(1024*1024):
                    total+=len(chunk);assert total<=limit;f.write(chunk);digest.update(chunk)
                    if time.monotonic()-last>10:
                        print(f'{dest.name}: {total/1e6:.1f} MB received',flush=True);last=time.monotonic()
            if expected:assert digest.hexdigest()==expected,'SHA256 differs from Git LFS pointer'
            print(json.dumps({'file':str(dest),'bytes':total,'sha256':digest.hexdigest()}),flush=True)
            return
        except Exception as e:
            print(type(e).__name__+': '+str(e)[:200],flush=True)
            if attempt==2:raise
            time.sleep(1)
fetch('https://api.github.com/repos/nprksh/landslide-mapping-with-cnn/git/trees/'+COMMIT+'?recursive=1',OUT/'repository_tree.json',limit=5_000_000)
for name in ['README.md','LS_mapping.ipynb']:
    fetch(BASE+name,OUT/name,limit=30_000_000)
pointer=OUT/'M_ALL_006.lfs_pointer.txt'
fetch(BASE+'M_ALL_006.hdf5',pointer,limit=1000)
weights=ROOT/'weights';weights.mkdir(exist_ok=True)
fetch('https://media.githubusercontent.com/media/nprksh/landslide-mapping-with-cnn/'+COMMIT+'/M_ALL_006.hdf5',weights/'M_ALL_006.hdf5',expected='24b81b52411917bb8b69dc87b994478a4ddbdce92c4f5c42e6b4061a763b6a3b')
print('Author code saved for inspection only; no model loaded or inference performed.',flush=True)
