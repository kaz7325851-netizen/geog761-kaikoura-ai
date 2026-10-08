"""Prepare an isolated portable legacy Python; never alter the Prithvi environment."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent

def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    return hashlib.sha256(dest.read_bytes()).hexdigest()

def safe_extract(z, root):
    root.mkdir(parents=True, exist_ok=True)
    for name in z.namelist():
        resolved = (root / name).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError('Unsafe archive member')
    z.extractall(root)

def main():
    archive=ROOT/'packages/python-3.7.9-embed-amd64.zip'
    sha=download('https://www.python.org/ftp/python/3.7.9/python-3.7.9-embed-amd64.zip',archive)
    runtime=ROOT/'legacy_python'
    if not (runtime/'python.exe').exists():
        with zipfile.ZipFile(archive) as z:
            safe_extract(z,runtime)
    records=[{'name':archive.name,'url':'https://www.python.org/ftp/python/3.7.9/python-3.7.9-embed-amd64.zip','sha256':sha}]
    for package,version in [('numpy','1.16.6'),('scipy','1.2.3'),('scikit-learn','0.20.3'),('joblib','0.13.2')]:
        with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json',timeout=30) as r:
            metadata=json.load(r)
        candidates=[f for f in metadata['urls'] if f['filename'].endswith('cp37-cp37m-win_amd64.whl') or
                    (package=='joblib' and f['filename'].endswith('py2.py3-none-any.whl'))]
        if len(candidates)!=1:
            raise ValueError((package,[f['filename'] for f in candidates]))
        f=candidates[0]
        dest=ROOT/'packages'/f['filename']
        sha=download(f['url'],dest)
        if sha!=f['digests']['sha256']:
            raise ValueError('Package hash mismatch')
        with zipfile.ZipFile(dest) as z:
            safe_extract(z,runtime/'Lib/site-packages')
        records.append({'name':f['filename'],'url':f['url'],'sha256':sha,'pypi_hash_verified':True})
        print('Prepared',package,version,flush=True)
    (ROOT/'legacy_sources.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Legacy runtime/packages prepared. Enable site-packages using the local _pth file.',flush=True)

if __name__=='__main__':
    main()
