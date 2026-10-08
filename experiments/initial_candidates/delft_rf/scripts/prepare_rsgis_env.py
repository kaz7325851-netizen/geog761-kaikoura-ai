"""Resolve a separate author-compatible geospatial environment under testdata."""
from pathlib import Path
import hashlib
import json
import subprocess
import urllib.request
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent

def main():
    url='https://api.github.com/repos/mamba-org/micromamba-releases/releases/tags/2.9.0-0'
    with urllib.request.urlopen(url,timeout=30) as response:
        data=json.load(response)
    asset=next(a for a in data['assets'] if a['name']=='micromamba-win-64.exe')
    exe=ROOT/'tools/micromamba.exe'
    exe.parent.mkdir(parents=True,exist_ok=True)
    if not exe.exists():
        with urllib.request.urlopen(asset['browser_download_url'],timeout=60) as response:
            exe.write_bytes(response.read())
    sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    if asset.get('digest') and asset['digest'] != 'sha256:'+sha:
        raise ValueError('Binary hash mismatch')
    (ROOT/'micromamba_source.json').write_text(json.dumps({'url':asset['browser_download_url'],'sha256':sha,'release_digest':asset.get('digest')},indent=2),encoding='utf-8')
    command=[str(exe), '--root-prefix',str(BASE/'cache/mamba'),'create','--prefix',str(ROOT/'rsgis_env'),
             '--override-channels','-c','conda-forge','--yes','--json',
             'python=3.7','rsgislib=4.1.95','gdal=3.0','rios=1.4.10','numpy=1.19','pandas=1.1',
             'scipy','pyyaml']
    if '--install' not in sys.argv:
        command.append('--dry-run')
    mode='install' if '--install' in sys.argv else 'solve'
    print('Starting isolated environment',mode,flush=True)
    with (ROOT/f'env_{mode}_stdout.json').open('w',encoding='utf-8') as output, (ROOT/f'env_{mode}_stderr.log').open('w',encoding='utf-8') as error:
        result=subprocess.run(command,stdout=output,stderr=error,timeout=240)
    print('Exit',result.returncode,flush=True)
    content=json.loads((ROOT/f'env_{mode}_stdout.json').read_text(encoding='utf-8'))
    print(json.dumps({k:content.get(k) for k in ['success','solver_problems','message']},indent=2),flush=True)
    print('Package count',len(content.get('actions',{}).get('LINK',[])),flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    if mode == 'install':
        # The old conda sklearn archive hits Windows long-path cache validation.
        # Install the already downloaded/hash-verified CPython 3.7 PyPI wheels instead.
        records=json.loads((ROOT/'legacy_sources.json').read_text(encoding='utf-8'))
        target=ROOT/'rsgis_env/Lib/site-packages'
        for record in records:
            if not record['name'].startswith(('scikit_learn-', 'joblib-')):
                continue
            wheel=ROOT/'packages'/record['name']
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Local wheel hash mismatch')
            with zipfile.ZipFile(wheel) as archive:
                for name in archive.namelist():
                    if not (target/name).resolve().is_relative_to(target.resolve()):
                        raise ValueError('Unsafe wheel member')
                archive.extractall(target)
            print('Installed verified local wheel',record['name'],flush=True)

if __name__=='__main__':
    main()
