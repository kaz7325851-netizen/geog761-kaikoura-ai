"""Read-only availability checks of author-published saved-model URLs."""
from pathlib import Path
import json
import urllib.request
import concurrent.futures
import datetime
ROOT=Path(__file__).resolve().parent
URLS=[
 'https://ohhan.net/wordpress/wp-content/uploads/2022/08/best_flood.pt',
 'https://ohhan.net/wordpress/wp-content/uploads/2022/08/best_road.pt',
 'https://ohhan.net/wordpress/wp-content/uploads/2022/08/best_building.pt',
 'http://ohhan.net/wordpress/wp-content/uploads/2022/08/best_flood.pt',
 'https://www.dropbox.com/s/2eoruek40n5hpfj/models_sn8_local.tar.gz?dl=1',
 'https://www.dropbox.com/s/o6io0342yb352q3/models_flood.tar.gz?dl=1',
 'https://api.github.com/repos/SpaceNetChallenge/SpaceNet8/releases',
 'https://motokimura-public-sn8.s3.amazonaws.com/exp_50000.zip',
 'https://motokimura-public-sn8.s3.amazonaws.com/exp_50010.zip',
]
def check(url):
    result={'url':url}
    try:
        request=urllib.request.Request(url,method='GET' if 'api.github' in url else 'HEAD',headers={'User-Agent':'GEOG761-model-audit'})
        with urllib.request.urlopen(request,timeout=25) as response:
            result.update(status=response.status,final_url=response.url,headers=dict(response.headers))
            if 'api.github' in url:
                result['releases']=json.load(response)
    except Exception as exc:
        result.update(error=repr(exc))
    return result
results=list(concurrent.futures.ThreadPoolExecutor(max_workers=6).map(check,URLS))
(ROOT/'source_checks.json').write_text(json.dumps({'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':results},indent=2),encoding='utf-8')
for result in results:
    print(json.dumps({k:v for k,v in result.items() if k!='headers'},ensure_ascii=False))
