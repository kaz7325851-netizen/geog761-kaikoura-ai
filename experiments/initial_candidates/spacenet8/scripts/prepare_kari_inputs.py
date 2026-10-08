"""Run with isolated GDAL 3.0.2 environment; reproduce author's test image warp."""
from pathlib import Path
import json
import numpy as np
from osgeo import gdal

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT/'test_sample_manifest.json').read_text())
data = ROOT/'data/Louisiana-West_Test_Public'
out = ROOT/'prepared_inputs'
out.mkdir(exist_ok=True)
checks = []
for sample in manifest['samples']:
    sid = Path(sample['label']).stem
    pre = gdal.Open(str(data/'PRE-event'/sample['pre-event image']))
    post = gdal.Warp('', str(data/'POST-event'/sample['post-event image 1']),
                     format='MEM', width=1300, height=1300,
                     resampleAlg=gdal.GRIORA_Bilinear, outputType=gdal.GDT_Byte)
    a, b = pre.ReadAsArray(), post.ReadAsArray()
    assert a.shape == b.shape == (3,1300,1300)
    assert a.dtype == b.dtype == np.uint8
    np.savez_compressed(str(out/(sid+'.npz')), pre=a, post=b)
    checks.append({'sample':sid,'shape':list(a.shape),'pre_transform':list(pre.GetGeoTransform()),
                   'warped_post_transform':list(post.GetGeoTransform()),
                   'same_transform':pre.GetGeoTransform()==post.GetGeoTransform(),
                   'method':'Exact author gdal.Warp MEM 1300x1300 bilinear Byte; first post only'})
    print(sid, 'prepared', flush=True)
(out/'checks.json').write_text(json.dumps({'gdal':gdal.VersionInfo(), 'samples':checks},indent=2))
