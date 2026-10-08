"""Validate downloaded image pair and reference annotations, with NO model use."""
from pathlib import Path
import json
import collections
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'sample_manifest.json').read_text())
mapping=manifest['sample_mapping']
base=ROOT/'data/Germany_Training_Public'
report={'status':'input_pair_check_running','dataset':manifest['dataset'],
        'trained_model_used':False,'accuracy_computed':False,'inputs':[]}
pre=base/'PRE-event'/mapping['pre-event image']
post=base/'POST-event'/mapping['post-event image 1']
with rasterio.open(pre) as a,rasterio.open(post) as b:
    for label,ds in [('pre',a),('post',b)]:
        data=ds.read()
        report['inputs'].append({'label':label,'shape':list(data.shape),'dtype':str(data.dtype),
                                 'crs':str(ds.crs),'transform':list(ds.transform)[:6],
                                 'bounds':list(ds.bounds),'resolution':list(ds.res),'finite':bool(np.isfinite(data).all()),
                                 'range':[int(data.min()),int(data.max())]})
        assert ds.count==3 and np.isfinite(data).all()
    aligned=np.zeros((3,a.height,a.width),dtype=np.uint8)
    for i in range(3):
        reproject(source=rasterio.band(b,i+1),destination=aligned[i],src_transform=b.transform,src_crs=b.crs,
                  dst_transform=a.transform,dst_crs=a.crs,resampling=Resampling.bilinear)
    report['alignment_check']={'target_shape':list(aligned.shape),'same_crs':a.crs==b.crs,
                               'original_same_grid':a.transform==b.transform and a.shape==b.shape,
                               'post_to_pre_grid_resampling_executed':True,'finite':bool(np.isfinite(aligned).all())}
labels=json.loads((base/'annotations'/mapping['label']).read_text())
features=labels['features']
report['annotations']={'features':len(features),'geometry_types':dict(collections.Counter(f['geometry']['type'] for f in features)),
                       'property_keys':sorted(set(k for f in features for k in f.get('properties',{}))),
                       'flooded_values':dict(collections.Counter(str(f.get('properties',{}).get('flooded')) for f in features))}
report['status']='input_pair_and_annotations_readable'
report['warning']='Public TRAINING subset and no task weights. Not an independent accuracy test and no predictions produced.'
(ROOT/'sample_input_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
