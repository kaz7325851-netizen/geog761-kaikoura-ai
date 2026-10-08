"""Independent output grid/count check, input quality, and FP32 vs mixed precision comparison."""
from pathlib import Path
import json
import numpy as np
import rasterio
ROOT=Path(__file__).resolve().parent
dirs=sorted((ROOT/'results').glob('*/metrics.json'))
assert len(dirs)>=2
mixed,full=dirs[-2].parent,dirs[-1].parent
manifest=json.loads((ROOT/'test_sample_manifest.json').read_text())
report={'mixed_precision_run':str(mixed),'fp32_run':str(full),'samples':[]}
for sample in manifest['samples']:
    sid=Path(sample['label']).stem
    with rasterio.open(full/(sid+'_prediction.tif')) as dst:
        pred=dst.read(1)
        profile={'crs':str(dst.crs),'transform':list(dst.transform),'shape':list(pred.shape)}
    with rasterio.open(mixed/(sid+'_prediction.tif')) as dst: other=dst.read(1)
    path=ROOT/'data/Louisiana-West_Test_Public/POST-event'/sample['post-event image 1']
    with rasterio.open(path) as src:
        original=src.read()
        source={'shape':list(original.shape),'nodata':src.nodata,
                'masked_fraction':float((src.dataset_mask()==0).mean()),
                'all_rgb_zero_fraction':float(np.all(original==0,axis=0).mean())}
    with rasterio.open(ROOT/'data/Louisiana-West_Test_Public/PRE-event'/sample['pre-event image']) as pre:
        assert profile['crs']==str(pre.crs) and profile['transform']==list(pre.transform)
        assert pred.shape==(pre.height,pre.width)
    prepped=np.load(ROOT/'prepared_inputs'/(sid+'.npz'),allow_pickle=False)['post']
    zeros=np.all(prepped==0,axis=0)
    report['samples'].append({'id':sid,'class_counts':np.bincount(pred.ravel(),minlength=5).tolist(),
                             'class_valid':bool(np.isin(pred,range(5)).all()),'grid_verified':True,
                             'fp32_autocast_changed_pixels':int((pred!=other).sum()),
                             'changed_fraction':float((pred!=other).mean()),'post_original':source,
                             'prepared_all_rgb_zero_fraction':float(zeros.mean()),
                             'prediction_counts_on_zero_rgb':np.bincount(pred[zeros],minlength=5).tolist()})
report['interpretation']='All-zero RGB / declared nodata is input coverage, NOT a valid negative label. Do not claim road safety there.'
(full/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
