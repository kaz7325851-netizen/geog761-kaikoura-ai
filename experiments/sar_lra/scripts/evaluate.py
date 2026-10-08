"""Fixed Sumatra pilot: author checkpoints, real SAR inputs, patch-level labels.

No training, threshold tuning, class balancing or architecture changes.
Outputs are coarse window classifications, NOT pixelwise landslide segmentation.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT.parent / 'cache/matplotlib')
from check_weights import load_model, tf
from utils import sliding_window, non_max_suppression_fast
import numpy as np
import rasterio
import json
import csv
import time
import hashlib
import psutil
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
OUT = ROOT / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True)
SIZE, STEP, THRESHOLD, BATCH = 64, 32, 0.6, 32

def metrics(truth, pred):
    tp = int(np.sum(truth & pred)); fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred)); tn = int(np.sum(~truth & ~pred))
    div = lambda a,b: float(a/b) if b else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=div(tp,tp+fp),
                recall=div(tp,tp+fn), F1=div(2*tp,2*tp+fp+fn),
                IoU=div(tp,tp+fp+fn), accuracy=div(tp+tn,len(truth)))

with rasterio.open(ROOT / 'data/reference/reference_mask.tif') as ref:
    reference = ref.read(1)
    reference_transform, reference_crs = ref.transform, ref.crs

rows = []
visuals = []
for pass_ in ['ASCENDING', 'DESCENDING']:
    with rasterio.open(ROOT / 'data' / (pass_+'.tif')) as src:
        assert src.transform == reference_transform and src.crs == reference_crs
        assert src.shape == reference.shape and src.count == 5
        data = src.read()
        profile = src.profile.copy()
        transform = src.transform
    image = np.transpose(data[:4], (1,2,0)).astype(np.float32)
    valid = data[4] == 1
    assert np.isfinite(image[valid]).all()
    patches, coords, fractions = [], [], []
    for x,y,patch in sliding_window(image, STEP, (SIZE,SIZE)):
        if not valid[y:y+SIZE,x:x+SIZE].all():
            continue
        patches.append(patch)
        coords.append((x,y))
        fractions.append(float(reference[y:y+SIZE,x:x+SIZE].mean()))
    patches = np.stack(patches)
    coords = np.asarray(coords, dtype=np.int32)
    fractions = np.asarray(fractions)
    truth = fractions > 0.05
    t = time.perf_counter()
    model, details = load_model(pass_)
    details['load_seconds'] = time.perf_counter()-t
    # Actual first patch warms execution; this is NOT a synthetic accuracy sample.
    t = time.perf_counter()
    warm = model(patches[:1], training=False).numpy().reshape(-1)
    details['first_patch_cold_seconds'] = time.perf_counter()-t
    t = time.perf_counter()
    probabilities = np.concatenate([
        model(patches[i:i+BATCH], training=False).numpy().reshape(-1)
        for i in range(0,len(patches),BATCH)])
    details['batch_inference_seconds'] = time.perf_counter()-t
    assert len(probabilities) == len(truth) and np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    repeat = model(patches[:2], training=False).numpy().reshape(-1)
    details['repeat_max_absolute_difference'] = float(np.max(np.abs(repeat-probabilities[:2])))
    assert np.allclose(repeat, probabilities[:2], atol=1e-6, rtol=1e-6)
    assert np.allclose(warm, probabilities[:1], atol=1e-6, rtol=1e-6)
    # Verify every weight remains unchanged after all inference calls.
    import h5py
    path = ROOT / 'official/SAR-LRA Tool V1/model_weights' / details['checkpoint']
    with h5py.File(path,'r') as h:
        saved = [np.asarray(h[name][w]) for name in h.attrs['layer_names']
                 for w in h[name].attrs['weight_names']]
    assert all(np.array_equal(a,b) for a,b in zip(saved,model.get_weights()))
    details['weights_unchanged_after_inference'] = True
    pred = probabilities > THRESHOLD
    boxes = np.asarray([(int(x),int(y),int(x+SIZE),int(y+SIZE))
                        for (x,y),positive in zip(coords,pred) if positive],dtype=np.int32).reshape(-1,4)
    kept = np.asarray(non_max_suppression_fast(boxes,0.1),dtype=np.int32).reshape(-1,4)
    scored = np.zeros(reference.shape,dtype=bool)
    max_score = np.full(reference.shape,-9999,dtype=np.float32)
    for (x,y),p in zip(coords,probabilities):
        scored[y:y+SIZE,x:x+SIZE] = True
        max_score[y:y+SIZE,x:x+SIZE] = np.maximum(max_score[y:y+SIZE,x:x+SIZE],p)
    coverage = np.zeros(reference.shape,dtype=np.uint8)
    for x1,y1,x2,y2 in kept:
        coverage[y1:y2,x1:x2] = 1
    coverage[~scored] = 255
    pass_dir = OUT / pass_
    pass_dir.mkdir()
    np.savez_compressed(pass_dir/'patch_predictions.npz',coords=coords,probabilities=probabilities,
                        reference_fraction=fractions,truth=truth,pred=pred,boxes=kept)
    with (pass_dir/'patch_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['x','y','reference_fraction','reference_positive_gt_5pct','score','predicted_positive_gt_0_6'])
        for (x,y),frac,label,p,positive in zip(coords,fractions,truth,probabilities,pred):
            writer.writerow([int(x),int(y),frac,int(label),float(p),int(positive)])
    profile.pop('photometric',None)
    profile.update(count=1,compress='deflate',dtype='uint8',nodata=255)
    with rasterio.open(pass_dir/'suspect_box_coverage.tif','w',**profile) as dst:
        dst.write(coverage,1)
        dst.update_tags(meaning='Coverage by coarse retained 64x64 detection boxes; NOT slide boundaries',
                        zero='No retained detection, NOT verified safe',nodata='Unscored border')
    profile.update(dtype='float32',nodata=-9999)
    with rasterio.open(pass_dir/'overlapping_window_max_score.tif','w',**profile) as dst:
        dst.write(max_score,1)
        dst.update_tags(meaning='Maximum overlapping WINDOW score; NOT calibrated pixel probability')
    features = []
    for x1,y1,x2,y2 in kept:
        ring = [transform*(int(x),int(y)) for x,y in [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]]
        index = np.flatnonzero((coords[:,0]==x1)&(coords[:,1]==y1))[0]
        features.append({'type':'Feature','properties':{'pass':pass_,'window_score':float(probabilities[index]),
                         'meaning':'suspect 640m window, not landslide boundary'},
                         'geometry':{'type':'Polygon','coordinates':[ring]}})
    (pass_dir/'detections.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}),encoding='utf-8')
    details.update(metrics(truth,pred))
    details.update(windows=len(truth),reference_positive_windows=int(truth.sum()),
        small_nonzero_reference_below_or_equal_5pct_windows=int(np.sum((fractions>0)&(~truth))),
        positive_windows_before_nms=int(pred.sum()),retained_boxes=len(kept),
        score_min=float(probabilities.min()),score_max=float(probabilities.max()),
        score_median=float(np.median(probabilities)),scored_pixels=int(scored.sum()),
        unscored_pixels=int((~scored).sum()),process_RSS_MB_after_inference=psutil.Process().memory_info().rss/1024**2,
        process_RSS_is_not_peak=True,
        input_sha256=hashlib.sha256((ROOT/'data'/(pass_+'.tif')).read_bytes()).hexdigest())
    rows.append(details)
    # 19x19 cells below represent overlapping windows, not 19x19 ground pixels.
    ys, xs = np.unique(coords[:,1]), np.unique(coords[:,0])
    assert len(coords) == len(xs)*len(ys)
    confusion = np.where(truth,np.where(pred,3,2),np.where(pred,1,0)).reshape(len(ys),len(xs))
    visuals.append((pass_,data[2],reference,probabilities.reshape(len(ys),len(xs)),confusion,details))
    print(json.dumps(details),flush=True)

report = {'test':'Fixed Sumatra 2022 real-SAR pilot, no training', 'created':datetime.now().isoformat(),
    'event':'2022-02-25 Sumatra earthquake','aoi_requested':[99.985,0.060,100.045,0.120],
    'raster_shape':list(reference.shape),'window_size_pixels':SIZE,'window_step_pixels':STEP,
    'classification_threshold':THRESHOLD,'reference_positive_rule':'inventory pixel fraction > 0.05',
    'batch_size':BATCH,'tensorflow':tf.__version__,'devices':[str(d) for d in tf.config.list_physical_devices()],
    'evaluation':'All valid overlapping windows before NMS, NOT pixel segmentation accuracy',
    'same_locations_observed_twice_not_independent_samples':True,
    'paper_reproduction_limitations':['Small single-event pilot, not full-area paper reproduction',
       'No pass-specific SAR shadow/layover mask cleaning; only acquisition-coverage mask',
       'Sumatra declared unseen by paper; checkpoint training provenance not independently audited',
       'Unlabelled inventory background is not certified safe; <=5% slide windows counted negative by rule',
       'No Kaikoura road closures, beach suitability, transport feasibility or transfer accuracy evaluated',
       'TensorFlow 2.16 legacy-Keras runtime rather than original TensorFlow/Keras 2.8',
       'Overlapping spatial windows are correlated; no confidence interval or test threshold fitting'],
    'NMS':'Author V1, overlap=intersection/other_box_area, threshold0.1, y2 sorted NOT score sorted',
    'raster_output_adaptation':'Filled box coverage for GIS; author V1 original raster draws only box outlines',
    'results':rows}
(OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
fig,axes = plt.subplots(2,4,figsize=(15,8),layout='constrained')
cm = ListedColormap(['#d6d6d6','#df6a58','#466ab7','#3b9f76'])
for r,(pass_,diff,ref,probs,conf,detail) in enumerate(visuals):
    im=axes[r,0].imshow(diff,cmap='RdBu_r',vmin=-6,vmax=6)
    axes[r,0].set_title(pass_+' dVV (dB)')
    fig.colorbar(im,ax=axes[r,0],shrink=.6)
    axes[r,1].imshow(ref,cmap=ListedColormap(['#eeeeee','#c58828']),vmin=0,vmax=1)
    axes[r,1].set_title('Reference slide pixels (669x669)')
    im=axes[r,2].imshow(probs,cmap='viridis',vmin=0,vmax=1)
    axes[r,2].set_title('WINDOW scores (19x19, cutoff 0.6)')
    fig.colorbar(im,ax=axes[r,2],shrink=.6)
    axes[r,3].imshow(conf,cmap=cm,vmin=-.5,vmax=3.5,interpolation='nearest')
    axes[r,3].set_title(f"WINDOW comparison\nP={detail['precision']:.1%}, R={detail['recall']:.1%}, F1={detail['F1']:.1%}")
    for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([])
axes[1,3].legend(handles=[Patch(color=cm.colors[i],label=l) for i,l in enumerate(['TN','FP','FN','TP'])],
                 loc='upper center',bbox_to_anchor=(.5,-.02),ncol=4,fontsize=9)
fig.suptitle('SAR-LRA real-weight pilot: Sumatra 2022\nReference >5% per 64x64 window; overlapping windows, not road or pixel accuracy',fontsize=13)
fig.savefig(OUT/'comparison.png',dpi=150)
plt.close(fig)
(ROOT/'latest_result.json').write_text(json.dumps({'result_directory':str(OUT)},indent=2),encoding='utf-8')
print('RESULT_DIRECTORY='+str(OUT),flush=True)
