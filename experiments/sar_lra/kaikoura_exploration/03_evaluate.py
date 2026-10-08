"""Run the original ascending author checkpoint on the frozen older-baseline pilot."""
from pathlib import Path
import os
import sys
import json
import csv
import time
import hashlib
from datetime import datetime
ROOT = Path(__file__).resolve().parent
SAR = ROOT.parent
os.environ['MPLCONFIGDIR'] = str(SAR.parent/'cache/matplotlib')
sys.path.insert(0,str(SAR))
from check_weights import load_model, tf
from utils import sliding_window, non_max_suppression_fast
from importlib.util import spec_from_file_location, module_from_spec
import numpy as np
import rasterio
from rasterio.warp import transform_geom
import h5py
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
config = json.loads((ROOT/'experiment_config.json').read_text(encoding='utf-8'))
manifest = json.loads((ROOT/'latest_input.json').read_text(encoding='utf-8'))
input_path = Path(manifest['input_file'])
assert hashlib.sha256(input_path.read_bytes()).hexdigest() == manifest['sha256']
spec = spec_from_file_location('download_validation', ROOT/'02_download_input.py')
validator = module_from_spec(spec)
spec.loader.exec_module(validator)
input_checks = validator.validate_raster(input_path,config)
with rasterio.open(input_path) as src:
    data = src.read()
    profile = src.profile.copy()
    grid = (src.shape,src.crs,src.transform)
    affine = src.transform
with rasterio.open(ROOT/'reference/reference_mask.tif') as src:
    assert (src.shape,src.crs,src.transform) == grid
    reference = src.read(1)
with rasterio.open(ROOT/'reference/road_corridor_mask.tif') as src:
    assert (src.shape,src.crs,src.transform) == grid
    road_corridor = src.read(1)
image = data[:4].transpose(1,2,0).astype(np.float32)
valid = data[4] == 1
coords,patches,fractions,road_fractions = [],[],[],[]
SIZE,STEP,THRESHOLD,BATCH = 64,32,0.6,32
candidate_count = 0
for x,y,patch in sliding_window(image, STEP, (SIZE,SIZE)):
    candidate_count += 1
    if not valid[y:y+SIZE,x:x+SIZE].all():
        continue
    coords.append((x,y))
    patches.append(patch)
    fractions.append(float(reference[y:y+SIZE,x:x+SIZE].mean()))
    road_fractions.append(float(road_corridor[y:y+SIZE,x:x+SIZE].mean()))
assert patches, 'No complete valid windows; no inference performed'
coords = np.asarray(coords,dtype=np.int32)
patches = np.stack(patches)
fractions = np.asarray(fractions)
road_fractions = np.asarray(road_fractions)
truth = fractions > 0.05
t = time.perf_counter()
model,details = load_model('ASCENDING')
details['load_seconds'] = time.perf_counter()-t
t = time.perf_counter()
warm = model(patches[:1],training=False).numpy().reshape(-1)
details['first_patch_seconds'] = time.perf_counter()-t
t = time.perf_counter()
scores = np.concatenate([model(patches[i:i+BATCH],training=False).numpy().reshape(-1) for i in range(0,len(patches),BATCH)])
details['warm_inference_seconds'] = time.perf_counter()-t
assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
repeat = model(patches[:2],training=False).numpy().reshape(-1)
assert np.allclose(warm,scores[:1],atol=1e-6,rtol=1e-6)
assert np.allclose(repeat,scores[:2],atol=1e-6,rtol=1e-6)
details['repeat_max_absolute_difference'] = float(np.max(np.abs(repeat-scores[:2])))
checkpoint = SAR/'official/SAR-LRA Tool V1/model_weights'/details['checkpoint']
with h5py.File(checkpoint,'r') as h:
    saved = [np.asarray(h[name][w]) for name in h.attrs['layer_names'] for w in h[name].attrs['weight_names']]
assert all(np.array_equal(a,b) for a,b in zip(saved,model.get_weights()))
details['weights_unchanged_after_inference'] = True
details['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
pred = scores > THRESHOLD

def metrics(selected):
    t,p = truth[selected],pred[selected]
    tp,fp,fn,tn = [int(x) for x in (np.sum(t&p),np.sum(~t&p),np.sum(t&~p),np.sum(~t&~p))]
    divide = lambda a,b:float(a/b) if b else None
    return {'windows':len(t),'reference_positive_windows':int(t.sum()),'predicted_positive_windows':int(p.sum()),
            'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':divide(tp,tp+fp),'recall':divide(tp,tp+fn),
            'F1':divide(2*tp,2*tp+fp+fn),'IoU':divide(tp,tp+fp+fn),'accuracy':divide(tp+tn,len(t)),
            'positive_support_too_small_for_general_claims':int(t.sum())<30}

boxes = np.asarray([(int(x),int(y),int(x+SIZE),int(y+SIZE)) for (x,y),positive in zip(coords,pred) if positive],dtype=np.int32).reshape(-1,4)
kept = np.asarray(non_max_suppression_fast(boxes,.1),dtype=np.int32).reshape(-1,4)
scored = np.zeros(reference.shape,dtype=bool)
max_score = np.full(reference.shape,-9999,dtype=np.float32)
for (x,y),score in zip(coords,scores):
    scored[y:y+SIZE,x:x+SIZE] = True
    max_score[y:y+SIZE,x:x+SIZE] = np.maximum(max_score[y:y+SIZE,x:x+SIZE],score)
coverage = np.zeros(reference.shape,dtype=np.uint8)
for x1,y1,x2,y2 in kept:
    coverage[y1:y2,x1:x2] = 1
coverage[~scored] = 255
OUT = ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S')
OUT.mkdir(parents=True,exist_ok=False)
np.savez_compressed(OUT/'patch_predictions.npz',coords=coords,scores=scores,reference_fraction=fractions,
                    road_corridor_fraction=road_fractions,truth=truth,pred=pred,boxes=kept)
with (OUT/'patch_predictions.csv').open('w',encoding='utf-8',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x','y','reference_fraction','reference_positive_gt_5pct','score','predicted_positive_gt_0_6','road_corridor_fraction'])
    for (x,y),frac,t,s,p,r in zip(coords,fractions,truth,scores,pred,road_fractions):
        writer.writerow([int(x),int(y),frac,int(t),float(s),int(p),r])
profile.pop('photometric',None)
for name,arr,dtype,nodata in [('suspect_box_coverage.tif',coverage,'uint8',255),('overlapping_window_max_score.tif',max_score,'float32',-9999)]:
    profile.update(count=1,dtype=dtype,nodata=nodata,compress='deflate')
    with rasterio.open(OUT/name,'w',**profile) as dst:
        dst.write(arr,1)
        dst.update_tags(meaning='Coarse window score/box coverage, not landslide boundary or road safety',experiment='82-day-old baseline; not original pre60 deployment')
features = []
for x1,y1,x2,y2 in kept:
    ring = [affine*(int(x),int(y)) for x,y in [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]]
    geom = transform_geom('EPSG:2193','EPSG:4326',{'type':'Polygon','coordinates':[ring]})
    idx = np.flatnonzero((coords[:,0]==x1)&(coords[:,1]==y1))[0]
    features.append({'type':'Feature','properties':{'window_score':float(scores[idx]),'pass':'ASCENDING','relative_orbit':52,
                     'meaning':'suspect 640m window, not landslide boundary or actual road closure'},'geometry':geom})
(OUT/'detections.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}),encoding='utf-8')
recent_valid = valid & (data[9]==1)
baseline_shift = data[8,recent_valid]
road_shift = data[8,recent_valid & (road_corridor==1)]
report = {'created':datetime.now().isoformat(),'configuration':config,'input_sha256':manifest['sha256'],
          'input_file':str(input_path),'input_checks':input_checks,'model':details,
          'all_valid_window_metrics':metrics(np.ones(len(coords),dtype=bool)),
          'window_center_in_old_road_corridor_metrics':metrics(np.asarray([bool(road_corridor[y+32,x+32]) for x,y in coords])),
          'candidate_windows':candidate_count,'excluded_incomplete_observation_windows':candidate_count-len(coords),
          'small_nonzero_reference_le_5pct_windows':int(np.sum((fractions>0)&(~truth))),
          'retained_boxes_after_nms':len(kept),'scored_pixels':int(scored.sum()),'unscored_pixels':int((~scored).sum()),
          'reference_pixels_in_unscored_areas':int(reference[~scored].sum()),
          'score_quantiles':np.quantile(scores,[0,.25,.5,.75,1]).tolist(),
          'baseline_diagnostic':{'recent_VV_minus_Aug23_VV_dB_quantiles':np.quantile(baseline_shift,[.05,.25,.5,.75,.95]).tolist(),
              'fraction_abs_shift_gt_3dB':float((np.abs(baseline_shift)>3).mean()),
              'old_road_corridor_abs_shift_gt_3dB_fraction':float((np.abs(road_shift)>3).mean()),
              'interpretation':'Observed SAR baseline difference; cannot isolate seasonal effects or prove VH stability. Includes speckle/geometry/moisture/land cover.',
              'recent_vv_median_vs_single_date_is_confounder':True},
          'classification_threshold':THRESHOLD,'reference_rule':'Inventory pixel fraction > 0.05 in a 64x64 window',
          'NMS':'Original author V1 y2-sorted, intersection/other-area threshold0.1, not score-sorted IoU',
          'process_RSS_MB_after_inference':psutil.Process().memory_info().rss/1024**2,'RSS_is_not_peak':True,
          'training_performed':False,'inference_performed':True,
          'limitations':['Small geometry-selected pilot, not final research area or event-wide accuracy','82-day single-date baseline, not 60-day median; no controlled time-window ablation possible without near-event VH',
                         'Raw optical inventory without ascending-specific shadow/layover cleaning','<=5% reference windows treated negative even if real small landslides exist',
                         'Overlapping windows are spatially correlated','Kaikoura absent from paper VV_VH development events, but exact checkpoint lineage independently unverified',
                         'Current OSM corridor not verified historic 2016 roads','No road closure, landing point or route feasibility validation','10m NZTM grid differs from previous Sumatra EPSG4326 pilot grid']}
(OUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

fig,axes = plt.subplots(2,3,figsize=(14,9),layout='constrained')
extent = [0,640,640,0]
valid_diff = np.where(valid,data[2],np.nan)
im = axes[0,0].imshow(valid_diff,cmap='RdBu_r',vmin=-6,vmax=6,extent=extent)
axes[0,0].set_title('Observed postVV - Aug23 preVV (dB)')
fig.colorbar(im,ax=axes[0,0],shrink=.7)
axes[0,1].imshow(reference,cmap=ListedColormap(['#eeeeee','#d49426']),vmin=0,vmax=1,extent=extent)
axes[0,1].contour(road_corridor,levels=[.5],colors=['#4198a4'],linewidths=.6)
axes[0,1].set_title('Reference landslide pixels + old road corridor')
axes[0,2].imshow(np.ma.masked_where(coverage==255,coverage),cmap=ListedColormap(['#eeeeee','#cd5c5c']),vmin=0,vmax=1,extent=extent)
axes[0,2].set_title(f'Retained coarse detection boxes ({len(kept)})\nGrey: no detection (not safe); white: unscored')
im = axes[1,0].imshow(np.where(recent_valid,data[8],np.nan),cmap='RdBu_r',vmin=-6,vmax=6,extent=extent)
axes[1,0].set_title('Recent preVV median - Aug23 preVV (dB)')
fig.colorbar(im,ax=axes[1,0],shrink=.7)
score_grid = np.full((18,18),np.nan)
comparison_grid = np.full((18,18),np.nan)
for (x,y),s,t,p in zip(coords,scores,truth,pred):
    score_grid[y//32,x//32] = s
    comparison_grid[y//32,x//32] = 3 if t and p else 2 if t else 1 if p else 0
im = axes[1,1].imshow(score_grid,vmin=0,vmax=1,cmap='viridis')
axes[1,1].set_title('Overlapping WINDOW scores (18x18 positions)')
fig.colorbar(im,ax=axes[1,1],shrink=.7)
cm = ListedColormap(['#d6d6d6','#df6a58','#466ab7','#3b9f76'])
axes[1,2].imshow(comparison_grid,cmap=cm,vmin=-.5,vmax=3.5,interpolation='nearest')
axes[1,2].set_title('WINDOW comparison (>5% reference; score>0.6)\nWhite positions were excluded')
axes[1,2].legend(handles=[Patch(color=cm.colors[i],label=label) for i,label in enumerate(['TN','FP','FN','TP'])],loc='upper center',bbox_to_anchor=(.5,-.02),ncol=4)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Kaikoura 2016 SAR-LRA exploration: real ascending checkpoint\n82-day-old baseline; coarse slide windows, NOT road passability',fontsize=14)
fig.savefig(OUT/'comparison.png',dpi=150)
plt.close(fig)
(ROOT/'latest_result.json').write_text(json.dumps({'result_directory':str(OUT)},indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['all_valid_window_metrics','window_center_in_old_road_corridor_metrics','candidate_windows','excluded_incomplete_observation_windows','retained_boxes_after_nms','score_quantiles','baseline_diagnostic']},indent=2),flush=True)
print('RESULT_DIRECTORY='+str(OUT),flush=True)
