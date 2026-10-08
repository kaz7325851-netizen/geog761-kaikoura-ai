"""Independent artifact verification: recompute labels/metrics without loading any model."""
from pathlib import Path
import json
import csv
import hashlib
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform
import pyproj

ROOT = Path(__file__).resolve().parent
OUT = Path(json.loads((ROOT/'latest_result.json').read_text())['result_directory'])
report = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
saved = np.load(OUT/'patch_predictions.npz')
with rasterio.open(report['input_file']) as src:
    data = src.read()
    grid = src.shape,src.crs,src.transform
    affine = src.transform
with rasterio.open(ROOT/'reference/reference_mask.tif') as src:
    reference = src.read(1)
    assert (src.shape,src.crs,src.transform) == grid
with rasterio.open(ROOT/'reference/road_corridor_mask.tif') as src:
    road_corridor = src.read(1)
    assert (src.shape,src.crs,src.transform) == grid
# Rerasterize the saved source polygons independently of the earlier reference TIFF.
convert = pyproj.Transformer.from_crs(4326,2193,always_xy=True).transform
features = json.loads((ROOT/'reference/pilot_inventory.geojson').read_text())['features']
independent = rasterize([(transform(convert,shape(x['geometry'])),1) for x in features],
                        out_shape=reference.shape,transform=affine,fill=0,all_touched=False,dtype='uint8')
assert np.array_equal(reference,independent)
valid = data[4] == 1
expected_coords = np.asarray([(x,y) for y in range(0,576,32) for x in range(0,576,32) if valid[y:y+64,x:x+64].all()],dtype=np.int32)
assert np.array_equal(expected_coords,saved['coords'])
coords,scores = saved['coords'],saved['scores']
fractions = np.asarray([reference[y:y+64,x:x+64].mean() for x,y in coords])
assert np.array_equal(fractions,saved['reference_fraction'])
truth,pred = fractions>.05,scores>.6
assert np.array_equal(truth,saved['truth']) and np.array_equal(pred,saved['pred'])
counts = {'TP':int(np.sum(truth&pred)),'FP':int(np.sum(~truth&pred)),
          'FN':int(np.sum(truth&~pred)),'TN':int(np.sum(~truth&~pred))}
assert all(v==report['all_valid_window_metrics'][k] for k,v in counts.items())
tp,fp,fn,tn = [counts[k] for k in ['TP','FP','FN','TN']]
assert report['all_valid_window_metrics']['F1'] == 2*tp/(2*tp+fp+fn)
assert report['all_valid_window_metrics']['precision'] is None if tp+fp==0 else report['all_valid_window_metrics']['precision']==tp/(tp+fp)
with (OUT/'patch_predictions.csv').open(encoding='utf-8',newline='') as handle:
    rows = list(csv.DictReader(handle))
assert len(rows)==len(coords)
assert np.array_equal(np.asarray([float(x['score']) for x in rows],dtype='float32'),scores)
scored = np.zeros(reference.shape,dtype=bool)
expected_coverage = np.zeros(reference.shape,dtype=np.uint8)
expected_score = np.full(reference.shape,-9999,dtype='float32')
for (x,y),score in zip(coords,scores):
    scored[y:y+64,x:x+64]=True
    expected_score[y:y+64,x:x+64]=np.maximum(expected_score[y:y+64,x:x+64],score)
for x1,y1,x2,y2 in saved['boxes']:
    expected_coverage[y1:y2,x1:x2]=1
expected_coverage[~scored]=255
for name,expected,nodata in [('suspect_box_coverage.tif',expected_coverage,255),('overlapping_window_max_score.tif',expected_score,-9999)]:
    with rasterio.open(OUT/name) as src:
        assert (src.shape,src.crs,src.transform)==grid and src.nodata==nodata
        assert np.array_equal(src.read(1),expected)
assert len(json.loads((OUT/'detections.geojson').read_text())['features'])==len(saved['boxes'])==report['retained_boxes_after_nms']
assert hashlib.sha256(Path(report['input_file']).read_bytes()).hexdigest()==report['input_sha256']
result = {'status':'PASS','reference_rerasterization_exact':True,'valid_window_positions_recomputed':True,
          'metrics_recomputed':counts,'csv_scores_exact':True,'geotiff_grids_values_and_nodata_exact':True,
          'geojson_box_count_verified':True,'input_sha256_verified':True,'inference_rerun':False,
          'reference_positive_score_quantiles':np.quantile(scores[truth],[0,.25,.5,.75,1]).tolist(),
          'reference_negative_score_quantiles':np.quantile(scores[~truth],[0,.25,.5,.75,1]).tolist(),
          'scored_pixels':int(scored.sum()),'unscored_pixels':int((~scored).sum()),
          'reference_pixels_in_unscored_areas':int(reference[~scored].sum())}
(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
