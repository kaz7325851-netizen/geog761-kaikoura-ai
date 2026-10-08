"""Independently recompute saved pilot results; no model imported or rerun."""
from pathlib import Path
import json
import csv
import numpy as np
import rasterio
ROOT=Path(__file__).resolve().parent
OUT=Path(json.loads((ROOT/'latest_result.json').read_text())['result_directory'])
report=json.loads((OUT/'metrics.json').read_text())
checks=[]
with rasterio.open(ROOT/'data/reference/reference_mask.tif') as src:
    ref=src.read(1); grid=(src.shape,src.crs,src.transform)
for row in report['results']:
    pass_=row['pass']; d=OUT/pass_
    saved=np.load(d/'patch_predictions.npz')
    coords=saved['coords']; scores=saved['probabilities']
    fractions=np.asarray([ref[y:y+64,x:x+64].mean() for x,y in coords])
    assert np.array_equal(fractions,saved['reference_fraction'])
    truth=fractions>0.05; pred=scores>0.6
    counts={'TP':int(np.sum(truth&pred)),'FP':int(np.sum(~truth&pred)),
            'FN':int(np.sum(truth&~pred)),'TN':int(np.sum(~truth&~pred))}
    assert all(row[k]==v for k,v in counts.items())
    assert sum(counts.values())==361
    f1=2*counts['TP']/(2*counts['TP']+counts['FP']+counts['FN'])
    assert abs(f1-row['F1'])<1e-12
    with (d/'patch_predictions.csv').open(newline='',encoding='utf-8') as f:
        csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==361
    assert np.array_equal(np.asarray([float(r['score']) for r in csv_rows],dtype='float32'),scores)
    with rasterio.open(d/'suspect_box_coverage.tif') as src:
        assert (src.shape,src.crs,src.transform)==grid and src.nodata==255
        cov=src.read(1)
    expected=np.zeros(ref.shape,dtype='uint8'); scored=np.zeros(ref.shape,dtype=bool)
    for x,y in coords:scored[y:y+64,x:x+64]=True
    for x1,y1,x2,y2 in saved['boxes']:expected[y1:y2,x1:x2]=1
    expected[~scored]=255
    assert np.array_equal(expected,cov)
    assert int(np.sum(cov==255))==row['unscored_pixels']
    features=json.loads((d/'detections.geojson').read_text())['features']
    assert len(features)==len(saved['boxes'])==row['retained_boxes']
    check={'pass':pass_,'recomputed_counts':counts,'F1':f1,'saved_scores_equal_csv':True,
        'geotiff_grid_and_box_coverage_verified':True,'geojson_box_count_verified':True,
        'nodata_pixels':int(np.sum(cov==255))}
    checks.append(check)
(OUT/'verification.json').write_text(json.dumps({'checks':checks,'status':'PASS'},indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
