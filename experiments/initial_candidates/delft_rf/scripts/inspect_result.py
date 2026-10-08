"""Independent output checks and a non-evaluation diagnostic plot."""
from pathlib import Path
import os
import sys
import json
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT.parent / 'cache/matplotlib')
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

out = Path(sys.argv[1])
meta = json.loads((out / 'run_metadata.json').read_text())
assert meta['status'] == 'actual_author_sample_inference_complete'
rows = pd.read_csv(out / 'classifier_features_predictions.csv')
with rasterio.open(out / 'prediction.tif') as ds:
    mask = ds.read(1)
    assert ds.nodata == 255
    transform, crs = ds.transform, ds.crs
    values, counts = np.unique(mask, return_counts=True)
    measured = {str(int(v)): int(c) for v, c in zip(values, counts)}
    assert measured == meta['pixel_counts']
with rasterio.open(ROOT / 'official/datasets/input/satellite_imagery/post_event_image/post_landslide_80.tif') as ds:
    assert ds.crs == crs and ds.transform == transform
    rgb = np.moveaxis(ds.read([1, 2, 3]), 0, -1)
    assert rgb.shape[:2] == mask.shape
rgb = np.nan_to_num(rgb)
low = np.percentile(rgb, 2, axis=(0, 1))
high = np.percentile(rgb, 98, axis=(0, 1))
rgb = np.clip((rgb - low) / np.maximum(high - low, 1e-6), 0, 1)
colors = np.full((*mask.shape, 3), .65)
colors[mask == 0] = [.15, .5, .75]
colors[mask == 1] = [.9, .15, .1]
overlay = rgb.copy()
overlay[mask == 1] = .4 * rgb[mask == 1] + .6 * np.array([1, 0, 0])
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, data, title in zip(axes, [rgb, overlay, colors],
        ['Author sample RGB (display stretch)', 'RF positive overlay', 'RF classification / coverage']):
    ax.imshow(data)
    ax.set_title(title)
    ax.axis('off')
fig.legend(handles=[Patch(color=(.9,.15,.1), label='RF landslide prediction'),
                    Patch(color=(.15,.5,.75), label='RF negative (NOT safe road)'),
                    Patch(color=(.65,.65,.65), label='Excluded / unknown')],
           loc='lower center', ncol=3, bbox_to_anchor=(.5,.07))
fig.suptitle('Delft RF: actual saved-model inference on author imagery sample', fontsize=15)
fig.text(.5,.02,'No reference labels: this figure cannot establish accuracy or road passability.', ha='center')
fig.tight_layout(rect=(0,.14,1,.94))
fig.savefig(out / 'sample_prediction.png', dpi=160)
plt.close(fig)
checks = {'verified_raster_counts': measured, 'verified_grid_alignment': True,
          'prediction_rows': len(rows), 'all_features_finite': bool(np.isfinite(rows.iloc[:,1:].values).all()),
          'no_reference_labels': True}
assert checks['all_features_finite']
(out / 'independent_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print(json.dumps(checks, indent=2))
