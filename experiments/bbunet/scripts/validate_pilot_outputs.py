"""Independent file-level recalculation and targeted visual diagnostic cases."""
import os,json,hashlib
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as transform_geom
from scipy.ndimage import label
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'pilot_results'/'20261007T203251Z'
INPUT=ROOT/'model_inputs'/'20261007T203029Z'
DESIGN=ROOT.parent/'pilot_design'/'20261005T200713Z'
STAGE=ROOT/'prepared_inputs'/'20261007T193918Z'
r=json.loads((OUT/'pilot_results.json').read_text())
def read(fn):
    with rasterio.open(fn) as ds:return ds.read(),ds.profile.copy()
valid,profile=read(OUT/'evaluation_mask.tif');valid=valid[0].astype(bool)
ref=read(OUT/'reference_inventory_center_raster.tif')[0][0].astype(bool)
road=read(STAGE/'road_100m_context_mask.tif')[0][0].astype(bool)
features=json.loads((DESIGN/'reference_inventory.geojson').read_text())['features']
project=Transformer.from_crs(4326,2193,always_xy=True).transform
polys={f['properties']['source_index']:transform_geom(project,shape(f['geometry'])) for f in features}
recreated=rasterize([(g,1) for g in polys.values()],out_shape=(640,640),transform=profile['transform'],dtype='uint8').astype(bool)
assert np.array_equal(recreated,ref)
verified={}
for name,v in r['variants'].items():
    for o in v['outputs']:assert hashlib.sha256(Path(o['file']).read_bytes()).hexdigest()==o['sha256']
    prob=read(OUT/f'{name}_probability_full_context.tif')[0][0]
    screened=read(OUT/f'{name}_probability_screened.tif')[0][0]
    binary=read(OUT/f'{name}_positive_screened.tif')[0][0]
    assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
    assert np.array_equal(screened[valid],prob[valid]) and (screened[~valid]==-9999).all()
    assert np.array_equal(binary[valid],prob[valid]>=.5) and (binary[~valid]==255).all()
    pos=(prob>=.5)&valid
    metrics=v['metrics'];assert pos.sum()==metrics['predicted_positive_evaluated_pixels']
    assert (pos&ref).sum()==metrics['positive_within_reference_pixels']
    assert (pos&road).sum()==metrics['road100m']['positive_pixels']
    objects=json.loads((OUT/f'{name}_inventory_objects.json').read_text())
    near=[o for o in objects if o['distance_to_sh1_m']<=100]
    assert len(near)==29
    for o in objects:
        pm=rasterize([(polys[o['source_index']],1)],out_shape=(640,640),transform=profile['transform'],dtype='uint8').astype(bool)
        assert int((pm&valid).sum())==o['evaluated_pixels']
        assert int((pm&pos).sum())==o['positive_pixels']
    verified[name]={'probability_range_pass':True,'hashes_pass':True,'nodata_and_threshold_pass':True,
                    'union_counts_pass':True,'object_raster_counts_recreated':len(objects),
                    'near_road_missed_source_indices':[o['source_index'] for o in near if o['positive_pixels']==0],
                    'near_road_missed_retained_fraction_min':min(o['retained_fraction'] for o in near if o['positive_pixels']==0),
                    'road100m_positive_outside_inventory_NOT_confirmed_FP':int((pos&road&~ref).sum())}
np_prob=read(OUT/'Horn_numpy_probability_full_context.tif')[0][0]
pre=read(INPUT/'pre_optical_12channels_normalized.tif')[0]
post=read(INPUT/'post_optical_12channels_normalized.tif')[0]
objs=json.loads((OUT/'Horn_numpy_inventory_objects.json').read_text())
large=next(o for o in sorted(objs,key=lambda x:x['source_area_m2']) if o['source_area_m2']>=10000 and o['distance_to_sh1_m']<=100)
def pixel_of_point(pt):
    row,col=rasterio.transform.rowcol(profile['transform'],pt.x,pt.y)
    return int(row),int(col)
cases=[{'name':f'Larger reference / source {large["source_index"]}','source_index':large['source_index'],'pixel':pixel_of_point(polys[large['source_index']].centroid),
        'type':'detected_reference_example'},
       {'name':'Missed reference / source 2679 / intersects CURRENT OSM road','source_index':2679,'pixel':pixel_of_point(polys[2679].centroid),
        'type':'missed_reference_not_verified_road_blockage'}]
unknown=(np_prob>=.5)&valid&road&~ref
lab,n=label(unknown)
if n:
    counts=np.bincount(lab.ravel());counts[0]=0;k=int(counts.argmax());rr,cc=np.where(lab==k)
    chosen=int(np.argmin((rr-rr.mean())**2+(cc-cc.mean())**2))
    cases.append({'name':'Positive OUTSIDE reference near road / unverified','pixel':(int(rr[chosen]),int(cc[chosen])),
                  'type':'unknown_positive_not_confirmed_false_alarm','component_pixels':int(len(rr))})
fig,axes=plt.subplots(len(cases),3,figsize=(14,4.6*len(cases)),squeeze=False)
for row,case in enumerate(cases):
    pr,pc=case['pixel'];a=max(0,min(pr-40,560));b=max(0,min(pc-40,560));sl=np.s_[a:a+80,b:b+80]
    for col,arr in enumerate([pre,post]):
        rgb=np.clip(arr[[3,2,1]][:,sl[0],sl[1]].transpose(1,2,0)/.3,0,1)
        axes[row,col].imshow(rgb)
    axes[row,2].imshow(np_prob[sl],cmap='magma',vmin=0,vmax=1)
    for col in range(3):
        if ref[sl].any() and (~ref[sl]).any():axes[row,col].contour(ref[sl],levels=[.5],colors=['yellow'],linewidths=.8)
        if 'source_index' in case:
            target=rasterize([(polys[case['source_index']],1)],out_shape=(640,640),transform=profile['transform'],dtype='uint8').astype(bool)
            axes[row,col].contour(target[sl],levels=[.5],colors=['magenta'],linewidths=1.6)
        else:
            axes[row,col].plot(pc-b,pr-a,'+',color='magenta',markersize=15,markeredgewidth=2)
        if road[sl].any() and (~road[sl]).any():axes[row,col].contour(road[sl],levels=[.5],colors=['cyan'],linewidths=.8)
        if valid[sl].any() and (~valid[sl]).any():axes[row,col].contour(valid[sl],levels=[.5],colors=['white'],linewidths=.7)
        axes[row,col].set_title(['PRE RGB','POST RGB','BBUnet probability'][col]);axes[row,col].set_axis_off()
    axes[row,0].text(0,1.13,case['name'],transform=axes[row,0].transAxes,fontsize=11)
    case['crop_grid_rows_cols']=[a,a+80,b,b+80]
fig.suptitle('Targeted diagnostics, NOT confirmed road blockage or false-alarm labels\nMagenta: target example | yellow: inventory | cyan: 100m road context | white: evaluation boundary',fontsize=12)
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'BBUnet_diagnostic_examples.png',dpi=160);plt.close(fig)
result={'pass':True,'variants':verified,'reference_mask_recreated':True,'visual_cases':cases,
        'independent_accuracy_validation':False,'external_ground_truth_checked':False}
(OUT/'independent_output_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
