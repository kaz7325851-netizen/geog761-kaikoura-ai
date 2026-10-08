"""Re-read saved controlled outputs and make a fixed-scale scientific comparison."""
from pathlib import Path
import os,json,hashlib
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import numpy as np,rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.transform import Affine
OUT=ROOT/'preprocessing_audit';NEW=OUT/'terrain_aligned_pilot'
old=json.loads((ROOT/'pilot_exploration/exploratory_results.json').read_text())
new=json.loads((NEW/'controlled_results.json').read_text())
checks=[]
for r in new['records']:
    p=Path(r['file']);z=np.load(p);baseline=np.load(r['baseline_input_file'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    assert all(np.array_equal(z[k],baseline[k]) for k in z.files if k!='topo')
    assert r['unchanged_except_topo']
    prob=Path(r['probability_file']);assert hashlib.sha256(prob.read_bytes()).hexdigest()==r['probability_sha256']
    raw=np.load(NEW/(r['tile_id']+'_raw_context_probability.npy'))
    mask=z['valid'].copy();central=np.zeros((224,224),bool);central[56:168,56:168]=True;mask&=central
    with rasterio.open(prob) as ds:
        a=ds.read(1)
        assert ds.crs.to_epsg()==2193 and ds.transform==Affine(*r['grid']) and ds.nodata==-9999
        assert np.all(a[~mask]==-9999) and np.array_equal(a[mask],raw[mask])
        assert np.isfinite(a[mask]).all() and np.all((a[mask]>=0)&(a[mask]<=1))
    target=z['target_inventory']&mask;coverage=int(np.sum(target&(raw>=.5)))
    assert coverage==r['target_covered_pixels']
    checks.append({'tile_id':r['tile_id'],'input_and_output_hashes_verified':True,'only_terrain_changed':True,
        'grid_range_and_border_verified':True,'coverage_recount_verified':True})
fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
titles=['Previous terrain / RGB 0-3000','Horn terrain / RGB 0-3000','Previous terrain / RGB 0-4000','Horn terrain / RGB 0-4000']
for i,sample in enumerate(['pilot_01','pilot_02']):
    for j,(where,upper) in enumerate([(ROOT/'pilot_exploration',3000),(NEW,3000),(ROOT/'pilot_exploration',4000),(NEW,4000)]):
        tid=f'{sample}_rgb0_{upper}';z=np.load(where/(tid+'.npz'))
        p=np.load(where/(tid+'_raw_context_probability.npy'))[56:168,56:168]
        ax=axes[i,j];im=ax.imshow(p,cmap='magma',vmin=0,vmax=1,extent=[0,672,672,0])
        for k,color in [('all_inventory','cyan'),('target_inventory','lime')]:
            a=z[k][56:168,56:168]
            if a.any():ax.contour(a,levels=[.5],colors=[color],linewidths=.8,extent=[0,672,672,0])
        rec=next(r for r in (old if where==ROOT/'pilot_exploration' else new)['records'] if r['tile_id']==tid)
        value=rec['target_inventory_overlap_fraction_not_validated_recall']
        ax.set_title(f'{titles[j]}\nReference target covered: {100*value:.1f}%',fontsize=10)
        ax.set_xlabel('metres');ax.set_ylabel(('Larger target\n' if i==0 else 'Smaller target\n')+'metres' if j==0 else 'metres')
fig.suptitle('Controlled terrain comparison: same dates, RGB arrays, masks, model and threshold\nCentral 112 x 112 output only; green = target inventory, cyan = other inventory; NOT road passability',fontsize=12)
fig.colorbar(im,ax=axes,shrink=.75,label='Uncalibrated model landslide probability')
plot=OUT/'terrain_controlled_comparison.png';fig.savefig(plot,dpi=150);plt.close(fig)
summary={'checks':checks,'all_checks_passed':True,'plot_file':str(plot),
    'weight_unchanged':hashlib.sha256((ROOT/'weights/M_ALL_006.hdf5').read_bytes()).hexdigest()==new['checkpoint_sha256'],
    'full_author_preprocessing_reproduced':False,'formal_accuracy_claim':False,'road_passability_claim':False}
(OUT/'saved_output_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
