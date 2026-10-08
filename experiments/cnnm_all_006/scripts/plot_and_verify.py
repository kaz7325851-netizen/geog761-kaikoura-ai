"""Saved-output QA and compact scientific comparison figures."""
from pathlib import Path
import json,hashlib,os
plot_cache=Path(__file__).resolve().parent/'cache/matplotlib'
plot_cache.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(plot_cache)
import numpy as np,rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
PILOT=ROOT/'pilot_exploration';AUTHOR=ROOT/'author_smoke'
results=json.loads((PILOT/'exploratory_results.json').read_text())
central=np.s_[56:168,56:168]
fig,axes=plt.subplots(3,5,figsize=(15,9))
labels=['Large reference landslide','Small reference landslide','Inventory-sparse road segment']
for row,sample in enumerate(['pilot_01','pilot_02','pilot_03']):
    records=[r for r in results['records'] if r['sample_id']==sample];byrange={r['tile_id'].split('_')[-1]:r for r in records}
    a,b=[np.load(byrange[k]['file']) for k in ['3000','4000']]
    prob=[]
    for record in records:
        assert hashlib.sha256(Path(record['file']).read_bytes()).hexdigest()==record['sha256']
        assert hashlib.sha256(Path(record['probability_file']).read_bytes()).hexdigest()==record['probability_sha256']
        with rasterio.open(record['probability_file']) as src:
            assert src.count==1 and src.crs.to_epsg()==2193 and src.nodata==-9999
            p=src.read(1);assert np.all(p[:56]==-9999) and np.all(p[:,168:]==-9999)
            assert np.isfinite(p[central]).all() and np.all((p[central]>=0)&(p[central]<=1))
        prob.append(p[central])
    images=[a['pre_rgb_uint8'][central],a['post_rgb_uint8'][central],prob[0],b['post_rgb_uint8'][central],prob[1]]
    for col,(ax,img) in enumerate(zip(axes[row],images)):
        if col in [2,4]:
            im=ax.imshow(img,vmin=0,vmax=1,cmap='inferno',extent=[0,672,0,672])
            target_overlap=records[0 if col==2 else 1]['target_inventory_overlap_fraction_not_validated_recall']
            ax.set_title('P(slide)' if target_overlap is None else f'P(slide); target overlap {target_overlap:.1%}',fontsize=10)
        else:
            ax.imshow(img,extent=[0,672,0,672]);ax.set_title(['Pre RGB [0,3000]','Post RGB [0,3000]','','Post RGB [0,4000]',''][col],fontsize=10)
        ref=a['all_inventory'][central];target=a['target_inventory'][central];road=a['road100m'][central]
        if ref.any() and not ref.all():ax.contour(ref,levels=[.5],colors=['#36dcff'],linewidths=.7,extent=[0,672,0,672],origin='upper')
        if target.any():ax.contour(target,levels=[.5],colors=['#ff39dc'],linewidths=1.2,extent=[0,672,0,672],origin='upper')
        if road.any() and not road.all():ax.contour(road,levels=[.5],colors=['#ffdb58'],linewidths=.65,extent=[0,672,0,672],origin='upper')
        ax.set_xticks([]);ax.set_yticks([])
    axes[row,0].set_ylabel(labels[row],fontsize=10)
fig.suptitle('Kaikoura pilot: same checkpoint and locations, different fixed RGB renderings',fontsize=14)
fig.subplots_adjust(left=.06,right=.91,bottom=.13,top=.91,wspace=.06,hspace=.22)
cax=fig.add_axes([.93,.2,.012,.57]);fig.colorbar(im,cax=cax,label='Model probability (not calibrated road risk)')
fig.legend(handles=[Line2D([0],[0],color='#36dcff',label='Reference inventory'),Line2D([0],[0],color='#ff39dc',label='Selected target'),Line2D([0],[0],color='#ffdb58',label='Road 100 m buffer')],loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False)
fig.text(.5,.025,'Shown: central 112 x 112 of each 224 x 224 input (672 m). Exploratory inputs; overlap is NOT validated recall.',ha='center',fontsize=10)
fig.savefig(PILOT/'pilot_rgb_sensitivity.png',dpi=150);plt.close(fig)

smoke=json.loads((AUTHOR/'smoke_results.json').read_text())
fig,axes=plt.subplots(3,4,figsize=(11,8))
diagnostics=[]
for row,r in enumerate(smoke['records']):
    z=np.load(r['input_file']);p=np.load(AUTHOR/(r['tile_id']+'_probability.npy'))[central]
    expected=z['expected_confidence'][central]/100;valid=z['valid'][central]
    post=np.clip((z['post'][central]+1)*127.5,0,255).astype('uint8')
    for col,img in enumerate([post,expected,p,np.abs(p-expected)]):
        ax=axes[row,col]
        if col==0:ax.imshow(img)
        else:ax.imshow(img,vmin=0,vmax=.5 if col==3 else 1,cmap='inferno')
        ax.axis('off')
        if row==0:ax.set_title(['Post RGB','Author saved prediction','Current inference','Absolute difference (0-0.5)'][col],fontsize=10)
    aa=(p>=.5)&valid;bb=(expected>=.5)&valid;tp=int((aa&bb).sum());den=int(aa.sum()+bb.sum())
    diagnostics.append({'tile_id':r['tile_id'],'binary_prediction_Dice_not_accuracy':float(2*tp/den) if den else 1,
        'author_positive_pixels':int(bb.sum()),'current_positive_pixels':int(aa.sum()),'disagreeing_valid_pixels':int(((aa!=bb)&valid).sum())})
fig.suptitle('Author Jiuzhaigou sample: implementation comparison, NOT ground-truth evaluation',fontsize=12)
fig.tight_layout(rect=[0,.07,1,.94]);fig.text(.5,.025,'Only the central 112 x 112 prediction is compared. Published output is quantized to integer percentages.',ha='center',fontsize=9)
fig.savefig(AUTHOR/'author_reproduction_comparison.png',dpi=150);plt.close(fig)
qa={'saved_input_hashes_verified':True,'saved_probability_hashes_verified':True,'output_crs_nodata_ranges_verified':True,
    'output_excludes_context_border':True,'author_prediction_diagnostics':diagnostics,'independent_accuracy_validated':False,'road_passability_validated':False}
(ROOT/'output_verification.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print(json.dumps(qa,indent=2))
