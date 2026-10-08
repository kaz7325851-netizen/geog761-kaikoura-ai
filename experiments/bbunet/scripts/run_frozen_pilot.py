"""Exploratory real BBUnet inference; fixed region, dates, weights and threshold.

Scores inventory overlap only, not verified accuracy or road passability.
Includes predeclared GEE-terrain sensitivity using otherwise identical inputs.
"""
import os,sys,json,hashlib,time,importlib.util
from argparse import Namespace
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'/'matplotlib'))
sys.path.insert(0,str(ROOT/'runtime_smp033'))
import numpy as np
import torch
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as transform_geom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT=ROOT/'model_inputs'/'20261007T203029Z'
DESIGN=ROOT.parent/'pilot_design'/'20261005T200713Z'
STAGE=ROOT/'prepared_inputs'/'20261007T193918Z'
OUT=ROOT/'pilot_results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True,exist_ok=False)
inp=json.loads((INPUT/'model_input_manifest.json').read_text(encoding='utf-8'))
for record in inp['files']:
    assert hashlib.sha256(Path(record['file']).read_bytes()).hexdigest()==record['sha256']
wp=ROOT/'weights'/'bbunet_r18.pt'
weight_audit=json.loads((ROOT/'weights'/'checkpoint_inspection.json').read_text(encoding='utf-8'))
assert hashlib.sha256(wp.read_bytes()).hexdigest()==weight_audit['sha256']
torch.set_num_threads(2)
torch.manual_seed(761)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
device='cuda' if torch.cuda.is_available() else 'cpu'
with torch.serialization.safe_globals([Namespace]):
    ckpt=torch.load(wp,map_location='cpu',weights_only=True)
hp=vars(ckpt['hparams'])
assert hp['IMAGE_SIZE']==256 and hp['STRIDE']==128 and hp['THRESHOLD']==.5
path=ROOT/'author_source_ce34d3ac'/'igarss-landslide-delineation-ce34d3ac02b80d4db7e542203b7e534649a46352'/'src'/'models'/'bbunet.py'
spec=importlib.util.spec_from_file_location('author_bbunet',path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
model=mod.BBUnet(s2_in_channels=12,dem_in_channels=4,classes=1)
model.load_state_dict(ckpt['state_dict'],strict=True);model.eval();model.to(device)

def read(fn):
    with rasterio.open(fn) as ds:
        return ds.read(),ds.profile.copy()
pre,profile=read(INPUT/'pre_optical_12channels_normalized.tif')
post,_=read(INPUT/'post_optical_12channels_normalized.tif')
valid=read(INPUT/'evaluation_conservative_mask.tif')[0][0].astype(bool)
road=read(STAGE/'road_100m_context_mask.tif')[0][0].astype(bool)
project=Transformer.from_crs(4326,2193,always_xy=True).transform
features=json.loads((DESIGN/'reference_inventory.geojson').read_text())['features']
polys=[transform_geom(project,shape(f['geometry'])) for f in features]
ref=rasterize([(p,1) for p in polys],out_shape=(640,640),transform=profile['transform'],dtype='uint8').astype(bool)
threshold=float(hp['THRESHOLD'])
positions=[(r,c) for r in range(0,385,128) for c in range(0,385,128)]
report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running','device':device,
        'pilot_id':'SH1-OPTICAL-01','input_manifest':str(INPUT/'model_input_manifest.json'),
        'checkpoint_sha256':weight_audit['sha256'],'smp_version':'0.3.3',
        'model_strict_loading':True,'real_imagery_BBUnet_inference':True,
        'training':False,'fine_tuning':False,'threshold':threshold,
        'evaluation_rule':'joint source valid, SENCLOUD12 clear both dates, SCL4/5/7 both dates',
        'patch_size':256,'stride':128,'patches_per_variant':len(positions),
        'aggregation':'unweighted average sigmoid probabilities across overlapping tiles',
        'object_diagnostic_thresholds_predeclared':[.1,.5],
        'independent_accuracy_test':False,'road_passability_inferred':False,
        'full_author_preprocessing_reproduced':False,'author_training_overlap_verified':False,
        'metrics_meaning':'Conditional overlap with reference landslide inventory; outside inventory is not confirmed negative',
        'variants':{},'limitations':inp['limitations']+[
            'Fixed inventory-guided single pilot, not random or representative corridor holdout',
            'No native RichDEM binary run; audited NumPy formulas on NZTM-resampled DSM',
            'Object hit uses any positive center raster pixel; size and quality exclusions affect this diagnostic',
            'Mean overlapping probability mosaic is a local inference policy, not documented author full-scene policy',
            'Current OSM geometry not verified 2016; no true road blockage/safe labels']}

def save(fn,arr,description,nodata):
    p=profile.copy();p.update(count=1,dtype=str(arr.dtype),nodata=nodata,compress='deflate',photometric='minisblack')
    dest=OUT/fn
    with rasterio.open(dest,'w',**p) as d:
        d.write(arr,1);d.set_band_description(1,description)
        d.update_tags(meaning='Exploratory landslide prediction, NOT road passability')
    with rasterio.open(dest) as d:
        assert np.array_equal(d.read(1),arr) and d.crs==profile['crs'] and d.transform==profile['transform']
    return {'file':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}

def stats(prob):
    pos=(prob>=threshold)&valid
    out={'evaluation_pixels':int(valid.sum()),'reference_union_evaluated_pixels':int((ref&valid).sum()),
         'predicted_positive_evaluated_pixels':int(pos.sum()),
         'positive_within_reference_pixels':int((pos&ref).sum()),
         'positive_outside_reference_NOT_confirmed_FP_pixels':int((pos&~ref).sum()),
         'reference_pixel_coverage_NOT_validated_recall':float(np.mean(prob[ref&valid]>=threshold)) if (ref&valid).any() else None,
         'evaluated_probability_percentiles':np.percentile(prob[valid],[0,25,50,75,95,99,100]).tolist(),
         'road100m':{'evaluation_pixels':int((valid&road).sum()),'reference_pixels':int((ref&valid&road).sum()),
                     'positive_pixels':int((pos&road).sum()),'positive_within_reference_pixels':int((pos&ref&road).sum()),
                     'reference_pixel_coverage_NOT_validated_recall':float(np.mean(prob[ref&valid&road]>=threshold)) if (ref&valid&road).any() else None}}
    per=[]
    for f,p in zip(features,polys):
        pm=rasterize([(p,1)],out_shape=(640,640),transform=profile['transform'],dtype='uint8').astype(bool)
        n=int(pm.sum());vm=pm&valid;vn=int(vm.sum());pn=int((vm&pos).sum())
        area=f['properties']['source_area_m2']
        per.append({**f['properties'],'size_group':'large_ge10000' if area>=10000 else 'small_100_to10000' if area>=100 else 'very_small_lt100',
                    'rasterized_pixels':n,'evaluated_pixels':vn,'retained_fraction':vn/n if n else None,
                    'positive_pixels':pn,'coverage_fraction':pn/vn if vn else None,
                    'probability_max':float(prob[vm].max()) if vn else None})
    out['objects']={}
    for group in ['all','near_road100m','large_near_road100m','small_near_road100m']:
        selected=[x for x in per if (group=='all' or x['distance_to_sh1_m']<=100) and
                  (group!='large_near_road100m' or x['size_group']=='large_ge10000') and
                  (group!='small_near_road100m' or x['size_group']=='small_100_to10000')]
        evaluated=[x for x in selected if x['evaluated_pixels']>0]
        out['objects'][group]={'source_polygons':len(selected),'polygons_with_evaluated_pixels':len(evaluated),
             'polygons_without_evaluated_pixels':len(selected)-len(evaluated),
             'polygons_with_any_positive':sum(x['positive_pixels']>0 for x in evaluated),
             'polygons_ge10pct_coverage':sum(x['coverage_fraction']>=.1 for x in evaluated),
             'polygons_ge50pct_coverage':sum(x['coverage_fraction']>=.5 for x in evaluated)}
    return out,per

predictions={}
try:
    for variant,filename in [('Horn_numpy','terrain_4channels_author_encoding.tif'),('GEE_sensitivity','terrain_4channels_GEE_sensitivity.tif')]:
        terrain=read(INPUT/filename)[0]
        summed=np.zeros((640,640),dtype='float64');squared=summed.copy();count=np.zeros((640,640),dtype='uint8')
        start=time.monotonic();repeat=None
        for j,(r,c) in enumerate(positions):
            arrays=[a[:,r:r+256,c:c+256][None] for a in [pre,post,terrain]]
            ts=[torch.from_numpy(a).to(device) for a in arrays]
            assert all(torch.isfinite(t).all() for t in ts)
            with torch.inference_mode():
                logits=model(*ts)
                p=logits.sigmoid()[0,0].cpu().numpy()
                if j==0:
                    again=model(*ts).sigmoid()[0,0].cpu().numpy()
                    repeat={'patch':[r,c],'max_abs_probability_difference':float(np.max(np.abs(p-again))),
                            'allclose_at_1e6':bool(np.allclose(p,again,rtol=0,atol=1e-6))}
            assert np.isfinite(p).all() and 0<=p.min()<=p.max()<=1
            summed[r:r+256,c:c+256]+=p;squared[r:r+256,c:c+256]+=p.astype('float64')**2;count[r:r+256,c:c+256]+=1
        assert (count>0).all()
        prob=(summed/count).astype('float32')
        std=np.sqrt(np.maximum(squared/count-(summed/count)**2,0)).astype('float32')
        predictions[variant]=prob
        metrics,per=stats(prob)
        outputs=[save(f'{variant}_probability_full_context.tif',prob,'exploratory_landslide_probability_UNSCREENED',None),
                 save(f'{variant}_probability_screened.tif',np.where(valid,prob,-9999).astype('float32'),'exploratory_landslide_probability_screened',-9999),
                 save(f'{variant}_positive_screened.tif',np.where(valid,prob>=threshold,255).astype('uint8'),'exploratory_landslide_positive_255excluded',255),
                 save(f'{variant}_overlap_context_std.tif',std,'overlapping_patch_probability_std_NOT_calibrated_uncertainty',None)]
        (OUT/f'{variant}_inventory_objects.json').write_text(json.dumps(per,indent=2),encoding='utf-8')
        report['variants'][variant]={'seconds':time.monotonic()-start,'metrics':metrics,'outputs':outputs,'repeatability':repeat,
             'overlap_context_std_road100m_percentiles':np.percentile(std[valid&road],[50,95,100]).tolist()}
        print(json.dumps({'variant':variant,'seconds':report['variants'][variant]['seconds'],'metrics':metrics},indent=2),flush=True)
    a,b=predictions.values()
    report['terrain_sensitivity']={'probability_MAE_evaluated':float(np.mean(np.abs(a[valid]-b[valid]))),
         'binary_disagreement_fraction_evaluated':float(np.mean((a[valid]>=threshold)!=(b[valid]>=threshold))),
         'probability_MAE_road100m':float(np.mean(np.abs(a[valid&road]-b[valid&road]))),
         'binary_disagreement_fraction_road100m':float(np.mean((a[valid&road]>=threshold)!=(b[valid&road]>=threshold))),
         'interpretation':'Two terrain-input variants, not an independent accuracy comparison'}
    save('reference_inventory_center_raster.tif',ref.astype('uint8'),'inventory_reference_NOT_road_status',None)
    save('evaluation_mask.tif',valid.astype('uint8'),'conservative_evaluation_mask',None)
    save('patch_context_count.tif',count,'overlapping_patch_count',None)
    fig,axes=plt.subplots(2,2,figsize=(13,12))
    rgb=post[[3,2,1]].transpose(1,2,0);rgb=np.clip(rgb/.3,0,1)
    axes[0,0].imshow(rgb);axes[0,0].contour(ref,levels=[.5],colors=['yellow'],linewidths=.3)
    axes[0,0].contour(road,levels=[.5],colors=['cyan'],linewidths=.4)
    axes[0,0].set_title('POST RGB | yellow: reference; cyan: road context')
    for ax,(name,p) in zip([axes[0,1],axes[1,0]],predictions.items()):
        ax.imshow(rgb)
        v=np.ma.masked_where(~valid,p)
        im=ax.imshow(v,cmap='magma',vmin=0,vmax=1,alpha=.9)
        ax.contour(ref,levels=[.5],colors=['cyan'],linewidths=.3)
        ax.set_title(f'{name} | screened probability; cyan: reference')
        fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
    axes[1,1].imshow(rgb)
    positive=(a>=threshold)&valid
    overlay=np.zeros((640,640,4));overlay[positive]=[1,0,0,.85]
    axes[1,1].imshow(overlay);axes[1,1].contour(ref,levels=[.5],colors=['yellow'],linewidths=.4)
    axes[1,1].contour(road,levels=[.5],colors=['cyan'],linewidths=.3)
    axes[1,1].set_title('Horn_numpy >=0.5 red | NOT road blockage')
    for ax in axes.flat:ax.set_axis_off()
    fig.suptitle('BBUnet real frozen-pilot exploratory inference | NO training, NOT road passability\nExcluded areas remain RGB, not negative predictions; inventory overlap is not independent accuracy',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'BBUnet_pilot_overview.png',dpi=160);plt.close(fig)
    report['status']='exploratory_pilot_complete'
finally:
    (OUT/'pilot_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'OUTPUT {OUT}',flush=True)
