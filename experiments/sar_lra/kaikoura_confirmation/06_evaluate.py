"""Run frozen same-checkpoint diagnostics; no training, no threshold fitting."""
from pathlib import Path
from datetime import datetime
import os,sys,json,csv,hashlib,time
ROOT=Path(__file__).resolve().parent;SAR=ROOT.parent
os.environ['MPLCONFIGDIR']=str(SAR.parent/'cache/matplotlib')
sys.path.insert(0,str(SAR))
from check_weights import load_model,tf
from utils import non_max_suppression_fast
import h5py,numpy as np,rasterio,pyproj
from rasterio.features import rasterize
from rasterio.warp import reproject,Resampling,transform_geom
from shapely.geometry import shape,box,mapping
from shapely.ops import transform,unary_union
import shapefile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

design=json.loads((ROOT/'frozen_design.json').read_text(encoding='utf-8'))
manifest=json.loads((ROOT/'latest_download.json').read_text(encoding='utf-8'))
assert hashlib.sha256((ROOT/'frozen_design.json').read_bytes()).hexdigest()==manifest['frozen_design_sha256']
tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
model,details=load_model('ASCENDING')
checkpoint=SAR/'official/SAR-LRA Tool V1/model_weights'/details['checkpoint']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==design['checkpoint_sha256']
OUT=ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S');OUT.mkdir(parents=True,exist_ok=False)

def predict(patches):
    return np.concatenate([model(patches[i:i+32],training=False).numpy().reshape(-1) for i in range(0,len(patches),32)])
def quant(values):return np.quantile(values,[0,.25,.5,.75,1]).tolist() if len(values) else None
def metric(pos,bg,pred,selected):
    p=pos&selected;b=bg&selected
    tp=int((p&pred).sum());fn=int((p&~pred).sum());fp=int((b&pred).sum());tn=int((b&~pred).sum())
    div=lambda a,d:float(a/d) if d else None
    return {'positive_windows':int(p.sum()),'reference_background_windows':int(b.sum()),'TP':tp,'FN':fn,'FP':fp,'TN':tn,
     'recall':div(tp,tp+fn),'background_false_alarm_rate':div(fp,fp+tn),
     'curated_precision_not_deployment_precision':div(tp,tp+fp),'curated_F1':div(2*tp,2*tp+fp+fn)}

# Re-run fixed old samples to rule out a changed loader/runtime causing new failure.
replay=[]
for name,input_path,pred_path,score_key in [
 ('Sumatra_ASC',SAR/'data/ASCENDING.tif',SAR/'results/20261005_160317/ASCENDING/patch_predictions.npz','probabilities'),
 ('Kaikoura_old_NZTM',Path(json.loads((SAR/'kaikoura_exploration/latest_input.json').read_text())['input_file']),SAR/'kaikoura_exploration/results/20261005_182848/patch_predictions.npz','scores')]:
    saved=np.load(pred_path)
    chosen=np.unique(np.concatenate([np.linspace(0,len(saved['coords'])-1,8,dtype=int),np.flatnonzero(saved['truth'])[:10]]))
    with rasterio.open(input_path) as src:arr=src.read([1,2,3,4]).transpose(1,2,0)
    patches=np.stack([arr[y:y+64,x:x+64] for x,y in saved['coords'][chosen]])
    actual=predict(patches);expected=saved[score_key][chosen]
    error=float(np.max(np.abs(actual-expected)));assert error<1e-6
    replay.append({'dataset':name,'sample_count':len(chosen),'max_abs_score_error':error,'unchanged_classifications':bool(np.array_equal(actual>.6,expected>.6))})

summaries=[];plotdata=[]
for record in manifest['records']:
    sid=record['id'];path=Path(record['file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    dest=OUT/sid;dest.mkdir()
    with rasterio.open(path) as src:
        data=src.read();profile=src.profile.copy();aff=src.transform;crs=src.crs;hw=src.shape
    valid=data[4]==1
    assert np.allclose(data[2,valid],data[0,valid]-data[5,valid],atol=2e-5,rtol=1e-5)
    assert np.allclose(data[3,valid],data[1,valid]-data[6,valid],atol=2e-5,rtol=1e-5)
    visibility_checks=[]
    if sid!='OLD_GEO':
        masks=[];bads=[]
        for i,heading in enumerate(record['metadata']['headings_deg']):
            angle,alpha,cover=data[10+3*i:13+3*i]
            known=(cover==1)&(angle!=-9999)&(alpha!=-9999)
            expected=np.degrees(np.arctan(np.tan(np.radians(data[8]))*np.cos(np.radians(heading-data[9]))))
            error=float(np.max(np.abs(expected[known]-alpha[known]))) if known.any() else None
            assert error is None or error<.001
            bad=known&((alpha>=angle)|(alpha<=angle-90))
            masks.append(known);bads.append(bad)
            visibility_checks.append({'scene':i,'known_pixels':int(known.sum()),'bad_pixels':int(bad.sum()),'alpha_formula_error_deg':error})
        visible=valid&masks[0]&(masks[1]|masks[2])&~(bads[0]|bads[1]|bads[2])
        known=valid&masks[0]&(masks[1]|masks[2])
        land=(data[7]>1)&(data[7]!=-9999)
        labels=[]
        for name in ['raw_reference_mask','inventory_buffer200_mask','road500_mask']:
            with rasterio.open(ROOT/'reference'/sid/(name+'.tif')) as src:
                assert src.shape==hw and src.crs==crs and src.transform==aff
                labels.append(src.read(1).astype(bool))
        raw,near_inventory,road=labels
    else:
        # Re-rasterize original inventory vectors; do not resample old reference raster.
        shp=SAR/'kaikoura_exploration/reference/Kaikora_2016.shp'
        invcrs=pyproj.CRS.from_wkt(shp.with_suffix('.prj').read_text())
        inv_to_wgs=pyproj.Transformer.from_crs(invcrs,4326,always_xy=True).transform
        extent=box(*rasterio.transform.array_bounds(*hw,aff));local=[];projected=[]
        inv_to_metric=pyproj.Transformer.from_crs(invcrs,2193,always_xy=True).transform
        for s in shapefile.Reader(str(shp)).shapes():
            geom=shape(s.__geo_interface__)
            if not geom.is_valid:geom=geom.buffer(0)
            wgs=transform(inv_to_wgs,geom)
            if wgs.intersects(extent):local.append(wgs)
            # Include neighbouring labels outside the rectangular export as well.
            # 0.003 degrees exceeds200m in both axes here; actual buffer is metric.
            if wgs.intersects(extent.buffer(.003)):projected.append(transform(inv_to_metric,geom))
        raw=rasterize([(g,1) for g in local],out_shape=hw,transform=aff,fill=0,dtype='uint8').astype(bool)
        metric_to_wgs=pyproj.Transformer.from_crs(2193,4326,always_xy=True).transform
        near_inventory=rasterize([(transform(metric_to_wgs,unary_union(projected).buffer(200)),1)],out_shape=hw,transform=aff,fill=0,dtype='uint8').astype(bool)
        fwd=pyproj.Transformer.from_crs(4326,2193,always_xy=True).transform
        roads=unary_union([transform(fwd,shape(f['geometry'])) for f in json.loads((SAR.parent/'sentinel1/study_route_lines.geojson').read_text())['features']])
        road=rasterize([(transform(metric_to_wgs,roads.buffer(500)),1)],out_shape=hw,transform=aff,fill=0,dtype='uint8').astype(bool)
        labeldir=SAR/'kaikoura_exploration/label_cleaning'
        oldclean=Path(json.loads((labeldir/'latest_cleaning.json').read_text())['result_directory'])
        def warp_old(oldpath,band):
            result=np.zeros(hw,dtype='float32')
            with rasterio.open(oldpath) as src:
                reproject(rasterio.band(src,band),result,src_transform=src.transform,src_crs=src.crs,
                          dst_transform=aff,dst_crs=crs,resampling=Resampling.nearest,dst_nodata=0)
            return result
        visible=(warp_old(oldclean/'visible_mask.tif',1)==1)&valid
        known=(warp_old(oldclean/'unknown_visibility_mask.tif',1)==0)&valid
        demfile=Path(json.loads((labeldir/'download_manifest.json').read_text())['file'])
        land=warp_old(demfile,1)>1
    clean=raw&visible
    coords=[];patches=[];rows=[]
    candidates=0
    image=data[:4].transpose(1,2,0).astype('float32')
    for y in range(0,hw[0]-64,32):
        for x in range(0,hw[1]-64,32):
            candidates+=1;sl=np.s_[y:y+64,x:x+64]
            if not valid[sl].all():continue
            coords.append((x,y));patches.append(image[sl])
            rawf=float(raw[sl].mean());cleanf=float(clean[sl].mean());visf=float(visible[sl].mean());landf=float(land[sl].mean())
            bg=rawf==0 and not near_inventory[sl].any() and visf>=.95 and landf>=.95
            rows.append({'x':x,'y':y,'raw_fraction':rawf,'clean_fraction':cleanf,'visible_fraction':visf,'land_fraction':landf,
                         'reference_background':bg,'road_near':bool(road[sl].any()),'nonoverlap_grid':x%64==0 and y%64==0})
    assert patches,f'No scored patches in {sid}'
    patches=np.stack(patches);coords=np.asarray(coords,dtype='int32')
    t=time.perf_counter();scores=predict(patches);seconds=time.perf_counter()-t
    assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    repeat=predict(patches[:2]);assert np.allclose(repeat,scores[:2],atol=1e-6)
    predicted=scores>.6
    fractions=np.asarray([r['clean_fraction'] for r in rows]);rawfractions=np.asarray([r['raw_fraction'] for r in rows]);positive=fractions>.05
    bg=np.asarray([r['reference_background'] for r in rows]);nearroad=np.asarray([r['road_near'] for r in rows]);nonoverlap=np.asarray([r['nonoverlap_grid'] for r in rows])
    allsel=np.ones(len(rows),dtype=bool)
    groups={'all':allsel,'nonoverlap':nonoverlap,'marginal5to15':positive&(fractions<=.15),
            'dense_gt15':fractions>.15,'road_near':nearroad,'road_near_marginal':nearroad&positive&(fractions<=.15)}
    metrics={name:metric(positive,bg,predicted,selection) for name,selection in groups.items()}
    for i,row in enumerate(rows):
        row.update(score=float(scores[i]),positive=bool(positive[i]),predicted=bool(predicted[i]),
                   evaluation_class='positive' if positive[i] else 'reference_background' if bg[i] else 'excluded_ambiguous')
    with (dest/'windows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(dest/'predictions.npz',coords=coords,scores=scores,raw_fraction=rawfractions,clean_fraction=fractions,
                        positive=positive,background=bg,predicted=predicted,road_near=nearroad,nonoverlap=nonoverlap)
    for name,arr in [('raw_reference',raw),('clean_reference',clean),('visible_mask',visible),('land_proxy',land),('near_inventory200',near_inventory),('road500',road)]:
        profile.update(count=1,dtype='uint8',nodata=None,compress='deflate')
        with rasterio.open(dest/(name+'.tif'),'w',**profile) as dst:dst.write(arr.astype('uint8'),1)
    boxes=np.asarray([(x,y,x+64,y+64) for (x,y),p in zip(coords,predicted) if p],dtype='int32').reshape(-1,4)
    kept=np.asarray(non_max_suppression_fast(boxes,.1),dtype='int32').reshape(-1,4)
    features=[]
    for x1,y1,x2,y2 in kept:
        ring=[aff*(int(x),int(y)) for x,y in [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]]
        geom=transform_geom(crs,4326,{'type':'Polygon','coordinates':[ring]})
        idx=np.flatnonzero((coords[:,0]==x1)&(coords[:,1]==y1))[0]
        features.append({'type':'Feature','properties':{'site':sid,'score':float(scores[idx]),'meaning':'coarse suspect slide window, NOT actual road blockage'},'geometry':geom})
    (dest/'detections.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}),encoding='utf-8')
    summary={'id':sid,'input_sha256':record['sha256'],'grid_crs':crs.to_string(),'shape':list(hw),'candidate_windows':candidates,
             'valid_scored_windows':len(rows),'excluded_acquisition_windows':candidates-len(rows),'metrics':metrics,
             'ambiguous_windows_excluded_from_binary_metrics':int((~positive&~bg).sum()),'predicted_positive_all_scored':int(predicted.sum()),
             'predicted_positive_ambiguous':int((predicted&~positive&~bg).sum()),'score_quantiles':quant(scores),
             'positive_score_quantiles':quant(scores[positive]),'background_score_quantiles':quant(scores[bg]),
             'positive_slide_fraction_quantiles':quant(fractions[positive]),'raw_slide_pixels':int(raw.sum()),'clean_slide_pixels':int(clean.sum()),
             'visibility_checks':visibility_checks,'inference_seconds':seconds,'NMS_retained_boxes':len(kept),
             'background_not_independently_confirmed':True,'all_counts_are_windows_not_distinct_slides':True}
    (dest/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');summaries.append(summary)
    print(json.dumps({'site':sid,'metrics':metrics,'scores_positive':summary['positive_score_quantiles'],'predicted_all':int(predicted.sum())},indent=2),flush=True)
    plotdata.append((sid,data,raw,clean,coords,scores,positive,bg,predicted))

with h5py.File(checkpoint,'r') as h:
    weights=[np.asarray(h[name][w]) for name in h.attrs['layer_names'] for w in h[name].attrs['weight_names']]
assert all(np.array_equal(a,b) for a,b in zip(weights,model.get_weights()))
assert hashlib.sha256((ROOT/'frozen_design.json').read_bytes()).hexdigest()==manifest['frozen_design_sha256']
report={'created':datetime.now().isoformat(),'frozen_design_sha256':manifest['frozen_design_sha256'],'design':design,'model':details,
        'checkpoint_sha256':design['checkpoint_sha256'],'model_weights_unchanged':True,'replay_controls':replay,'sites':summaries,
        'classification_threshold':.6,'positive_rule':'visible slide pixels /4096 >.05','training':False,'threshold_tuned':False,
        'background_false_alarm_rule':'raw inventory zero,200m away,>=95%land and active-geometry-visible; reference background only',
        'limits':['82day single-date baseline unchanged','Label-stratified sites, NOT whole-event random accuracy',
                  'Background is inventory-derived, not field-verified safe','0m-buffer active visibility not exact author original cleaned labels',
                  '64x64 coarse classification not road status; road500 intersects is proximity only','Geographic control changes physical footprint/resampling, not perfect paired-size ablation']}
(OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'latest_result.json').write_text(json.dumps({'result_directory':str(OUT)},indent=2),encoding='utf-8')
fig,axes=plt.subplots(4,3,figsize=(13,15),layout='constrained')
for row,(sid,data,raw,clean,coords,scores,pos,bg,prd) in enumerate(plotdata):
    im=axes[row,0].imshow(np.where(data[4]==1,data[2],np.nan),cmap='RdBu_r',vmin=-6,vmax=6)
    axes[row,0].set_title(sid+': observed dVV (dB)');fig.colorbar(im,ax=axes[row,0],shrink=.6)
    axes[row,1].imshow(clean,cmap=ListedColormap(['#eeeeee','#d49426']),vmin=0,vmax=1)
    axes[row,1].set_title(sid+': geometry-filtered inventory')
    comparison=np.full((int(np.ceil((data.shape[1]-64)/32)),int(np.ceil((data.shape[2]-64)/32))),np.nan)
    for (x,y),p,b,q in zip(coords,pos,bg,prd):comparison[y//32,x//32]=3 if p and q else 2 if p else 1 if b and q else 0 if b else 4
    axes[row,2].imshow(comparison,cmap=ListedColormap(['#bbbbbb','#e9654c','#456bb1','#29a27d','#f1e4c1']),vmin=-.5,vmax=4.5,interpolation='nearest')
    m=summaries[row]['metrics']['all'];axes[row,2].set_title(f"{sid}: TP{m['TP']} FN{m['FN']} FP{m['FP']} TN{m['TN']}")
for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Frozen Kaikoura diagnostics | same real ASC checkpoint, score>0.6\nGreen TP; blue FN; red reference-background FP; grey reference-background TN; cream ambiguous; white unscored',fontsize=12)
fig.savefig(OUT/'comparison.png',dpi=140);plt.close(fig)
print('RESULT_DIRECTORY='+str(OUT),flush=True)
