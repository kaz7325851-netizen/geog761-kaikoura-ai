"""Real first-place flood checkpoint inference, not whole three-checkpoint winning pipeline."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT.parent/'cache/matplotlib')
import json
import argparse
from contextlib import nullcontext
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from load_kari import load_model

LABELS=['Background','Non-flooded building','Flooded building','Non-flooded road','Flooded road']
COLORS=['#303030','#d8ca8a','#da38ae','#42c5bc','#ec3030']
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--precision',choices=['fp32','autocast'],default='fp32')
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    model, source=load_model()
    device=torch.device('cuda')
    model=model.to(device)
    manifest=json.loads((ROOT/'test_sample_manifest.json').read_text())
    results=[]
    fig,axes=plt.subplots(3,3,figsize=(12,12))
    mean=np.asarray([.249566,.318912,.21801],dtype=np.float32)[:,None,None]
    std=np.asarray([.12903,.11784,.10739],dtype=np.float32)[:,None,None]
    cmap=ListedColormap(COLORS)
    norm=BoundaryNorm(np.arange(-.5,5.5),5)
    for i,sample in enumerate(manifest['samples']):
        sid=Path(sample['label']).stem
        arrays=np.load(ROOT/'prepared_inputs'/(sid+'.npz'),allow_pickle=False)
        pre,post=arrays['pre'],arrays['post']
        x1=torch.from_numpy((pre.astype(np.float32)/255-mean)/std).unsqueeze(0).to(device)
        x2=torch.from_numpy((post.astype(np.float32)/255-mean)/std).unsqueeze(0).to(device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start=time.perf_counter()
        precision_context = torch.autocast('cuda',dtype=torch.float16) if args.precision=='autocast' else nullcontext()
        with torch.inference_mode(),precision_context:
            outputs=model(x1,x2)
        with torch.inference_mode():
            logits=F.interpolate(outputs[5].float(),size=(1300,1300),mode='bilinear',align_corners=False)
            assert torch.isfinite(logits).all()
            probs=logits.softmax(1)
            raw=probs.argmax(1)
            classified=torch.where((raw==2)&(probs[:,2]<.9),1,raw)
            classified=torch.where((classified==4)&(probs[:,4]<.9),3,classified)
            raw_np=raw[0].cpu().numpy().astype(np.uint8)
            classified_np=classified[0].cpu().numpy().astype(np.uint8)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated()/1024**2
        data=ROOT/'data/Louisiana-West_Test_Public'
        with rasterio.open(data/'PRE-event'/sample['pre-event image']) as src:
            profile=src.profile.copy()
        profile.update(count=1,dtype='uint8',compress='deflate',nodata=None)
        profile.pop('photometric',None)
        for suffix,array in [('raw',raw_np),('prediction',classified_np)]:
            with rasterio.open(out/(sid+'_'+suffix+'.tif'),'w',**profile) as dst:
                dst.write(array,1)
                dst.update_tags(description='Model prediction, NOT ground truth',classes=json.dumps(LABELS))
        with rasterio.open(out/(sid+'_prediction.tif')) as check:
            reread=check.read(1)
            assert np.array_equal(reread,classified_np)
            assert check.transform==profile['transform'] and check.crs==profile['crs']
        row={'sample':sid,'inference_seconds':elapsed,'peak_allocated_mb':peak,
             'post_all_rgb_zero_pixels':int(np.all(post==0,axis=0).sum()),
             'post_all_rgb_zero_fraction':float(np.all(post==0,axis=0).mean()),
             'class_counts':np.bincount(classified_np.ravel(),minlength=5).tolist(),
             'raw_class_counts':np.bincount(raw_np.ravel(),minlength=5).tolist(),
             'threshold_changed_pixels':int((raw_np!=classified_np).sum()),
             'saved_raster_independent_readback_passed':True}
        results.append(row)
        print(json.dumps(row),flush=True)
        for j,(image_,title) in enumerate([(pre.transpose(1,2,0),'Pre-event RGB'),
                (post.transpose(1,2,0),'Post-event RGB (warped)'),(classified_np,'Prediction (threshold 0.9)')]):
            if j==2: axes[i,j].imshow(image_,cmap=cmap,norm=norm,interpolation='nearest')
            else: axes[i,j].imshow(image_)
            axes[i,j].set_title(sid+' | '+title,fontsize=10)
            axes[i,j].axis('off')
        del outputs,logits,probs,raw,classified,x1,x2
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(COLORS,LABELS)],loc='lower center',ncol=3)
    fig.suptitle('SpaceNet8 KARI best_flood checkpoint | 3 fixed public test tiles\nPredictions only: no reference labels or accuracy scores',fontsize=13)
    fig.tight_layout(rect=(0,.065,1,.95))
    fig.savefig(out/'predictions.png',dpi=160)
    plt.close(fig)
    report={'status':'Real pretrained checkpoint inference completed; accuracy not evaluated',
            'source':source,'samples':results,'class_order':LABELS,
            'input_preprocessing':'Exact author GDAL post warp; RGB /255 then author mean/std; no input crop or downsize',
            'precision':args.precision,
            'flood_threshold':.9,'augmentation':False,'training_performed':False,
            'scope':'Single best_flood.pt; NOT best_road/best_building or vectorisation/submission pipeline',
            'label_status':'Requested 3 public test GeoJSON annotation URLs returned 404; no scoring reference acquired',
            'test_selection':manifest['selection'],'model_exposure':'Not independently audited',
            'versions':{'torch':torch.__version__,'numpy':np.__version__,'rasterio':rasterio.__version__,
                        'gpu':torch.cuda.get_device_name()},
            'timing_scope':'Model forward, output resampling/classification and CPU transfer; excludes data read, model load, maps/saves; first tile includes cold kernels',
            'limitations':['Flood road/building labels, not landslide or earthquake road blockage',
                           'Sub-metre optical RGB, not 10m Sentinel1 SAR or Sentinel2 multispectral',
                           '3 tiles are a pilot, not independent performance evaluation',
                           'Class 0 background / nonflooded class does not prove road passability'],
            'local_compatibility_edit':'seg_hrnet_ocr.py np.int -> int; no parameter or architecture alteration'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('RESULT_DIR='+str(out),flush=True)

if __name__=='__main__':main()
