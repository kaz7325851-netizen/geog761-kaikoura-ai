"""SENCLOUD12 screening of staged real L2A data, not landslide inference.

Matches inspected author's 13-band order, scaling, reflect padding and argmax.
Uses local restricted-loaded public weights, with explicit invalid-data masking.
"""
import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'/'matplotlib'))
sys.path.insert(0,str(ROOT/'runtime_smp033'))
import numpy as np
import torch
import rasterio
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

STAGE = ROOT / 'prepared_inputs' / '20261007T193918Z'
OUT = ROOT / 'cloud_screen' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2)
model = smp.Unet(encoder_name='mobilenet_v2',encoder_weights=None,classes=4,in_channels=13)
wp = ROOT/'weights'/'mobilenetv2_l2a_all.pth'
state = torch.load(wp,map_location='cpu',weights_only=True)
model.load_state_dict(state,strict=True)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
with rasterio.open(STAGE/'road_100m_context_mask.tif') as ds:
    road = ds.read(1).astype(bool)
with rasterio.open(STAGE/'joint_SCL_conservative_land_quality.tif') as ds:
    scl_joint = ds.read(1).astype(bool)
manifest = json.loads((STAGE/'input_manifest.json').read_text(encoding='utf-8'))
bands = manifest['optical_band_order']
indices = [i for i,b in enumerate(bands) if b!='B08']
results = {}
valid_dates = {}
cloud_dates = {}
rgb_dates = {}
report = {'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
          'input_stage':str(STAGE),'device':device,'smp_version':smp.__version__,
          'weights_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),
          'strict_weight_load_pass':True,'landslide_model_run':False,
          'cloud_band_order':[b for b in bands if b!='B08']+['AOT','WVP'],
          'scaling':'optical BOA clip minimum 0 (N0500 offset already applied), AOT_DN/65535, WVP_DN/65535',
          'invalid_input_handling':'fill optical/aux invalid values with 1 only for cloud forward; output 255 where optical12 or auxiliary DN invalid',
          'padding':'author ((size//32+1)*32), centered reflect; no rescale',
          'classes':{'0':'clear','1':'thick cloud','2':'thin cloud','3':'cloud shadow','255':'invalid source'},
          'checkpoint_cloud_mapping':[1,-1,-1,1],
          'warning':'Cloud QA only; cloud predictions are not independent ground truth, not landslide or road passability predictions'}
try:
    for date in ['pre','post']:
        fn = f'{date}_{manifest["records"][date]["nz_date"]}NZ_L2A_BOA_12bands.tif'
        with rasterio.open(STAGE/fn) as ds:
            boa = ds.read()
            profile = ds.profile.copy()
        with rasterio.open(STAGE/f'{date}_spectral_validity.tif') as ds:
            spectral = ds.read(1).astype(bool)
        with rasterio.open(STAGE/f'{date}_AOT_WVP_DN_20m_to10m_nearest.tif') as ds:
            aux = ds.read().astype(np.float32)
        valid = spectral & (aux>0).all(axis=0)
        optical = np.maximum(boa[indices],0)
        optical[:,~spectral] = 1
        aux_scaled = aux/65535
        aux_scaled[aux==0] = 1
        array = np.concatenate([optical,aux_scaled],axis=0).astype('float32')
        assert array.shape==(13,640,640) and np.isfinite(array).all()
        t = torch.from_numpy(array).unsqueeze(0).to(device)
        _,_,h,w = t.shape
        dh = (h//32+1)*32-h
        dw = (w//32+1)*32-w
        top,bottom = dh//2,(dh+1)//2
        left,right = dw//2,(dw+1)//2
        t = torch.nn.functional.pad(t,(left,right,top,bottom),mode='reflect')
        with torch.inference_mode():
            logits = model(t)
        assert torch.isfinite(logits).all()
        cloud = logits.argmax(dim=1)[0,top:-bottom,left:-right].cpu().numpy().astype('uint8')
        assert cloud.shape==(640,640)
        cloud[~valid] = 255
        dest = OUT/f'{date}_SENCLOUD12.tif'
        profile.update(count=1,dtype='uint8',nodata=255,compress='lzw')
        with rasterio.open(dest,'w',**profile) as ds:
            ds.write(cloud,1)
            ds.set_band_description(1,'SENCLOUD12_cloud_class_NOT_landslide')
        with rasterio.open(dest) as ds:
            assert np.array_equal(ds.read(1),cloud)
        results[date] = {'source_valid_aoi_fraction':float(valid.mean()),
                         'source_valid_road100m_fraction':float(valid[road].mean()),
                         'aoi_class_fraction_all_pixels':{str(i):float((cloud==i).mean()) for i in [0,1,2,3,255]},
                         'road100m_class_fraction_all_pixels':{str(i):float((cloud[road]==i).mean()) for i in [0,1,2,3,255]},
                         'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        valid_dates[date]=valid
        cloud_dates[date]=cloud
        rgb_dates[date]=np.clip(boa[[3,2,1]]/.3,0,1).transpose(1,2,0)
        print(f'{date}: '+json.dumps(results[date],indent=2),flush=True)
    joint_valid = valid_dates['pre'] & valid_dates['post']
    author_mask = joint_valid & np.isin(cloud_dates['pre'],[0,3]) & np.isin(cloud_dates['post'],[0,3])
    clear_only = joint_valid & (cloud_dates['pre']==0) & (cloud_dates['post']==0)
    report['dates']=results
    report['joint']={'source_valid_road100m_fraction':float(joint_valid[road].mean()),
                     'checkpoint_cloud_mapping_retains_road100m_fraction':float(author_mask[road].mean()),
                     'clear_only_road100m_fraction':float(clear_only[road].mean()),
                     'SCL_and_clear_only_road100m_fraction':float((scl_joint&clear_only)[road].mean()),
                     'note':'Checkpoint mapping keeps class 3 shadow and does not exclude ocean; clear does not mean suitable land or safe road'}
    for name,arr in [('joint_author_cloud_visibility',author_mask),('joint_clear_only',clear_only)]:
        dest=OUT/f'{name}.tif'
        profile.update(dtype='uint8',nodata=None)
        with rasterio.open(dest,'w',**profile) as ds:
            ds.write(arr.astype('uint8'),1)
            ds.set_band_description(1,name+'_NOT_landslide')
        with rasterio.open(dest) as ds:
            assert np.array_equal(ds.read(1),arr.astype('uint8'))
    cm=ListedColormap(['#67b966','#dddddd','#f5cf55','#70489a','#202020'])
    fig,axes=plt.subplots(2,2,figsize=(12,11))
    for row,date in enumerate(['pre','post']):
        axes[row,0].imshow(rgb_dates[date]); axes[row,0].contour(road,levels=[.5],colors=['cyan'],linewidths=.4)
        axes[row,0].set_title(f'{date.upper()} L2A RGB | cyan: 100m road context')
        visual=cloud_dates[date].copy();visual[visual==255]=4
        axes[row,1].imshow(visual,cmap=cm,norm=BoundaryNorm(np.arange(-.5,5.5),5))
        axes[row,1].contour(road,levels=[.5],colors=['cyan'],linewidths=.4)
        axes[row,1].set_title(f'{date.upper()} SENCLOUD12 | CLOUD QA, NOT LANDSLIDES')
    for ax in axes.flat: ax.set_axis_off()
    fig.suptitle('Green: clear | grey: thick cloud | yellow: thin cloud | purple: shadow | black: invalid\nModel cloud classes are not ground truth; cloud-free ocean is not usable land',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(OUT/'cloud_quality_overview.png',dpi=160)
    plt.close(fig)
    report['status']='cloud_screen_complete_NOT_landslide_inference'
finally:
    (OUT/'cloud_screen_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'OUTPUT {OUT}',flush=True)
