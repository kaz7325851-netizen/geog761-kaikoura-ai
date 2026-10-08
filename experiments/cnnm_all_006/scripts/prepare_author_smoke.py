"""Reproduce inspected notebook array processing for 3 author-sample tiles.
Selection uses author's published prediction for implementation diagnostics,
NOT independent ground-truth accuracy evaluation.
"""
from pathlib import Path
import json,hashlib
import cv2,numpy as np,rasterio
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'sources';RAW=SRC/'RawData/AOI_E_S2-S2'
OUT=ROOT/'author_smoke';OUT.mkdir(exist_ok=True)
def read(path):
    with rasterio.open(path) as s:
        a=s.read();nd=s.nodata;aff=s.transform
    if a.dtype.kind=='f' and nd is not None:a[a==nd]=np.nan
    return (np.moveaxis(a,0,-1) if len(a)>1 else a[0]),aff
boundary,aff=read(RAW/'testBoundary/Test_006.tif')
yy,xx=np.where(boundary>0);bbox=[int(yy.min()),int(yy.max()),int(xx.min()),int(xx.max())]
y0,y1,x0,x1=bbox;binary=boundary[y0:y1,x0:x1];h,w=binary.shape
def resize(path):
    a,_=read(path)
    if a.shape[:2]!=(h,w):a=cv2.resize(a,(w,h),interpolation=cv2.INTER_CUBIC)
    return a[y0:y1,x0:x1]
pre=resize(RAW/'fromOptical/JIUZ_PRE_S2_RGB_010_UINT8.tif')
post=resize(RAW/'fromOptical/JIUZ_POST_S2_RGB_010_UINT8.tif')
hs=resize(RAW/'DEM/DEM_030_HILLSHADE.tif')
slope=resize(RAW/'DEM/DEM_030_SLOPE.tif')
mask=1-resize(RAW/'noDataMask/SNOW_CLOUD_MASK_010.tif');mask[binary==0]=0
reference,_=read(SRC/'Mapping_results/Predict_LS_Conf.tif')
assert reference.shape==(h,w)
locations=[]
for x in range(113,w-113,112):
    for y in range(113,h-113,112):
        tilemask=mask[y-112:y+112,x-112:x+112]
        if tilemask.mean()<.5:continue
        cm=mask[y-56:y+56,x-56:x+56]>0
        if cm.mean()<.95:continue
        ref=reference[y-56:y+56,x-56:x+56]
        locations.append({'x':x,'y':y,'central_reference_mean':float(ref[cm].mean()),
            'central_reference_positive_fraction':float(np.mean(ref[cm]>=50))})
ranked=sorted(locations,key=lambda l:l['central_reference_positive_fraction'],reverse=True)
chosen=[ranked[0]]
chosen.append(next(l for l in ranked if (l['x']-chosen[0]['x'])**2+(l['y']-chosen[0]['y'])**2>500**2))
chosen.append(min(locations,key=lambda l:l['central_reference_mean']))
records=[]
for i,l in enumerate(chosen,1):
    x,y=l['x'],l['y'];sl=np.s_[y-112:y+112,x-112:x+112]
    a=np.concatenate([post[sl],pre[sl],hs[sl][...,None],slope[sl][...,None]],axis=2)
    a=np.nan_to_num(a).astype('float32')
    a[:,:,:7]=a[:,:,:7]/127.5-1
    a[:,:,-1]=a[:,:,-1]/45-1
    np.savez_compressed(OUT/f'author_tile_{i:02d}.npz',pre=a[:,:,3:6],post=a[:,:,:3],topo=a[:,:,-2:],
        expected_confidence=reference[sl],valid=mask[sl]>0)
    records.append({**l,'tile_id':f'author_tile_{i:02d}','selection_role':'reference_prediction_rich' if i<3 else 'reference_prediction_sparse',
        'source_window_col_row':[x-112,y-112],'input_file':str(OUT/f'author_tile_{i:02d}.npz'),
        'input_sha256':hashlib.sha256((OUT/f'author_tile_{i:02d}.npz').read_bytes()).hexdigest()})
manifest={'records':records,'shape':[224,224],'physical_resolution_m':6,
    'bbox_rule':'author maximum index used as exclusive stop, preserved for reproducibility',
    'resize':'OpenCV INTER_CUBIC whole array to cropped Test_006 dimensions, matching notebook',
    'input_order':['pre_RGB','post_RGB','hillshade_and_slope'],
    'normalization':'RGB and hillshade /127.5 -1; slope /45 -1',
    'selection_before_local_prediction':True,'reference_is_author_prediction_not_ground_truth':True,'training':False}
(OUT/'sample_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
