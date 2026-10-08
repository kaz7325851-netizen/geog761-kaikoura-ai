"""Compare fixed GDAL derivative algorithms against published author rasters.
No fitted light angles, RGB parameters, weights, or model thresholds.
"""
from pathlib import Path
import os,subprocess,json,hashlib
import numpy as np,rasterio
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'preprocessing_audit/terrain';OUT.mkdir(parents=True,exist_ok=True)
BIN=ROOT.parent/'delft_rf/rsgis_env/Library/bin'
EXE=BIN/'gdaldem.exe'
env=os.environ.copy();env['PATH']=str(BIN)+os.pathsep+env.get('PATH','')
env['GDAL_DATA']=str(BIN.parent/'share/gdal')
RAW=ROOT/'sources/RawData/AOI_E_S2-S2/DEM'
DATA=ROOT.parent/'pilot_design/prepared_inputs/20261005T203524Z'
version=subprocess.run([str(BIN/'gdalinfo.exe'),'--version'],env=env,check=True,capture_output=True,text=True).stdout.strip()
records=[]
for alg in ['Horn','ZevenbergenThorne']:
    for mode,suffix in [('slope','SLOPE'),('hillshade','HILLSHADE')]:
        dst=OUT/f'author_dem_{alg}_{mode}.tif'
        cmd=[str(EXE),mode,str(RAW/'DEM_030.tif'),str(dst),'-s','1','-q']
        if mode=='hillshade':cmd+=['-az','315','-alt','45','-z','1']
        if alg!='Horn':cmd+=['-alg',alg]
        subprocess.run(cmd,env=env,check=True,capture_output=True,text=True)
        with rasterio.open(RAW/f'DEM_030_{suffix}.tif') as ds:expected=ds.read(1);em=ds.read_masks(1)>0
        with rasterio.open(dst) as ds:actual=ds.read(1);am=ds.read_masks(1)>0
        with rasterio.open(RAW/'DEM_030.tif') as ds:dem=ds.read(1)
        for region,mask in [('all_joint_valid',em&am),('nonzero_dem_joint_valid',em&am&(dem>0))]:
            diff=np.abs(actual.astype('float64')-expected.astype('float64'))[mask]
            records.append({'algorithm':alg,'mode':mode,'region':region,'pixels':int(mask.sum()),
                'mean_absolute_error':float(diff.mean()),'max_absolute_error':float(diff.max()),
                'exact_equal_fraction':float(np.mean(diff==0)),
                'within_1e_4_fraction':float(np.mean(diff<=1e-4)),
                'source_valid_pixels':int(em.sum()),'candidate_valid_pixels':int(am.sum()),
                'valid_mask_disagreement_pixels':int(np.sum(em!=am)),
                'candidate_file':str(dst),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
with rasterio.open(DATA/'ALOS_terrain_30m_audit.tif') as ds:
    profile=ds.profile.copy();profile.update(count=1,dtype='float32',nodata=-9999,compress='deflate')
    dem=ds.read(1)
    with rasterio.open(OUT/'pilot_dem_30m.tif','w',**profile) as w:w.write(dem,1)
for mode in ['slope','hillshade']:
    dst=OUT/f'pilot_Horn_{mode}_30m.tif'
    cmd=[str(EXE),mode,str(OUT/'pilot_dem_30m.tif'),str(dst),'-s','1','-q']
    if mode=='hillshade':cmd+=['-az','315','-alt','45','-z','1']
    subprocess.run(cmd,env=env,check=True,capture_output=True,text=True)
    with rasterio.open(dst) as ds:a=ds.read(1).astype('float32');valid=ds.read_masks(1)>0
    with rasterio.open(DATA/'ALOS_terrain_30m_audit.tif') as ds:b=ds.read(2 if mode=='slope' else 4)
    diff=np.abs(a-b)[valid]
    records.append({'source':'pilot','algorithm':'Horn','mode':mode,'region':'joint_valid','pixels':int(valid.sum()),
        'mean_absolute_error':float(diff.mean()),'max_absolute_error':float(diff.max()),'candidate_file':str(dst)})
summary={'gdaldem_version':version,'commands_fixed_before_comparison':True,
    'default_hillshade_settings':{'azimuth_deg':315,'altitude_deg':45,'z_factor':1,'scale':1},
    'author_workflow_documented':False,'records':records}
(OUT/'terrain_algorithm_comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
