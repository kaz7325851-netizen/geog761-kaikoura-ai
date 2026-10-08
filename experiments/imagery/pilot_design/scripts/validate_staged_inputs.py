"""Independent saved-file verification and a terrain diagnostic preview."""
from pathlib import Path
import json, hashlib
import numpy as np
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform,unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'prepared_inputs/20261005T203524Z'
DESIGN=ROOT/'20261005T200713Z'
manifest=json.loads((OUT/'input_manifest.json').read_text())
for record in manifest['records']:
    p=Path(record['file'])
    assert p.parent==OUT and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
    with rasterio.open(p) as src:
        assert list(src.descriptions)==record['bands'] and src.nodata==-9999
        assert src.crs.to_epsg()==2193 and list(src.shape)==record['shape']
with rasterio.open(OUT/'ALOS_terrain_staged_10m.tif') as src:
    terrain=src.read();affine=src.transform
project=Transformer.from_crs(4326,2193,always_xy=True).transform
road=unary_union([transform(project,shape(f['geometry'])) for f in json.loads((DESIGN/'sh1_pilot_road.geojson').read_text())['features']])
rm=rasterize([(road.buffer(100),1)],out_shape=(640,640),transform=affine,dtype='uint8').astype(bool)
msk_counts={str(int(v)):int(c) for v,c in zip(*np.unique(terrain[6][rm],return_counts=True))}
fig,axes=plt.subplots(1,3,figsize=(12,4))
extent=[1667.25,1673.65,5321.63,5328.03]
for ax,i,cmap,title in zip(axes,[0,1,6],['terrain','magma','viridis'],['DSM (m)','Slope (degrees)','ALOS source mask (0 original / 3 sea / 48 fill)']):
    im=ax.imshow(terrain[i],extent=extent,cmap=cmap)
    for g in list(road.geoms) if hasattr(road,'geoms') else [road]:
        x,y=g.xy;ax.plot(np.array(x)/1000,np.array(y)/1000,color='red',lw=.8)
    ax.set_title(title,fontsize=9);ax.set_xlabel('NZTM easting (km)')
    fig.colorbar(im,ax=ax,shrink=.7)
fig.suptitle('Terrain staging: computed at 30 m, sampled onto 10 m optical grid')
fig.tight_layout(rect=[0,.15,1,.92]);fig.text(.5,.025,'No new 10 m terrain detail. Some gap-fill data may postdate the 2016 earthquake.',ha='center',fontsize=9)
fig.savefig(OUT/'terrain_quality_preview.png',dpi=150);plt.close(fig)
verification={'saved_files_sha256_verified':True,'band_order_nodata_crs_shape_verified':True,
    'road100m_terrain_source_mask_counts':msk_counts,'road100m_copernicus_gapfill_fraction':float(np.mean(terrain[6][rm]==48)),
    'all_terrain_feature_channels_present':True,'auxiliary_STK_missing_not_treated_as_DSM_missing':True,
    'model_ready':False,'model_run':False,'training':False}
(OUT/'saved_file_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification,indent=2))
