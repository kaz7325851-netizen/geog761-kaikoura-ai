"""Independent pixel-coordinate checks and QA rendering of existing staged files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as project_geom, unary_union
import cv2
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

parser=argparse.ArgumentParser()
parser.add_argument('stage',type=Path)
args=parser.parse_args()
stage=args.stage.resolve()
root=Path(__file__).resolve().parent
assert stage.parent==root/'prepared_inputs'
m=json.loads((stage/'input_manifest.json').read_text(encoding='utf-8'))
samples=[(r,c) for r in [53,163,273,383,593] for c in [47,157,267,377,587]]
coordinate_checks=[]
arrays={}
for k,nzdate in [('pre','20161103'),('post','20161123')]:
    with rasterio.open(stage/f'{k}_{nzdate}NZ_L2A_original_DN_12bands.tif') as s:
        dn=s.read(); grid=s.transform
        xx,yy=rasterio.transform.xy(grid,[r for r,c in samples],[c for r,c in samples],offset='center')
    with rasterio.open(stage/f'{k}_{nzdate}NZ_L2A_BOA_12bands.tif') as s:
        boa=s.read()
    with rasterio.open(stage/f'{k}_SCL_20m_to10m_nearest.tif') as s:
        scl=s.read(1)
    archive=root/(m['records'][k]['product']+'.zip')
    xsrc,ysrc=Transformer.from_crs(2193,32759,always_xy=True).transform(xx,yy)
    for i,b in enumerate(m['optical_band_order']):
        member=next(x['member'] for x in m['records'][k]['members'] if x['band']==b)
        with rasterio.open('/vsizip/'+archive.as_posix()+'/'+member) as s:
            rows,cols=rasterio.transform.rowcol(s.transform,xsrc,ysrc)
            r0,c0=min(rows),min(cols);r1,c1=max(rows)+1,max(cols)+1
            source=s.read(1,window=Window(c0,r0,c1-c0,r1-r0))
            expected=np.array([source[r-r0,c-c0] for r,c in zip(rows,cols)])
        actual=np.array([dn[i,r,c] for r,c in samples])
        assert np.array_equal(actual,expected),(k,b,actual,expected)
        valid=(dn[i]!=0)&(dn[i]!=65535)
        expected_boa=(dn[i].astype('float64')+m['records'][k]['offsets'][i])/m['records'][k]['quantification']
        maxerr=float(np.max(np.abs(boa[i][valid]-expected_boa[valid])))
        assert maxerr<1e-6
        assert np.all(boa[i][~valid]==-9999)
        coordinate_checks.append({'date':k,'band':b,'sample_count':len(samples),
                                  'original_JP2_DN_matches':True,'BOA_full_array_max_abs_error':maxerr})
    arrays[k]={'dn':dn,'boa':boa,'scl':scl}

for f in m['files']:
    p=Path(f['path'])
    assert p.parent==stage
    assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
    with rasterio.open(p) as s:
        assert s.crs.to_epsg()==2193 and s.shape==(640,640) and s.transform==grid
        assert list(s.descriptions)==f['bands']
project=Transformer.from_crs(4326,2193,always_xy=True).transform
design=root.parent/'pilot_design'/'20261005T200713Z'
road=unary_union([project_geom(project,shape(f['geometry'])) for f in json.loads((design/'sh1_pilot_road.geojson').read_text())['features']])
inventory=unary_union([project_geom(project,shape(f['geometry'])) for f in json.loads((design/'reference_inventory.geojson').read_text())['features']])
roadparts=list(road.geoms) if hasattr(road,'geoms') else [road]
points=[]
for line in roadparts:
    points.extend([line.interpolate(d) for d in np.arange(0,line.length,5)])
rows,cols=rasterio.transform.rowcol(grid,[p.x for p in points],[p.y for p in points])
indices=[(r,c) for r,c in zip(rows,cols) if 0<=r<640 and 0<=c<640]
with rasterio.open(stage/'joint_SCL_conservative_land_quality.tif') as s:
    joint=s.read(1)
with rasterio.open(stage/'road_100m_context_mask.tif') as s:
    roadmask=s.read(1).astype(bool)
with rasterio.open(stage/'terrain_PROVISIONAL_DEM_slope_aspect.tif') as s:
    dem=s.read(1)
report={'all_saved_raster_hashes_match':True,'all_grids_match':True,
        'independent_JP2_pixel_checks':coordinate_checks,'JP2_comparisons_total':len(coordinate_checks)*len(samples),
        'road_centerline_sampling':{'spacing_m':5,'points':len(indices),
                                    'joint_SCL_rule_pass_fraction':float(np.mean([joint[r,c] for r,c in indices])),
                                    'warning':'Current OSM road centerline, mixed 10m pixels; not road passability'},
        'road_buffer_SCL_breakdown':{k:{str(i):float(((arrays[k]['scl']==i)&roadmask).sum()/roadmask.sum())
                                       for i in range(12)} for k in ['pre','post']},
        'snow_flagged_pre_road_context':{'pixels':int(((arrays['pre']['scl']==11)&roadmask).sum()),
                                       'provisional_DSM_m_percentiles':np.percentile(dem[(arrays['pre']['scl']==11)&roadmask],[0,50,100]).tolist(),
                                       'interpretation':'Possible bright coastal-feature misclassification; no SCL labels changed'},
        'independent_geolocation_ground_truth_checked':False,'model_run':False}

fig,axs=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
extent=[1667250,1673650,5321630,5328030]
def draw(ax,geom,color,width):
    parts=list(geom.geoms) if hasattr(geom,'geoms') else [geom]
    for g in parts:
        if g.geom_type=='LineString': xy=np.array(g.coords)
        elif g.geom_type=='Polygon': xy=np.array(g.exterior.coords)
        else: continue
        ax.plot(xy[:,0],xy[:,1],color=color,lw=width)
for j,k in enumerate(['pre','post']):
    rgb=np.clip(arrays[k]['boa'][[3,2,1]].transpose(1,2,0)/0.3,0,1)**(1/1.6)
    axs[0,j].imshow(rgb,extent=extent)
    axs[0,j].set_title(f'{k.upper()} L2A BOA RGB\nFixed stretch; cyan=SH1, yellow=inventory',fontsize=10)
    draw(axs[0,j],road,'cyan',1);draw(axs[0,j],inventory,'yellow',.4)
    cmap=ListedColormap(['black','red','#555555','#993366','#218c35','#ac864e','#217dcc','#cccccc','#bbbbff','white','#7d00e0','#00ffff'])
    axs[1,j].imshow(arrays[k]['scl'],extent=extent,cmap=cmap,norm=BoundaryNorm(np.arange(-.5,12.5),12))
    axs[1,j].set_title(f'{k.upper()} Sen2Cor SCL\nBlue=water; cyan=snow/ice flag (needs review)',fontsize=10)
    draw(axs[1,j],road,'cyan',1)
axs[0,2].imshow(joint,extent=extent,cmap=ListedColormap(['#444444','#59b65f']))
axs[0,2].set_title('Conservative SCL quality, both dates\nGreen=passes screening; NOT prediction',fontsize=10)
draw(axs[0,2],road,'cyan',1)
axs[1,2].imshow(dem,extent=extent,cmap='terrain',vmin=0,vmax=1200)
axs[1,2].set_title('Provisional ALOS DSM\nNOT author terrain reproduction',fontsize=10)
draw(axs[1,2],road,'cyan',1)
for ax in axs.ravel():
    ax.set_xlabel('NZTM easting (m)');ax.set_ylabel('NZTM northing (m)')
    ax.ticklabel_format(style='plain',axis='both',useOffset=False)
fig.suptitle('Frozen SH1 pilot | 2016-11-03 / 2016-11-23 NZ | DATA QA, NOT PREDICTIONS',fontsize=13)
fig.savefig(stage/'L2A_quality_overview_verified.png',dpi=140)
plt.close(fig)
(stage/'independent_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='independent_JP2_pixel_checks'},ensure_ascii=False,indent=2))
