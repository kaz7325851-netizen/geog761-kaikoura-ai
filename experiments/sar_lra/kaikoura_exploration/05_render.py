"""Render existing saved results; does not load or rerun the model."""
from pathlib import Path
import os
import json
import numpy as np
import rasterio
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT.parent.parent/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
OUT=Path(json.loads((ROOT/'latest_result.json').read_text())['result_directory'])
report=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
saved=np.load(OUT/'patch_predictions.npz')
with rasterio.open(report['input_file']) as src:data=src.read()
with rasterio.open(ROOT/'reference/reference_mask.tif') as src:reference=src.read(1)
with rasterio.open(ROOT/'reference/road_corridor_mask.tif') as src:road=src.read(1)
with rasterio.open(OUT/'suspect_box_coverage.tif') as src:coverage=src.read(1)
valid=data[4]==1
recent=valid&(data[9]==1)
fig,axes=plt.subplots(2,3,figsize=(15,9.5),layout='constrained')
for ax,arr,title in [(axes[0,0],np.where(valid,data[2],np.nan),'postVV - Aug23 preVV (dB)'),
                     (axes[1,0],np.where(recent,data[8],np.nan),'Recent preVV median - Aug23 preVV (dB)')]:
    im=ax.imshow(arr,cmap='RdBu_r',vmin=-6,vmax=6)
    ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
axes[0,1].imshow(reference,cmap=ListedColormap(['#eeeeee','#d49426']),vmin=0,vmax=1)
axes[0,1].contour(road,levels=[.5],colors=['#4198a4'],linewidths=.6)
axes[0,1].set_title('Reference slide pixels + old road corridor')
axes[0,2].imshow(np.ma.masked_where(coverage==255,coverage),cmap=ListedColormap(['#eeeeee','#cd5c5c']),vmin=0,vmax=1)
axes[0,2].set_title(f"Retained coarse detection boxes: {report['retained_boxes_after_nms']}\nGrey: no detection (not safe); white: unscored",fontsize=11)
scores=np.full((18,18),np.nan);confusion=np.full((18,18),np.nan)
for (x,y),s,t,p in zip(saved['coords'],saved['scores'],saved['truth'],saved['pred']):
    scores[y//32,x//32]=s
    confusion[y//32,x//32]=3 if t and p else 2 if t else 1 if p else 0
im=axes[1,1].imshow(scores,vmin=0,vmax=1,cmap='viridis')
axes[1,1].set_title('Overlapping WINDOW scores (18x18 positions)')
fig.colorbar(im,ax=axes[1,1],shrink=.7)
cm=ListedColormap(['#d6d6d6','#df6a58','#466ab7','#3b9f76'])
axes[1,2].imshow(confusion,cmap=cm,vmin=-.5,vmax=3.5,interpolation='nearest')
axes[1,2].set_title('WINDOW comparison (>5% reference; score>0.6)\nWhite positions were excluded',fontsize=11)
axes[1,2].legend(handles=[Patch(color=cm.colors[i],label=l) for i,l in enumerate(['TN','FP','FN','TP'])],loc='upper center',bbox_to_anchor=(.5,-.02),ncol=4)
for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Kaikoura 2016 SAR-LRA exploration: real ascending checkpoint\n82-day-old baseline; coarse slide windows, NOT road passability',fontsize=14)
fig.savefig(OUT/'comparison.png',dpi=150)
plt.close(fig)
print(OUT/'comparison.png')
