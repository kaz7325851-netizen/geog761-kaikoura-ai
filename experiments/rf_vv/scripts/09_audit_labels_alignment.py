"""VV label audit, geography-only split, and local registration diagnostics.

No model fitting or auto-registration. Candidate negatives remain UNKNOWN.
Optical panels are reference review ONLY, never RF predictor inputs.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json, os
import numpy as np
import rasterio, shapefile
from rasterio.features import rasterize
from shapely.geometry import shape, box, Point, mapping
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree
from pyproj import Transformer, CRS
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_erosion

ROOT = Path(__file__).resolve().parent
PREP = Path(json.loads((ROOT/'latest_prepared_regions.json').read_text(encoding='utf-8'))['directory'])
prepared = json.loads((PREP/'manifest.json').read_text(encoding='utf-8'))
design = json.loads((ROOT/'frozen_data_design.json').read_text(encoding='utf-8'))
OUT = ROOT/'label_alignment_audit'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True, exist_ok=False)
OPT = ROOT.parent/'bbunet/expanded_test/20261007T210924Z'
SHP = ROOT.parent/'sar_lra/kaikoura_exploration/reference/Kaikora_2016.shp'
GNS = ROOT/'official_reference_points/GNS_V2_source_centroids_2193.json'
assert GNS.exists(), 'Official GNS cross-check not yet downloaded'
os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
def read(p):
    with rasterio.open(p) as src: return src.read(), src.profile.copy()
def write_mask(dest, values, profile, meaning):
    cfg = dict(profile); cfg.update(count=1, dtype='uint8', nodata=None, compress='deflate')
    with rasterio.open(dest, 'w', **cfg) as dst:
        dst.write(values.astype('uint8'), 1)
        dst.set_band_description(1, meaning)
        dst.update_tags(training_ready='false', verified_negative='false')
    a, _ = read(dest); assert np.array_equal(a[0], values.astype('uint8'))

# Freeze by geography before label/registration diagnostics. No class-count optimisation.
sites = sorted(design['sites'], key=lambda s:(s['bounds'][1]+s['bounds'][3])/2)
assign = {s['id']: ('train' if i<5 else 'validation' if i==5 else 'test') for i,s in enumerate(sites)}
geoms = {s['id']:box(*s['bounds']) for s in sites}
parts = {k:unary_union([geoms[r] for r in assign if assign[r]==k]) for k in ['train','validation','test']}
distances = {f'{a}_to_{b}_m':parts[a].distance(parts[b])
             for a,b in [('train','validation'),('train','test'),('validation','test')]}
assert min(distances.values())>=1000
split = {'version':'v1', 'design_sha256':sha(ROOT/'frozen_data_design.json'), 'assignments':assign,
         'selection_rule':'Sort all frozen regions south-to-north: southern5 train, next1 validation, northern2 test',
         'minimum_aoi_gap_m':distances, 'feature_neighborhood_radius_m':30,
         'status':'geographic_assignment_frozen_labels_NOT_ready', 'not_blind':True,
         'prior_exposure':'Inventory distributions and earlier optical model results are already known',
         'policy':['No random pixel split; do not change after RF scores are observed',
                   'Test labels may be independently reviewed before training, not used for tuning',
                   'Quarantine whole source polygons if they intersect multiple roles',
                   'This is within-event spatial transfer, not cross-event validation']}
split_path=ROOT/'spatial_split_v1.json'
if split_path.exists():
    assert json.loads(split_path.read_text(encoding='utf-8'))==split
else: dump(split_path, split)
dump(OUT/'spatial_split_snapshot.json', split)

reader=shapefile.Reader(str(SHP))
project=Transformer.from_crs(CRS.from_wkt(SHP.with_suffix('.prj').read_text()),2193,always_xy=True).transform
inverse=Transformer.from_crs(2193,CRS.from_wkt(SHP.with_suffix('.prj').read_text()),always_xy=True).transform
whole=unary_union(list(geoms.values())).buffer(1100)
sourcebox=transform(inverse,whole).envelope.buffer(100)
polys=[]; attributes=[]; repairs=0
for idx, sr in enumerate(reader.iterShapeRecords()):
    if not box(*sr.shape.bbox).intersects(sourcebox): continue
    g=transform(project,shape(sr.shape.__geo_interface__))
    if not g.is_valid: g=g.buffer(0); repairs+=1
    if g.is_empty or not g.intersects(whole): continue
    polys.append(g); attributes.append({'source_index':idx, **sr.record.as_dict()})
tree=STRtree(polys)
points=[Point(f['geometry']['x'],f['geometry']['y']) for f in json.loads(GNS.read_text(encoding='utf-8'))['features']]
ptree=STRtree(points)
cross_roles=[]
for i,g in enumerate(polys):
    roles=[k for k,p in parts.items() if g.intersects(p)]
    if len(roles)>1: cross_roles.append(attributes[i]['source_index'])
fields=[f[0] for f in reader.fields[1:]]
metadata={'source_shp':str(SHP), 'sha256':sha(SHP),'total_records':len(reader), 'fields':fields,
          'LD_counts':dict(Counter(r['LD'] for r in reader.iterRecords())),
          'LD_not_binary_slide_label':True,
          'reason':'XML lineage joins landslide polygons to dams; LD=0 polygons are NOT non-slide examples',
          'per_object_event_date_or_confidence_fields_present':False,
          'local_polygons':len(polys), 'local_geometry_repairs':repairs,
          'cross_role_polygon_source_indices_quarantined':cross_roles,
          'metadata_title':'Kaikoura_Ls_Final_2020_May_S5',
          'official_gns_layer':'https://gis.gns.cri.nz/server/rest/services/EILD/EILD_V2_Map/MapServer/0',
          'official_gns_points_sha256':sha(GNS),
          'gns_points_interpretation':'V2 source centroids ONLY; supports omission warnings, not polygon truth',
          'v3_polygon_source':'https://doi.org/10.17603/ds2-rpyg-0h84',
          'v3_download_status':'Not downloaded; publication verified, download page returned403 in web tool',
          'all_candidate_labels_remain_unverified':True}
dump(OUT/'reference_source_audit.json',metadata)

def smooth_highpass(a):
    a=gaussian_filter(a.astype('float64'),1.5)
    return a-gaussian_filter(a,8)
def ncc_diagnostic(pre, post, mask, yc, xc, half=64, search=3, minimum=1800):
    # (dy,dx) shifts POST to PRE. Compare PRE[y,x] against POST[y-dy,x-dx].
    sl=(slice(yc-half,yc+half),slice(xc-half,xc+half))
    a=pre[sl]; ma=mask[sl]; scores=[]
    for dy in range(-search,search+1):
        for dx in range(-search,search+1):
            ps=(slice(yc-half-dy,yc+half-dy),slice(xc-half-dx,xc+half-dx))
            m=ma&mask[ps]; n=int(m.sum())
            if n<minimum: continue
            x=a[m]; y=post[ps][m]; x=x-x.mean(); y=y-y.mean()
            den=np.sqrt(np.dot(x,x)*np.dot(y,y))
            if den<=1e-8: continue
            scores.append((float(np.dot(x,y)/den),dy,dx,n))
    if not scores: return {'row':yc,'col':xc,'status':'insufficient_land_support'}
    scores.sort(reverse=True); best,dy,dx,n=scores[0]
    distant=[s[0] for s in scores if max(abs(s[1]-dy),abs(s[2]-dx))>1]
    margin=best-max(distant) if distant else 0
    zero=next((s[0] for s in scores if s[1]==0 and s[2]==0),None)
    reliable=best>=.55 and margin>=.03 and abs(dy)<search and abs(dx)<search
    return {'row':yc,'col':xc,'best_post_correction_px':[dy,dx], 'best_ncc':best,
            'zero_shift_ncc':zero,'peak_margin_outside1px':margin,'support_pixels':n,
            'status':'usable_diagnostic_peak' if reliable else 'ambiguous_NOT_coregistration_pass'}

# Unit test sign convention, deterministic known shifts, NOT real-image accuracy.
rng=np.random.default_rng(761)
base=gaussian_filter(rng.normal(size=(256,256)),1.5)
synthetic=[]
for displacement in [(0,0),(1,-2),(-2,1)]:
    moving=np.roll(base,displacement,axis=(0,1))
    match=ncc_diagnostic(base,moving,np.ones_like(base,dtype=bool),128,128)
    assert match['best_post_correction_px']==[-displacement[0],-displacement[1]]
    synthetic.append({'post_displacement_px':displacement,'found_correction':match['best_post_correction_px']})

records=[]; samples=[]; cache={}
for r in prepared['records']:
    rid=r['id']; dest=OUT/rid; dest.mkdir()
    raw,profile=read(r['raw_file']); grid=(raw.shape[1:],profile['transform'],profile['crs'])
    feat,_=read(PREP/rid/'VV_feature_candidates.tif')
    screen,_=read(PREP/rid/'screened_feature_mask.tif'); screen=screen[0]==1
    inv,_=read(PREP/rid/'inventory_reference_only.tif'); inv=inv[0]==1
    road,_=read(PREP/rid/'road100m_context_mask.tif'); road=road[0]==1
    region=geoms[rid]
    chosen=list(map(int,tree.query(region.buffer(100))))
    direct=[i for i in chosen if polys[i].intersects(region)]
    def raster(geometries):
        nonempty=[(mapping(g),1) for g in geometries if not g.is_empty]
        return rasterize(nonempty,out_shape=grid[0],transform=grid[1],dtype='uint8').astype(bool) if nonempty else np.zeros(grid[0],bool)
    reproduced=raster([polys[i] for i in direct]); assert np.array_equal(reproduced,inv)
    core=raster([polys[i].buffer(-20) for i in direct if attributes[i]['source_index'] not in cross_roles])&screen
    guard=raster([polys[i].buffer(100) for i in chosen])
    pp=[int(i) for i in ptree.query(region.buffer(110)) if points[int(i)].distance(region)<110]
    pointguard=raster([points[i].buffer(100) for i in pp])
    elevation=raw[7]
    # This is just a coarse land proxy; coastal uncertainty is not certified away.
    land=(elevation>5)&np.isfinite(elevation)&(elevation!=-9999)
    negative=screen&land&~guard&~pointguard
    gaps=[]
    for i in pp:
        if region.covers(points[i]):
            nearest=min((points[i].distance(polys[j]) for j in chosen),default=float('inf'))
            gaps.append({'point_index':i,'distance_to_archive_polygon_m':nearest if np.isfinite(nearest) else None})
    warn=[g for g in gaps if g['distance_to_archive_polygon_m'] is None or g['distance_to_archive_polygon_m']>100]
    write_mask(dest/'inventory_interior20m_CANDIDATE.tif',core,profile,'inventory_interior20m_not_verified_positive')
    write_mask(dest/'inventory_buffer100m.tif',guard,profile,'experimental_uncertainty_buffer_not_accuracy_bound')
    write_mask(dest/'gns_source_point_buffer100m.tif',pointguard,profile,'centroid_warning_buffer_not_slide_extent')
    write_mask(dest/'negative_review_candidates_NOT_LABELS.tif',negative,profile,'review_proposals_only_unknown_not_class0')
    # Labels are not assigned to negative proposals. Preserve uncertainty explicitly.
    labels=np.full(grid[0],255,dtype='uint8'); labels[core]=1
    cfg=dict(profile);cfg.update(count=1,dtype='uint8',nodata=255,compress='deflate')
    with rasterio.open(dest/'provisional_core_labels_NOT_TRAIN_READY.tif','w',**cfg) as dst:
        dst.write(labels,1);dst.set_band_description(1,'1_unverified_inventory_interior_255_unknown_NO_verified0')
    check,_=read(dest/'provisional_core_labels_NOT_TRAIN_READY.tif')
    assert np.array_equal(check[0],labels) and 0 not in np.unique(labels)
    # Potentially unchanging terrain, NOT independently verified stable ground.
    matchmask=binary_erosion(screen&land&~guard&~pointguard,np.ones((11,11)))
    pre=smooth_highpass(np.where(raw[0]!=-9999,raw[0],-20))
    post=smooth_highpass(np.where(raw[1]!=-9999,raw[1],-20))
    matches=[ncc_diagnostic(pre,post,matchmask,y,x) for y in [80,240,400,560] for x in [80,240,400,560]]
    usable=[m for m in matches if m['status']=='usable_diagnostic_peak']
    optical=[]
    for kind in ['pre','post']:
        a,p=read(OPT/rid/f'{kind}_BOA.tif'); scl,sp=read(OPT/rid/f'{kind}_SCL.tif')
        assert (a.shape[1:],p['transform'],p['crs'])==grid
        assert (scl.shape[1:],sp['transform'],sp['crs'])==grid
        rgb=np.moveaxis(a[[3,2,1]],0,-1)
        quality=np.isin(scl[0],[4,5])&np.isfinite(rgb).all(axis=2)&(rgb!=-9999).all(axis=2)
        optical.append((np.clip(rgb/.3,0,1)**.7,quality))
    # Deterministic proposal sampling; no VV changes, predictions, or optical result used to select labels.
    choices=[]; localrng=np.random.default_rng(761+sites.index(next(s for s in sites if s['id']==rid)))
    edge=np.zeros_like(screen); edge[40:-40,40:-40]=True
    for near in [True,False]:
        pool=np.argwhere(negative&edge&(road if near else ~road))
        for low,high in [(0,10),(10,25),(25,90)]:
            validpool=pool[(raw[8,pool[:,0],pool[:,1]]>=low)&(raw[8,pool[:,0],pool[:,1]]<high)] if len(pool) else pool
            if len(validpool):
                perm=localrng.permutation(len(validpool))
                for z in perm:
                    y,x=map(int,validpool[z])
                    if all(np.hypot(y-c['row'],x-c['col'])>=20 for c in choices):
                        choices.append({'row':y,'col':x,'kind':'negative_review_proposal',
                                        'road100m':near,'slope_stratum_deg':[low,high]});break
    for near in [True,False]:
        pool=np.argwhere(core&edge&(road if near else ~road))
        if len(pool):
            y,x=map(int,pool[localrng.integers(len(pool))]); choices.append({'row':y,'col':x,
                'kind':'positive_inventory_review_proposal','road100m':near})
    for i,c in enumerate(choices):
        c.update(id=f'{rid}-R{i+1:02}',region_id=rid,role=assign[rid],review_status='UNREVIEWED',
                 final_label=None, x_nztm=float(grid[1].c+10*(c['col']+.5)),
                 y_nztm=float(grid[1].f-10*(c['row']+.5)),
                 pre_optical_center_clear=bool(optical[0][1][c['row'],c['col']]),
                 post_optical_center_clear=bool(optical[1][1][c['row'],c['col']]),
                 optical_use='Independent coarse label-review context only; not RF inputs',
                 optical_time_caveat='pre2016-11-02UTC/post2016-11-22UTC, postlaterthanVV2016-11-15')
        samples.append(c)
    rec={'id':rid,'role':assign[rid],'archive_polygon_count':len(direct),
         'archive_inventory_pixels':int(inv.sum()),'positive_core20m_candidate_pixels':int(core.sum()),
         'negative_review_candidate_pixels_NOT_verified0':int(negative.sum()),
         'verified_negative_pixels':0,'gns_source_centroids_in_region':len(gaps),
         'gns_centroids_over100m_from_archive_polygons_WARNING':len(warn),
         'centroid_warnings':warn,'registration_windows':matches,'usable_registration_windows':len(usable),
         'usable_post_corrections_px':[m['best_post_correction_px'] for m in usable],
         'registration_certified':False,'no_resampling_applied':True,
         'candidate_review_samples':len(choices),'source_hashes':{
             'raw':sha(r['raw_file']), 'feature':sha(PREP/rid/'VV_feature_candidates.tif'),
             'pre_optical_review_only':sha(OPT/rid/'pre_BOA.tif'),
             'post_optical_review_only':sha(OPT/rid/'post_BOA.tif')}}
    dump(dest/'audit.json',rec);records.append(rec)
    cache[rid]={'VV_pre':raw[0],'VV_post':raw[1],'optical':optical,'inv':inv,'road':road}
    print(json.dumps({k:rec[k] for k in ['id','positive_core20m_candidate_pixels','gns_centroids_over100m_from_archive_polygons_WARNING','usable_registration_windows']},ensure_ascii=False),flush=True)

dump(OUT/'review_samples_UNREVIEWED.json',samples)
# Review cards: four samples per page, all plotted with the same stretches.
for page in range((len(samples)+3)//4):
    subset=samples[page*4:page*4+4]
    fig,axes=plt.subplots(len(subset),4,figsize=(12,3.0*len(subset)),squeeze=False,layout='constrained')
    for row,c in enumerate(subset):
        d=cache[c['region_id']]; y,x=c['row'],c['col']; sl=(slice(y-32,y+32),slice(x-32,x+32))
        for j in [0,1]: axes[row,j].imshow(d['VV_pre' if j==0 else 'VV_post'][sl],cmap='gray',vmin=-25,vmax=0)
        for j in [2,3]:
            rgb,quality=d['optical'][j-2]
            axes[row,j].imshow(rgb[sl])
            axes[row,j].imshow(np.where(quality[sl],np.nan,1),cmap='Reds',alpha=.20,vmin=0,vmax=1)
        for ax in axes[row]:
            local=d['inv'][sl]
            if local.any() and not local.all(): ax.contour(local,levels=[.5],colors=['yellow'],linewidths=.6)
            local=d['road'][sl]
            if local.any() and not local.all(): ax.contour(local,levels=[.5],colors=['cyan'],linewidths=.6)
            ax.plot(32,32,'+',color='magenta',ms=9);ax.set_xticks([]);ax.set_yticks([])
        axes[row,0].set_ylabel(f"{c['id']}\n{c['kind'].replace('_review_proposal','')}\n{c['role']}",fontsize=8)
    for ax,t in zip(axes[0],['VV pre Nov3 UTC','VV post Nov15 UTC','S2 pre Nov2 UTC (review only)','S2 post Nov22 UTC (review only)']):ax.set_title(t,fontsize=9)
    fig.suptitle('UNREVIEWED samples: yellow=archive polygons, cyan=100m road context, magenta=sample\n640m context. Red optical tint=non-clear SCL. Post-S2 is 7 days later than post-VV.',fontsize=10)
    fig.savefig(OUT/f'review_page_{page+1:02}.png',dpi=120);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,9),layout='constrained')
colors={'train':'#399358','validation':'#d79b25','test':'#4a73b5'}
for s in sites:
    g=geoms[s['id']]; x,y=g.exterior.xy
    ax.fill(x,y,color=colors[assign[s['id']]],alpha=.4)
    ax.text(g.centroid.x,g.centroid.y,f"{s['id']}\n{assign[s['id']]}",fontsize=9,ha='center')
ax.set_aspect('equal');ax.ticklabel_format(style='plain');ax.set_xlabel('NZTM Easting (m)');ax.set_ylabel('NZTM Northing (m)')
ax.set_title('Frozen within-event geography-only split\n5 train / 1 validation / 2 test; prior regions already known, NOT blind')
fig.savefig(OUT/'spatial_split_map.png',dpi=140);plt.close(fig)

report={'directory':str(OUT),'status':'audit_complete_NOT_training_ready','created_utc':datetime.now(timezone.utc).isoformat(),
        'prepared_manifest_sha256':sha(PREP/'manifest.json'),'source_metadata':metadata,
        'split_file':str(split_path),'split_sha256':sha(split_path),'split':split,'synthetic_registration_tests':synthetic,
        'regions':records,'review_sample_count':len(samples),'reviewed_sample_count':0,
        'verified_negative_pixels':0,'training':False,'coregistration_certified':False,
        'candidate_rules':'Interior20m and guard100m are experimental QC heuristics, not author rules or validated accuracy bounds',
        'label_caveats':['No negatives automatically assigned from absent inventory or low VV change',
                        'GNS source points can warn of omissions but not reconstruct debris extents',
                        'S2 available only as coarse independent review, not RF input',
                        'Later optical image cannot establish every Nov15 state; small slides can be invisible',
                        'Local NCC peaks do not certify ground truth, terrain visibility, or alignment everywhere']}
dump(OUT/'audit_summary.json',report)
dump(ROOT/'latest_label_alignment_audit.json',{'directory':str(OUT),'training_ready':False})
rows=['|区域|集合|清单多边形|20m内核候选像素|官方中心点距离旧清单>100m的警告|可用对齐诊断窗口|',
      '|---|---|---:|---:|---:|---:|']
for r in records:
    rows.append(f"|{r['id']}|{r['role']}|{r['archive_polygon_count']}|{r['positive_core20m_candidate_pixels']}|{r['gns_centroids_over100m_from_archive_polygons_WARNING']}|{r['usable_registration_windows']}/16|")
text='''# RF下一步：标签、配准和空间划分核查

本轮未训练RF，也没有自动将待核验点标成负类。所有原始数据和旧结果保留。

## 已完成

1. 核查作者存档清单的字段、几何和栅格化一致性。LD=0不能当作“无滑坡”；元数据过程记录把滑坡图层与堰塞坝图层连接，LD不是本项目滑坡正负标签。文件缺少逐条事件日期/置信度字段。
2. 从[GNS官方公开接口](https://gis.gns.cri.nz/server/rest/services/EILD/EILD_V2_Map/MapServer/0)下载V2滑坡源区中心点，辅助检查旧清单遗漏。它不是完整多边形，也不是道路受阻数据。
3. 为避免边界混合，生成清单向内20m的候选核；为待核验负样本加入清单和官方中心点100m排除范围。这些距离是实验规则，不是课程、作者或官方规定，也不是精度保证。小滑坡可能没有20m内核，不等于不存在。
4. 灾前后VV局部归一化相关匹配：每区固定16个窗口，搜索±3像素，不根据结果自动平移影像。已用已知位移的合成影像检查算法符号；这不是实际配准精度验证。
5. 按南北地理位置固定5区训练、1区验证、2区测试，不按像素随机拆分，不按RF分数重选。最小不同集合AOI间隙超过1km。长多边形如果跨集合，将整对象隔离。旧区域已被其他模型观察过，因此不是盲测，也不是跨事件验证。
6. 生成待核验样本清单和四列对照图（VV灾前/灾后、S2灾前/灾后）。S2仅用于独立标签审查，不加入本版RF预测特征；S2灾后日期比VV晚7天，不能自动代表11月15日状态。

## 分区摘要

'''+ '\n'.join(rows)+f'''

## 文件位置

核查输出目录：`{OUT}`。

- `audit_summary.json`：数值诊断、来源、限制。
- `review_samples_UNREVIEWED.json`：{len(samples)}个待核验候选，final_label均为空；不是可训练样本表。
- `review_page_*.png`：候选点影像对照；红色半透明表示光学SCL不清晰类别。
- `spatial_split_map.png`：固定地理分区。
- 每个分区的`negative_review_candidates_NOT_LABELS.tif`仅是复核提案，不是负类标签。
- 正类内核仍是清单候选；所有未知像素保留255，没有已核实0类。

## 下一步及当前缺口

先用独立证据复核代表性的正、负样本，区分“确认为非滑坡”和“资料不足”；对本地匹配出现异常位移或相关性不足的窗口进一步人工检查，不能自动宣布配准通过。完整核验后再形成标签版本及训练清单。

[官方V3发布页](https://doi.org/10.17603/ds2-rpyg-0h84)提供源区与堆积/运动轨迹分开的多边形，比中心点更适合检查范围；本轮验证了发布说明，但网页下载入口返回403，没有下载V3，绝不把当前存档冒充V3。

在可靠负样本缺失前不训练，也不报告准确率。若之后使用弱监督参考标签做探索性RF，应明确它检验的是与参考清单的一致性，而非道路通行准确率。
'''
(ROOT/'标签与配准核查.md').write_text(text,encoding='utf-8')
print(json.dumps({'status':report['status'],'directory':str(OUT),'samples':len(samples),
                  'usable_registration_windows':sum(r['usable_registration_windows'] for r in records),
                  'gns_omission_warning_points':sum(r['gns_centroids_over100m_from_archive_polygons_WARNING'] for r in records)},ensure_ascii=False),flush=True)
