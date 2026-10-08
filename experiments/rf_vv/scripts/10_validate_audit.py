"""Independent readback invariants; no predictions or label approval."""
from pathlib import Path
from collections import Counter
import json, hashlib
import numpy as np
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parent
OUT=Path(json.loads((ROOT/'latest_label_alignment_audit.json').read_text(encoding='utf-8'))['directory'])
PREP=Path(json.loads((ROOT/'latest_prepared_regions.json').read_text(encoding='utf-8'))['directory'])
summary=json.loads((OUT/'audit_summary.json').read_text(encoding='utf-8'))
prepared=json.loads((PREP/'manifest.json').read_text(encoding='utf-8'))
design=json.loads((ROOT/'frozen_data_design.json').read_text(encoding='utf-8'))
split=json.loads((ROOT/'spatial_split_v1.json').read_text(encoding='utf-8'))
samples=json.loads((OUT/'review_samples_UNREVIEWED.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with rasterio.open(p) as s:return s.read(1),(s.shape,s.transform,s.crs),s.nodata
assert sha(ROOT/'spatial_split_v1.json')==summary['split_sha256']
assert sha(PREP/'manifest.json')==summary['prepared_manifest_sha256']
regions={s['id']:box(*s['bounds']) for s in design['sites']}
group={k:unary_union([regions[r] for r in regions if split['assignments'][r]==k]) for k in ['train','validation','test']}
assert Counter(split['assignments'].values())=={'train':5,'validation':1,'test':2}
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert group[a].distance(group[b])==split['minimum_aoi_gap_m'][f'{a}_to_{b}_m']
    assert group[a].distance(group[b])>=1000
gnsmanifest=json.loads((ROOT/'official_reference_points/manifest.json').read_text(encoding='utf-8'))
gns=json.loads(Path(gnsmanifest['file']).read_text(encoding='utf-8'))
assert sha(gnsmanifest['file'])==gnsmanifest['sha256']
assert len(gns['features'])==gnsmanifest['count']==len(set(f['attributes']['objectid'] for f in gns['features']))
verified=[];peaks=Counter();inventory_total=0;core_total=0;warn_total=0
for r in prepared['records']:
    rid=r['id']; dest=OUT/rid; rec=json.loads((dest/'audit.json').read_text(encoding='utf-8'))
    assert sha(r['raw_file'])==r['raw_sha256']==rec['source_hashes']['raw']
    for output in r['outputs']:assert sha(output['file'])==output['sha256']
    inv,grid,_=read(PREP/rid/'inventory_reference_only.tif')
    screened,_,_=read(PREP/rid/'screened_feature_mask.tif')
    masks={}
    for name in ['inventory_interior20m_CANDIDATE','inventory_buffer100m',
                 'gns_source_point_buffer100m','negative_review_candidates_NOT_LABELS']:
        a,g,n=read(dest/f'{name}.tif');assert g==grid and n is None and np.isin(a,[0,1]).all();masks[name]=a==1
    core=masks['inventory_interior20m_CANDIDATE']; neg=masks['negative_review_candidates_NOT_LABELS']
    assert not (core&((inv!=1)|(screened!=1))).any()
    assert not (neg&(masks['inventory_buffer100m']|masks['gns_source_point_buffer100m']|(screened!=1))).any()
    labels,g,n=read(dest/'provisional_core_labels_NOT_TRAIN_READY.tif')
    assert g==grid and n==255 and np.isin(labels,[1,255]).all()
    assert np.array_equal(labels==1,core)
    assert int(core.sum())==rec['positive_core20m_candidate_pixels']
    assert int(neg.sum())==rec['negative_review_candidate_pixels_NOT_verified0']
    windows=rec['registration_windows'];assert len(windows)==16
    usable=[m for m in windows if m['status']=='usable_diagnostic_peak']
    assert len(usable)==rec['usable_registration_windows']
    for m in usable:
        assert m['best_ncc']>=.55 and m['peak_margin_outside1px']>=.03
        assert max(map(abs,m['best_post_correction_px']))<3
        peaks[tuple(m['best_post_correction_px'])]+=1
    for s in [s for s in samples if s['region_id']==rid]:
        y,x=s['row'],s['col'];assert s['role']==split['assignments'][rid]
        assert s['review_status']=='UNREVIEWED' and s['final_label'] is None
        assert (neg if s['kind'].startswith('negative') else core)[y,x]
    inventory_total+=int(inv.sum());core_total+=int(core.sum())
    warn_total+=rec['gns_centroids_over100m_from_archive_polygons_WARNING']
    verified.append({'id':rid,'readback_invariants':'PASS'})
assert len(set(s['id'] for s in samples))==len(samples)==summary['review_sample_count']
result={'status':'independent_artifact_invariants_PASS_NOT_label_truth_or_registration_certification',
        'regions':verified,'original_raw_and_feature_hashes_unchanged':True,
        'official_download_count_broad_envelope':gnsmanifest['count'],
        'warning_point_count_inside8regions':warn_total,'raw_inventory_pixels':inventory_total,
        'core20m_pixels':core_total,'core_retained_pct':100*core_total/inventory_total,
        'usable_registration_windows':sum(peaks.values()),'windows_total':128,
        'integer_post_correction_counts':{str(k):v for k,v in sorted(peaks.items())},
        'review_proposal_counts':dict(Counter(s['kind'] for s in samples)),
        'verified_negative_pixels':0,'training_ready':False,'training':False}
(OUT/'independent_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
addition=f'''

## 本轮核查的关键发现（独立读回确认）

- GNS官方大包围框下载了{gnsmanifest['count']}个中心点，不是全部都落在8个采样区。8区内有{warn_total}个中心点距离作者存档清单多边形超过100m，提示版本、遗漏或空间定义差异，需要逐一核实，不能把它们直接扩成滑坡范围，也不能把点数当作模型漏检数。
- 20m向内收缩后保留{core_total}/{inventory_total}个清单像素，约{result['core_retained_pct']:.1f}%。SH1-EXP-02/03仅各剩1个像素，因此该规则过于排斥小滑坡，不能直接作为最终训练标签。原清单、原候选标签没有修改；内核只是保守性敏感度诊断。
- 128个固定窗口中64个满足本次诊断可靠性阈值，其中56个最佳整数偏移为0，8个为1像素（10m输出网格）。其余窗口不能判定。没有应用位移，也不能从这个整数搜索推断所有区域误差≤10m或亚像素对齐通过。
- 已生成{len(samples)}个待复核点：{result['review_proposal_counts'].get('negative_review_proposal',0)}个负类提案、{result['review_proposal_counts'].get('positive_inventory_review_proposal',0)}个清单正类提案，尚未批准最终标签。
- 所有原始VV与已有11波段特征文件哈希未变，新输出读回检查通过。标签真值和全域配准仍需独立证据。

建议下一步优先补充完整滑坡多边形（源区及堆积范围），核查与11月15日影像的时间一致性，再复核非滑坡样本；不要现在就把这套待核验标签送入RF。若完整多边形暂不可得，必须将后续实验明确标成弱监督探索，不报告道路通行准确率。
'''
report=ROOT/'标签与配准核查.md'
existing=report.read_text(encoding='utf-8')
marker='\n\n## 本轮核查的关键发现（独立读回确认）'
report.write_text(existing.split(marker)[0]+addition,encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
