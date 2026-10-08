"""Independent artifact readback and a current progress report, without training."""
from pathlib import Path
import hashlib
import json
import numpy as np
import rasterio

ROOT=Path(__file__).resolve().parent
OUT=Path(json.loads((ROOT/'latest_prepared_regions.json').read_text(encoding='utf-8'))['directory'])
m=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
download_path=Path(m['download_manifest_file'])
assert hashlib.sha256(download_path.read_bytes()).hexdigest()==m['download_manifest_sha256']
download=json.loads(download_path.read_text(encoding='utf-8'))
assert len(m['records'])==8
assert download['common']['pre_date_utc']=='2016-11-03'
assert download['common']['post_date_utc']=='2016-11-15'
assert all('20161103' in s for s in download['common']['ids']['pre'])
assert all('20161115' in s for s in download['common']['ids']['post'])
rows=[]
def read(p):
    with rasterio.open(p) as s:
        return s.read(), (s.shape,s.transform,s.crs)
for r in m['records']:
    rid=r['id']; dest=OUT/rid
    assert hashlib.sha256(Path(r['raw_file']).read_bytes()).hexdigest()==r['raw_sha256']
    for key in ['reference','road']:
        assert hashlib.sha256(Path(r[key+'_file']).read_bytes()).hexdigest()==r[key+'_sha256']
    raw,grid=read(r['raw_file'])
    assert grid[0]==(640,640) and grid[2].to_epsg()==2193
    for record in r['outputs']:
        p=Path(record['file']); assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
        with rasterio.open(p) as s:
            assert (s.shape,s.transform,s.crs)==grid
            assert list(s.descriptions)==record['bands'] and s.nodata==record['nodata']
    f,_=read(dest/'VV_feature_candidates.tif')
    valid=read(dest/'data_validity_mask.tif')[0][0]==1
    numerical=read(dest/'feature_numerical_validity_mask.tif')[0][0]==1
    geom=read(dest/'local_geometry_approx_mask.tif')[0][0]==1
    screened=read(dest/'screened_feature_mask.tif')[0][0]==1
    labels=read(dest/'candidate_labels_NOT_TRAIN_READY.tif')[0][0]
    inventory=read(dest/'inventory_reference_only.tif')[0][0]==1
    assert np.array_equal(valid,(raw[12]==1)&(raw[13]==1)&np.isfinite(raw[:3]).all(axis=0)&(raw[:3]!=-9999).all(axis=0))
    assert np.allclose(raw[2,valid],raw[1,valid]-raw[0,valid],atol=2e-5)
    mv=valid&(raw[14]==1)&(raw[3]!=-9999)&(raw[4]!=-9999)
    assert np.allclose(raw[4,mv],raw[1,mv]-raw[3,mv],atol=2e-5)
    assert np.array_equal(geom,(raw[15]==1)&(raw[16]==1))
    assert np.array_equal(screened,numerical&geom)
    assert np.isfinite(f[:,numerical]).all() and (f[:,~numerical]==-9999).all()
    assert np.allclose(f[2,numerical],f[1,numerical]-f[0,numerical],atol=2e-5)
    assert np.allclose(f[3,numerical],np.abs(f[2,numerical]),atol=2e-5)
    assert np.allclose(f[9,numerical]**2+f[10,numerical]**2,1,atol=1e-6)
    assert set(np.unique(labels))=={1,255}
    assert np.array_equal(labels==1,inventory&screened)
    assert int((labels==1).sum())==r['candidate_label_pixels']
    rows.append({'id':rid,'status':'PASS readback/grid/hash/mask/formula checks','training_ready':False})
check={'status':'all8regions_pass_independent_numerical_artifact_checks',
       'regions':rows,'full_geometric_visibility_proven':False,
       'negative_labels_verified':False,'training_ready':False,
       'trained':False,'optical_predictor_used':False}
(OUT/'independent_validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
total_bytes=sum(r['bytes'] for r in download['records'])
total_pixels=sum(r['data_valid_pixels'] for r in m['records'])
pooled_data=100*total_pixels/(8*640*640)
screened_mean=np.mean([r['screened_feature_coverage_pct'] for r in m['records']])
lines=['# Sentinel-1 VV 数据准备进展（已恢复连接）','',
       '更新日期：2026-10-08。Earth Engine新标签页正常初始化，真实服务器查询及8区数据导出成功。此前连接故障已不再阻挡本轮下载。','',
       '## 本轮实际完成','',
       '- 下载11月3日灾前、11月15日灾后Sentinel-1A升轨52的VV；每个日期包含同一次过境的两个相邻切片，不是两次拍摄。',
       '- 同时保留9月16日、10月10日、11月3日三日期灾前中值作为单独标明的备选基线；本次主特征包使用11月3日单日期。',
       '- 8个之前固定的SH1采样方区，总面积327.68平方公里。不是完整SH1/Route70，也不是团队已确认的全部正式研究区。',
       f'- 实际下载8份17波段GeoTIFF，总计{total_bytes:,}字节（约{total_bytes/1_000_000:.2f} MB），均已校验。',
       '- 整理8套VV特征包：灾前/灾后后向散射、变化/绝对变化、局部纹理，以及地形等11项候选特征。没有输入Sentinel-2影像，也没有使用BBUnet预测作为特征或标签。',
       '- 保存灾前后数据掩膜、局部活跃阴影/叠掩近似掩膜、数值特征掩膜、道路100m邻近掩膜和清单候选参考标签。',
       '- 独立读回核验全部通过：来源/输出哈希、网格、波段映射、nodata、两个原始变化量、特征差值、坡向编码、掩膜和候选标签一致。','',
       f'原始下载：`{download_path.parent}`。',f'整理后的输入包：`{OUT}`。','',
       '## 数据覆盖（不是精度或道路可通行率）','',
       '|采样区|灾前后有数值覆盖|数值特征＋局部几何近似筛选覆盖|清单候选像素|',
       '|---|---:|---:|---:|']
for r in m['records']:
    lines.append(f"|{r['id']}|{r['data_coverage_pct']:.3f}%|{r['screened_feature_coverage_pct']:.3f}%|{r['candidate_label_pixels']}|")
lines += ['',f'合并数据存在覆盖：{pooled_data:.3f}%；经过特征邻域边缘和局部几何近似筛选后：{screened_mean:.3f}%。',
          '', '特征计算排除了方区边缘及缺测邻域；降低的比例不全是雷达不可见地形。局部几何筛选不是完整可见性证明。','',
          '## 标签和训练尚未完成','',
          '- 清单源为此前已保存的作者存档Kaikora_2016原始多边形，并非NZTA道路封闭记录。复用BBUnet目录中的原始清单栅格和道路掩膜，仅为避免重复栅格化，不复用光学预测或云掩膜。',
          '- 标签1仅表示清单中的滑坡候选；255表示未知。没有已核实的负类0，也不把清单外自动当成无滑坡。',
          '- 仍需核查滑坡是不是本次地震新增/重新活动、范围是否适当，以及建立有证据支持的非滑坡样本。',
          '- 尚未固定训练/验证/测试空间划分，也没有训练RF或产生新模型预测。','',
          '## 必须保留的限制','',
          '- 影像同网格不等于局部配准已经验证。下一轮需核查稳定地物位置及差值伪影。',
          '- 几何掩膜仅为SRTM约30m地形支持的局部活跃阴影/叠掩近似，没有被动畸变射线追踪、误差缓冲或辐射地形平坦化。',
          '- 3×3线性功率均值滤波、5×5局部标准差是第一版实验特征，可能模糊小滑坡，不能声称已经优化。',
          '- 采样方区沿用以前受光学瓦片范围约束的选区，且以前的清单分布和光学测试结果已知，不应称盲测或事件全域抽样。',
          '- 道路100m缓冲表示邻近关系，不能直接等同道路受阻；当前OSM路网也未完成2016年历史拓扑核验。','',
          '## 下一步','',
          '先核查正/负参考样本和局部配准，明确保留/未知区域，再固定空间留出划分。满足这些条件后才开始RF训练。','',
          '旧40.96平方公里三日期中值临时包保留在prepared_inputs目录，没有删除、覆盖或把它重新描述为单日期影像。']
(ROOT/'数据准备进展.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':check['status'],'download_bytes':total_bytes,'data_coverage_pct':pooled_data,
                  'screened_feature_coverage_pct':float(screened_mean),'regions':8,'training_ready':False},ensure_ascii=False,indent=2))
