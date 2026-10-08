"""Independent readback verification; no fitting or source modification."""
from pathlib import Path
import json
import hashlib
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent
OUT = Path(json.loads((ROOT/'latest_prepared_inputs.json').read_text(encoding='utf-8'))['directory'])
m = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
for name, expected in m['source_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
for record in m['outputs']:
    assert hashlib.sha256(Path(record['file']).read_bytes()).hexdigest() == record['sha256']
    with rasterio.open(record['file']) as s:
        assert s.shape == (640,640) and s.crs.to_epsg() == 2193
        assert list(s.descriptions) == record['bands'] and s.nodata == record['nodata']
        assert np.allclose(list(s.transform)[:6], m['transform'])
def read(name):
    with rasterio.open(OUT/name) as s:
        return s.read()
a = read('VV_preMedian_post_raw.tif')
v = read('data_validity_mask.tif')[0] == 1
f = read('VV_feature_candidates.tif')
fv = read('feature_numerical_validity_mask.tif')[0] == 1
labels = read('candidate_labels_NOT_TRAIN_READY.tif')[0]
post_visible = read('post_geometry_approx_mask.tif')[0] == 1
inventory = read('inventory_reference_only.tif')[0] == 1
assert np.allclose(a[2,v], a[1,v]-a[0,v], atol=2e-5)
assert (a[:,~v] == -9999).all()
assert np.isfinite(f[:,fv]).all() and (f[:,~fv] == -9999).all()
assert np.allclose(f[2,fv], f[1,fv]-f[0,fv], atol=2e-5)
assert np.allclose(f[3,fv], np.abs(f[2,fv]), atol=2e-5)
assert np.allclose(f[9,fv]**2+f[10,fv]**2, 1, atol=1e-6)
assert set(np.unique(labels)) == {1,255}
assert np.array_equal(labels==1, inventory & fv & post_visible)
assert int(v.sum()) == m['data_valid_pixels']
assert int(fv.sum()) == m['feature_numerical_valid_pixels']
assert int((labels==1).sum()) == m['candidate_label_pixels']
assert m['training_ready'] is False and m['verified_negative_pixels'] == 0
check = {'status':'PASS for local numerical/artifact consistency, NOT training readiness or accuracy',
         'checks':['source/output hashes','exact grids','nodata and band names','correct recent-median difference',
                   'finite features','aspect encoding','labels stay1/255 and contain no invented negatives'],
         'output_count':len(m['outputs']), 'data_coverage_pct':float(v.mean()*100),
         'feature_numerical_coverage_pct':float(fv.mean()*100),'training_ready':False}
(OUT/'independent_validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
report = f'''# Sentinel-1 VV 数据准备进展

日期：2026-10-08。状态：第一份本地临时输入包已完成并校验；新单日期影像获取受连接故障阻挡；尚未达到训练条件。

## 已完成

- 用户指定Sentinel-1灾前/灾后VV为主方案；Sentinel-2仅保留备选，不作为这版RF输入。
- 复用实际本地40.96平方公里试验区：灾前是9月16日、10月10日、11月3日VV的dB空间中值，灾后为11月15日。两者均升轨52、Sentinel-1A。
- 重新计算灾后减近期灾前合成的dVV，没有复用8月23日基线对应的旧diffVV。
- 输出原始VV三波段、11项候选特征、数据/特征有效性掩膜、仅灾后局部几何近似掩膜、道路100m邻近掩膜和临时参考标签。
- 数据存在覆盖{check['data_coverage_pct']:.3f}%；扣除纹理邻域边缘和数值缺失后的特征覆盖{check['feature_numerical_coverage_pct']:.3f}%。这不是道路可观测率或模型精度。
- {m['candidate_label_pixels']}个像素保留为清单滑坡候选；没有任何已核实负类。其余标签为未知255，不是无滑坡0。
- 独立读回校验通过：原始文件和输出哈希、网格、波段名称、nodata、差值公式、有限特征、坡向编码及标签类别一致。

本地输入包：`{OUT}`。

## 新数据的获取状态

已生成`02_export_vv.js`，拟获取11月3日/11月15日同轨VV，并在之前固定的7个SH1方区及光学试验锚点中检查数据，共8区327.68平方公里。这些只是采样方区，不是团队正式完整研究区；原选区受光学瓦片范围限制，因此不能声称事件全域代表性。

Earth Engine页面已显示原账户，但客户端提示无法连接earthengine.googleapis.com。进入编辑器后，只读集合查询报`Cannot set properties of undefined (setting 'toString')`；尝试刷新没有恢复可用查询。新8区导出脚本尚未在服务器成功运行，也没有新的VV单日期下载。

这是平台连接/客户端初始化问题，不是目录核查发现11月3日VV不存在。没有获取浏览器凭证，没有创建账户/密钥，没有绕过安全设置。

## 尚未完成，不能开始正式训练

1. 11月3日单日期VV及8区栅格获取。
2. 近期灾前雷达几何核查、局部配准检查；现有8月23日灾前几何不能代替近期日期。
3. 候选滑坡的事件日期和边界核验，以及独立确认的非滑坡样本。
4. 空间训练/验证/测试划分。

当前仅做了灾后局部活跃阴影/叠掩近似核查，没有被动畸变射线追踪、完整可见性证明或辐射地形平坦化。DEM约30m；10m输出网格不增加真实地形细节。

3×3线性功率均值滤波和5×5局部标准差仅作为可复现的第一版特征方案，可能模糊小滑坡，尚未通过模型性能验证。特征数值有效不等于地物可见。

QA图中海面有大面积变化，说明变化量也会响应非滑坡因素，不能把所有变化都标为滑坡。当前道路100m缓冲仅表示邻近位置，不表示道路阻断。

## 文件入口

- `manifest.json`：来源、日期、处理参数、输出哈希及限制。
- `independent_validation.json`：独立读回核验。
- `local_input_QA.png`：灾前合成、灾后、变化量及候选标签示意。
- `candidate_labels_NOT_TRAIN_READY.tif`：1为清单候选，255为未知；没有负类0。

没有训练RF、运行新预测、删除或覆盖已有测试结果。

参考：[Earth Engine Sentinel-1预处理说明](https://developers.google.com/earth-engine/guides/sentinel1)。
'''
(ROOT/'数据准备进展.md').write_text(report,encoding='utf-8')
print(json.dumps(check,ensure_ascii=False,indent=2))
