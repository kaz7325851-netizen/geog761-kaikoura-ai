"""Write current Chinese data-preparation handoff from verified artifacts."""
from pathlib import Path
import json
from collections import Counter

ROOT=Path(__file__).resolve().parent
OUT=Path(json.loads((ROOT/'latest_v3_reference_preparation.json').read_text(encoding='utf-8'))['directory'])
data=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
validation=json.loads((OUT/'independent_validation.json').read_text(encoding='utf-8'))
assert validation['status'].startswith('GDAL_independent')
reviews=json.loads((OUT/'coarse_visual_review_all54.json').read_text(encoding='utf-8'))
total={k:sum(r[k] for r in data['regions']) for k in ['old_inventory_pixels','v3_union_pixels','v3_union_pixels_not_old','old_pixels_outside_v3','event_reference_positive_pixels','negative_proposal_pixels']}
rows=['|集合|参考正类提案像素|背景负类提案像素|合计|','|---|---:|---:|---:|']
for role,zh in [('train','训练'),('validation','验证'),('test','测试')]:
    counts=validation['role_class_counts']; positive=counts[f'{role}_class1']; negative=counts[f'{role}_class0']
    rows.append(f'|{zh}|{positive}|{negative}|{positive+negative}|')
regions=['|区域|V3足迹像素|V3新增覆盖像素（相对旧存档）|背景提案组|','|---|---:|---:|---:|']
for r in data['regions']:regions.append(f"|{r['id']}|{r['v3_union_pixels']}|{r['v3_union_pixels_not_old']}|{r['coarse_review_negative_groups']}|")
text=f'''# RF数据准备：官方V3参考范围及首版弱标签提案

日期：2026-10-08。本轮只处理数据，未训练RF、未调参、未产生模型预测。所有旧VV影像、11项特征和历史模型结果保留。

## 这次已经解决了什么

此前网页读取工具返回403，只能核对旧作者存档和官方中心点。本轮通过DesignSafe正常匿名公开下载入口成功取得最新发布目录的官方V3 GIS文件与元数据，共19个文件，压缩包34,529,362字节（约34.5MB）。18个GIS/HTML组件均与发布者SHA-512清单一致。

[官方发布记录](https://doi.org/10.17603/ds2-rpyg-0h84)及实际文件包含31,623个滑坡源区、26,559个堆积/运动轨迹多边形。文件采用NZTM2000 / EPSG:2193，与现有VV网格坐标系一致。记录声明CC BY 4.0，后续使用应引用Jones等官方数据集，且保留其不保证完整性/适用性的声明。

官方原文件保存在`{ROOT/'official_v3_acquisition/components'}`。本轮没有下载或分析CSV/DOCX，使用的是GIS属性、官方HTML元数据和发布说明。

## 新旧清单核对

8区旧清单中心点栅格共{total['old_inventory_pixels']}像素，V3源区和堆积范围联合栅格共{total['v3_union_pixels']}像素。其中{total['v3_union_pixels_not_old']}像素仅V3覆盖，{total['old_pixels_outside_v3']}像素仅旧存档覆盖。这既包括新增对象，也包括版本、范围定义及边界修订；不能简单把所有差异都当成新滑坡。

之前727个距旧存档多边形超过100m的官方V2源区中心点，本次全部距V3源区≤100m。这个100m规则仅用于位置一致性诊断，不是边界精度、漏检率或模型召回率。

'''+ '\n'.join(regions)+f'''

## 标签规则及分级

1. 正类参考：使用V3源区＋堆积足迹，在现有VV数值/近似几何筛选范围内，按4×4子采样估计像素覆盖比例≥75%形成候选正类。75%是实验性边界规则，既非官方要求也非精度保证；不再统一内缩20m，小滑坡不因没有20m内核就被整对象删除。小于一个像素或边缘混合仍可能无法可靠标注。
2. 模糊对象：官方属性中Cao等相关慢动/初始失稳、明确已有滑坡等源区另作未知处理；跨训练/验证/测试角色的同一多边形也隔离。本次没有检测到多边形跨角色。源区和堆积多边形未提供完整一一关联，不将对象ID总数等同于独立滑坡数。
3. 背景负类提案：基于54个候选的粗尺度目视复核，从植被/草地外观持续存在的点出发；要求点周围3×3像素两期SCL均为清晰陆地、VV特征可用、位于官方多边形、旧清单、官方中心点100m排除范围之外。未利用VV变化大小或任何模型预测来选负类。
4. 筛出20组背景提案，共180个像素。它们是“粗尺度未见滑坡证据”的参考背景，不是经过现场认证的绝对无滑坡、无落石或安全道路。每组相邻9个像素高度相关，不能当成9个独立地点。
5. 其余区域保留未知255。正式参考栅格`reference_positive_UNKNOWN_background.tif`仍无0类；包含0类的是单独清楚标记的`weak_labels_PROPOSAL_NOT_APPROVED.tif`，没有覆盖旧标签。

本轮将54个上下文均完成粗尺度目视筛查，但最终地面真值审批仍未完成。两个旧清单正类提案在灾前光学影像中已能看到裸地/扰动（SH1-EXP-02-R06、SH1-EXP-07-R07），不能直接视为本次新生滑坡；保留其歧义，不将它们当作非滑坡。

## 首版VV特征点数据包

已提取530个样本像素的11项已有VV/地形特征：350个官方事件参考正类提案、180个粗尺度背景负类提案。没有将Sentinel-2、官方地质/体积字段或BBUnet预测加入X。NPZ和JSON逐值与原始特征文件相符。

'''+ '\n'.join(rows)+f'''

训练/验证/测试地域仍沿用已固定的5/1/2区划分，不按像素随机拆分，不依据后续RF成绩更换区域。当前只是小样本数据流程/可行性试验的提案，类别比例是采样产生的，不代表真实滑坡发生率，也不应直接用于概率校准。245个关联源区/堆积对象ID没有跨角色复用，这不等于245个完全独立滑坡。

目录：`{OUT}`。

- `VV_feature_point_proposals_NOT_APPROVED.npz`：X、提案y、region、role、row、col和特征名，无pickle。
- 同名JSON：逐点来源和对象/背景复核组ID。
- `coarse_visual_review_all54.json`：全部54个候选的目视意见、V3对应对象、光学可用性及提案准入结果。
- `v3_source_clipped_2193.geojson`、`v3_debris_clipped_2193.geojson`：采样区附近裁切矢量，文件明确声明EPSG:2193，不能误当WGS84经纬度。
- 各区目录：新官方参考栅格、覆盖比例、未知掩膜、参考正类图、单独的弱标签提案图。
- `independent_validation.json`：另一套GDAL矢量读取路径独立复算了8区源区/堆积栅格及4×4覆盖比例，全部一致；所有旧VV/特征文件哈希未变。

## 仍然存在的限制

- 官方清单是推断由主震及余震触发的事件滑坡。它没有逐对象11月15日前/后的发生时间，因此不能把全部足迹当作11月15日即时状态真值。
- 标签审查所用S2灾前是11月2日UTC、灾后是11月22日UTC，灾后比VV11月15日晚7天；其辅助身份不改变VV主模型方案。
- V3发布说明指出仍未包含所有小滑坡（源区通常<500平方米），清单外、光学未见改变也不能保证无小滑坡。
- 可用对齐诊断只是部分固定窗口，尚未完成全区几何配准认证或辐射地形校正。原始几何质量掩膜仍是近似，不能保证全部道路可见。
- 几何读取出现过一个孤立环方向警告；GDAL独立复算在8区没有像素分歧，但一致不等于真实边界已被现场验证。
- 只代表已知8个SH1采样区，不是完整道路网络，测试区此前已被其他模型观察，不能称盲测或跨事件验证。
- 正类足迹不能直接等同道路受阻，背景负类也不等同道路可通行。

## 下一步

数据流程已形成一版明确分级的弱参考提案。如果继续训练，应先明确接受“官方事件清单＋粗尺度背景”为弱标签的探索性RF测试，结果只能说明当前样本与参考标签的可分性。先用固定的地域留出验证，并按复核组/对象而非仅像素解释结果；不能把它升级成可靠道路通行率。

若要求验证11月15日真实道路状态，则仍需同期影像/记录及道路阻断证据，不能仅靠本提案包得出结论。本轮保持`training_ready=false`（正式验证条件未满足），不启动训练。
'''
(ROOT/'官方V3标签准备进展.md').write_text(text,encoding='utf-8')
print(json.dumps({'report':str(ROOT/'官方V3标签准备进展.md'),'totals':total,
                  'weak_background_groups':data['weak_negative_proposal_groups'],'sampled_pixels':sum(validation['class_counts'].values())},ensure_ascii=False,indent=2))
