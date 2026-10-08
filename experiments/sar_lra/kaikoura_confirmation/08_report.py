"""Write evidence-based Chinese handoff from verified saved outputs."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
out=Path(json.loads((ROOT/'latest_result.json').read_text(encoding='utf-8'))['result_directory'])
r=json.loads((out/'summary.json').read_text(encoding='utf-8'))
v=json.loads((out/'verification.json').read_text(encoding='utf-8'));assert v['status']=='PASS'
sites={s['id']:s for s in r['sites']}
rank={}
for sid in sites:
    d=np.load(out/sid/'predictions.npz');p=d['scores'][d['positive']];b=d['scores'][d['background']]
    # Threshold-free descriptive separation; no calibration/threshold fitting.
    rank[sid]=float(np.mean((p[:,None]>b[None,:])+0.5*(p[:,None]==b[None,:]))) if len(p) and len(b) else None
diagnostic={'rank_AUROC_on_curated_reference_samples':rank,'threshold_changed':False,
 'interpretation':'Descriptive within-site score ranking only; overlapping windows and label-stratified selection preclude general accuracy claims.'}
(out/'score_ranking_diagnostic.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
primary=[sites[sid] for sid in ['A','B','C']]
positive=sum(s['metrics']['all']['positive_windows'] for s in primary)
nonoverlap=sum(s['metrics']['nonoverlap']['positive_windows'] for s in primary)
bg=sum(s['metrics']['all']['reference_background_windows'] for s in primary)
support=[s for s in primary if s['metrics']['all']['positive_windows']]
stop=all(s['metrics']['all']['recall']<=.1 for s in support)
decision={'stop_condition_satisfied':stop,'decision':'Do not expand unchanged configuration to full Kaikoura',
 'new_site_positive_windows':positive,'new_site_nonoverlap_positive_windows':nonoverlap,
 'new_site_reference_background_windows':bg,'new_site_true_positives':0,'all_new_site_predicted_positives':0,
 'original_checkpoint_is_not_globally_disproven':True,'next_work_requires_explicit_request':True}
(out/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
roles={'A':'较大滑坡区','B':'道路附近滑坡对照区','C':'清单稀疏背景区','OLD_GEO':'旧区经纬度网格对照'}
lines=['# SAR-LRA 凯库拉第二轮有限验证报告','',
 '2026-10-05。本轮按用户同意的流程复核输入，固定三个不重叠新区，并增加旧区网格对照；实际下载影像、运行作者真实权重，独立核验通过。所有文件保存在 testdata/sar_lra/kaikoura_confirmation，不覆盖前轮结果。','',
 '## 结论','',
 f'三个新区共 {sum(s["valid_scored_windows"] for s in primary)} 个有效预测窗口；{positive} 个清洗后正类窗口全部漏检。只看不重叠64像素网格子集，{nonoverlap} 个正类窗口仍全部漏检。这不是 {positive} 或 {nonoverlap} 处独立滑坡。',
 '测试前冻结的停止条件为：每个有正类支持的区域召回率仍≤10%，则停止当前配置的盲目全域扩展。本轮 A、B 均为0%，已触发该条件。建议不把现有 VV+VH 权重＋82天前单日期基线＋0.6阈值方案继续直接扩展到全凯库拉，也不将“未检测到滑坡”解释为道路安全。',
 '', '## 实际结果','',
 '|区域|有效评分窗口|清洗后正类|检出/漏检|参考背景误报|正类召回率|',
 '|---|---:|---:|---|---|---|']
for sid,s in sites.items():
    m=s['metrics']['all'];rec='不适用（无正类）' if m['recall'] is None else f'{m["recall"]*100:.1f}%'
    lines.append(f'|{sid}：{roles[sid]}|{s["valid_scored_windows"]}|{m["positive_windows"]}|{m["TP"]}/{m["FN"]}|{m["FP"]}/{m["reference_background_windows"]}|{rec}|')
lines += ['',f'新区共有 {bg} 个符合本轮筛选规则的参考背景窗口，均未报正类。由于所有有效窗口都低于0.6，“零误报”不能证明有效：模型没有同时完成检出正类的任务。背景也并非经实地确认无滑坡或道路安全。',
 '', '## 这轮排查了哪些问题','',
 '1. 波段与公式：固定顺序 postVV、postVH、diffVV、diffVH；输入是 GEE dB 数据，无额外标准化或辐射坡度校正。逐区核验 diffVV=postVV−preVV、diffVH=postVH−preVH，有限值和有效覆盖通过。后面的地形/标签通道没有输入模型。',
 '2. 权重和运行：同一作者 ASCENDING 检查点，SHA256 '+r['checkpoint_sha256']+'；严格拓扑加载，各张量吻合，预测后权重不变，没有训练。',
 '3. 回放对照：各抽取18个此前苏门答腊、凯库拉保存窗口重新预测，两组与原始分数最大差都为0，类别也相同。未发现此次运行环境/加载改变导致突然失效的证据。这仍不等于证明完全复现作者原训练环境。',
 '4. 标签清洗：同三景ASC52真实入射角、公开 Vollrath 局部活跃阴影/叠掩算法、SRTM30、0m默认缓冲。灾前/灾后都需可见；多个参与灾后切片任一存在活跃畸变则保守剔除。不是作者原始掩膜的精确复现，也未射线追踪非局部被动畸变。正类仍按可见滑坡像素/4096>5%，没有改分母。',
 '5. 网格对照：旧区由 NZTM10m 方格改为作者公开部署使用的 EPSG4326 名义10m网格，在同一影像/权重/阈值下，13个正类窗口仍全部漏检。不能仅用“原网格不同”解释失效。但经纬度像元在42°纬度的地面东西宽度更窄，64像素实际覆盖范围不同，所以13与旧区10不能逐个配对，不能声称是严格同物理窗口的投影消融。',
 '6. 滑坡大小对照：A区正类的滑坡占比中位数18.38%，最高94.17%；A、B合计36个滑坡占比>15%的正类窗口仍全部漏检。因此失效不局限于刚超过5%门槛的小占比窗口。道路500m缓冲附近的正类窗口 A5、B34也均漏检，但道路邻近≠实际道路受损。',
 '', '## 分数还有没有信息','',
 f'固定0.6阈值下所有区均无预测正类；不能因此声称所有分数完全随机。仅作不调阈值的排序诊断，A区参考样本AUROC约{rank["A"]:.3f}，B区约{rank["B"]:.3f}。B区仍存在一定区分信号，但A区没有呈现稳定正类排序优势，说明跨区不稳。此诊断没有改变预测类别，也未用它选择新阈值。',
 f'B区正类最高分{sites["B"]["positive_score_quantiles"][-1]:.3f}接近0.6，但A区正类最高仅{sites["A"]["positive_score_quantiles"][-1]:.3f}，A区参考背景最高{sites["A"]["background_score_quantiles"][-1]:.3f}。不能靠把阈值略降一点就声称整体问题解决。若未来做本地阈值校准，需另设校准区和独立留出测试区；不应拿本轮测试标签直接调到成绩好看。',
 '', '## 区域和背景如何选择','',
 '区域使用公开滑坡清单分层筛选，冻结在新影像和预测之前，不是随机代表性样本。A：现有道路10km内最大清单多边形，避开旧区。B：沿道路每2km建立候选区，以不重叠参考网格中道路500m附近5–15%滑坡占比窗口最多者入选。C：剩余不重叠候选中原始清单占比最小者，优先较西侧道路中心。70个候选及排序规则完整保存在 frozen_design.json。没有看预测后换区或删除失败区。',
 '每区6.4×6.4km，共122.88km²新区；A、B、C互不重叠，且不与旧试验区重叠。这不是团队最终研究区。道路为已有当前OSM线，未独立验证2016位置。',
 '参考背景要求：原始滑坡像素为0，窗口内无距原清单200m以内区域，≥95%像素为DEM>1m陆地代理，≥95%像素通过局部活跃可见性规则。含少量滑坡、被清洗掉的正类或不符合背景条件的窗口保留预测，但不冒充可靠真负类。陆地代理也不是精确海岸线或永久水体验证。',
 '', '## 尚未解决的限制','',
 '灾前仍是2016-08-23单日期（约82天前），灾后2016-11-15（约44小时后）；没有近地震双极化基线，不是原60天多日期中值方案。此次不能确定究竟是时间基线、单日期噪声、区域差异或其他训练/输入分布差异导致失败。',
 'VV单极化检查点仍未获得。论文凯库拉属于VV数据方案，不能把本轮VV+VH结果推广成论文VV模型也失败。苏门答腊初测成功也不能自动证明凯库拉适用。',
 '窗口有重叠；即使取不重叠子集，也可能来自同一滑坡、同一局部地形，不是独立事件样本。指标仅用于本轮固定诊断，不是凯库拉全域准确率。模型结果是粗略区域分类，不是滑坡精确边界、道路通行或海滩登陆判定。',
 '', '## 文件和复现','',
 '当前结果目录：`'+str(out)+'`。','',
 '- frozen_design.json：测试前固定的区域、参考分层规则、阈值及停止条件。',
 '- input_audit.json：两轮输入数值、网格、差值与权重检查。',
 '- downloads/：四组实际GEE GeoTIFF与哈希、元数据。',
 '- results/当前时间戳/summary.json、verification.json、decision.json：实测和独立核验。',
 '- 每区 windows.csv、predictions.npz：所有有效窗口分数和标签；含模糊窗口，不只保留用于指标的样本。',
 '- 每区 clean_reference.tif、visible_mask.tif 等：地理对齐的参考掩膜；detections.geojson为原V1 NMS之后的疑似框，本轮全部为空。',
 '- comparison.png：四组dVV、标签及窗口判定图，蓝色为漏检，浅黄色为评价类别不确定，白色为无完整观测。',
 '源代码01–08均保存；05_download.py需有效像素下载URL，06_evaluate.py和07_verify.py可使用已有本地数据离线复现，重跑预测会建立新时间戳结果目录。中间重复运行的旧结果也保留。',
 '', '[作者部署源代码](https://github.com/lorenzonava96/SAR-and-DL-for-Landslide-Rapid-Assessment)',
 '[作者论文](https://gmd.copernicus.org/articles/19/167/2026/gmd-19-167-2026.html)',
 '[ESA公开几何掩膜方法](https://cin.philab.esa.int/databases/projects/angular-based-radiometric-slope-correction-of-sentinel-1)']
(ROOT/'凯库拉第二轮验证报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'报告快照.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'report':str(ROOT/'凯库拉第二轮验证报告.md'),'decision':decision,'score_ranking':diagnostic},ensure_ascii=False,indent=2))
