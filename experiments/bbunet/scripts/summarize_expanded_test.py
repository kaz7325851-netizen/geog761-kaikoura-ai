"""Readable Chinese report and layout-checked scientific diagnostic figures."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'expanded_test'/'20261007T210924Z'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'/'matplotlib'))
import numpy as np,rasterio,shapefile
from pyproj import CRS,Transformer
from shapely.geometry import shape
from shapely.ops import transform
from rasterio.features import rasterize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads((OUT/'expanded_results.json').read_text())
assert r['status']=='expanded_test_complete_independently_rechecked'
def read(p):
    with rasterio.open(p) as d:return d.read()
fig,axs=plt.subplots(2,4,figsize=(20,12),layout='constrained')
fig.get_layout_engine().set(hspace=.12,wspace=.025)
for ax,x in zip(axs.flat,r['regions']):
    d=OUT/x['region_id'];rgb=np.clip(read(d/'post_normalized.tif')[[3,2,1]].transpose(1,2,0)/.3,0,1)
    v=read(d/'evaluation_mask.tif')[0].astype(bool);p=read(d/'probability_full.tif')[0]
    ref=read(d/'inventory_raster.tif')[0];road=read(d/'road100m_mask.tif')[0]
    ax.imshow(rgb);layer=np.zeros((*v.shape,4));layer[v&(p>=.5)]=[1,0,0,.8];ax.imshow(layer)
    if ref.any():ax.contour(ref,[.5],colors=['yellow'],linewidths=.4)
    ax.contour(road,[.5],colors=['cyan'],linewidths=.4)
    n=x['objects']['near_road100m'];den=n['evaluated_objects'];score=f"{n['ge10pct']}/{den}" if den else 'N/A (no reference objects)'
    ax.set_title(f"{x['region_id']} | >=10% overlap: {score}\nRoad quality retained: {x['quality_fraction_road100m']:.1%}",fontsize=11,pad=9)
    ax.set_axis_off()
fig.suptitle('BBUnet: 7 NEW frozen SH1 regions + original-pilot CONTROL (excluded from new scores)\nRed: screened predictions | yellow: landslide inventory | cyan: road context | NOT road passability',fontsize=15)
fig.savefig(OUT/'expanded_overview.png',dpi=150);plt.close(fig)

# Diagnostic cases fixed AFTER scoring to illustrate successes/failures, not for selection or aggregate scores.
cases=[('SH1-EXP-04',2593,'Detected large reference'),('SH1-EXP-04',2597,'Detected small reference'),
       ('SH1-EXP-02',4884,'Missed small reference'),('SH1-EXP-07',13006,'Missed small reference')]
refpath=ROOT.parent/'sar_lra'/'kaikoura_exploration'/'reference'/'Kaikora_2016.shp'
reader=shapefile.Reader(str(refpath));proj=Transformer.from_crs(CRS.from_wkt(refpath.with_suffix('.prj').read_text()),2193,always_xy=True).transform
fig,axs=plt.subplots(4,3,figsize=(12,15),layout='constrained');notes=[]
for row,(rid,idx,label) in enumerate(cases):
    d=OUT/rid;g=transform(proj,shape(reader.shape(idx).__geo_interface__))
    if not g.is_valid:g=g.buffer(0)
    with rasterio.open(d/'probability_full.tif') as ds:
        p=ds.read(1);aff=ds.transform;rr,cc=ds.index(g.centroid.x,g.centroid.y)
    top=max(0,min(640-64,rr-32));left=max(0,min(640-64,cc-32));sl=(slice(top,top+64),slice(left,left+64))
    pm=rasterize([(g,1)],out_shape=(640,640),transform=aff,dtype='uint8')
    v=read(d/'evaluation_mask.tif')[0].astype(bool)
    obj=next(x for x in json.loads((d/'inventory_objects.json').read_text()) if x['source_index']==idx)
    for col,date in enumerate(['pre','post']):
        rgb=np.clip(read(d/f'{date}_normalized.tif')[[3,2,1]].transpose(1,2,0)/.3,0,1)
        axs[row,col].imshow(rgb[sl],interpolation='nearest');axs[row,col].contour(pm[sl],[.5],colors=['yellow'],linewidths=1)
        axs[row,col].set_title(f'{date.upper()} | {rid} / ID {idx}',fontsize=10)
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#888888')
    im=axs[row,2].imshow(np.ma.masked_where(~v[sl],p[sl]),vmin=0,vmax=1,cmap=cmap,interpolation='nearest')
    axs[row,2].contour(pm[sl],[.5],colors=['cyan'],linewidths=1)
    axs[row,2].set_title(f"{label}\nArea {obj['source_area_m2']:.0f} m2 | overlap {obj['coverage_fraction']:.1%}",fontsize=10)
    fig.colorbar(im,ax=axs[row,2],fraction=.04,pad=.02)
    for ax in axs[row]:ax.set_axis_off()
    notes.append({'region_id':rid,'source_index':idx,'purpose':'Post-score illustration only; not independent manual truth labels'})
fig.suptitle('Selected post-score diagnostics (640m windows), NOT representative sample or road-status validation\nYellow/cyan: reference outline | third column: probability | grey: unassessed',fontsize=12)
fig.savefig(OUT/'expanded_diagnostic_cases.png',dpi=150);plt.close(fig)
(OUT/'diagnostic_case_selection.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')

near=r['pooled_unique_objects']['near_road100m'];small=r['pooled_unique_objects']['small_near_road100m'];large=r['pooled_unique_objects']['large_near_road100m']
objs=json.loads((OUT/'pooled_unique_objects.json').read_text());nr=[x for x in objs if x['distance_to_sh1_m']<=100 and x['evaluated_pixels']]
full=[x for x in nr if x['retained_fraction']==1];miss=[x for x in nr if not x['positive_pixels']]
rows=['| 区域 | SH1长度(km) | 道路周边质量保留 | 近路参考滑坡 | 可评价 | ≥10%重叠 | 未评价 |',
      '|---|---:|---:|---:|---:|---:|---:|']
for x in r['regions'][:-1]:
    n=x['objects']['near_road100m'];rows.append(f"| {x['region_id']} | {x['sh1_length_km']:.2f} | {x['quality_fraction_road100m']:.1%} | {n['source_objects']} | {n['evaluated_objects']} | {n['ge10pct']} | {n['unassessed_objects']} |")
boundary=sum(len(x['GDAL_approximate_warp_boundary_samples']) for x in r['independent_output_validation']['regions'])
text=f'''# BBUnet扩大测试报告：可运行，但暂不支持推荐为可靠道路风险检测主模型

完成日期：2026-10-08（新西兰）。实际运行作者完整任务权重，不是随机权重，也没有重新训练或调阈值。

## 结论

扩大测试没有维持首轮试验区的检出水平。新增7个区域中，道路100米附近共有56个参考滑坡，其中28个有可评价像素，10个检出（35.7%），18个在阈值0.5下完全没有预测重叠；另28个未评价，不能当作漏检或安全。10个检出的对象在“任一正像素”“至少10%可评价面积重叠”“至少50%重叠”三项诊断中计数均为10。

目前可向团队介绍BBUnet为**已经能运行的候选滑坡筛查原型**，但不宜称为“表现稳定、已适合道路通行判断”或“已证明最好的模型”。最明显的问题仍是相对于参考清单的小滑坡未检出和区域差异，真实失败原因还需标签/影像核实。道路附近的新增大滑坡仅1个可评价并检出，不能用1/1证明大滑坡泛化可靠。

## 测试设计与规模

沿当前OSM SH1路网，测试7个新6.4×6.4公里方形区域，共286.72平方公里、约55.14公里SH1，是首轮面积的7倍；另保留40.96平方公里原试验区作为处理对照，不混入新增成绩。新区域与原区不重叠，新区彼此也不重叠，并有500米分隔规则。

选区在预测前冻结，只使用道路长度、空间位置和两期选定产品覆盖，不使用滑坡数量或预测表现筛选。按既定规则保留所有区域，包括没有近路正样本、重云和表现差的区域；不是随机代表性抽样，也不是全部道路走廊覆盖。

输入：2016-11-03/11-23（新西兰日期）Sentinel-2 L2A的12个波段，以及ALOS地形4通道；正式BBUnet权重严格完整加载，阈值固定0.5。每区16个256像素图块，步长128，重叠处取sigmoid概率均值，共128图块（112新增、16对照）。模型输出为滑坡候选像素，而非道路状态。

新地形来源为Planetary Computer ALOS公开数据，源文件和哈希在sources；30米网格计算经过源码核查的RichDEM NumPy地形公式，再按作者坡向半角编码构造输入。未运行原生RichDEM，未完整复现作者全部流程。原试验区地形来自GEE，故必须单独对照。

参考标签：原Kaikora_2016滑坡清单，按原多边形面积区分大≥10,000平方米、小100–<10,000平方米；这是本地探索性分组，不是道路是否堵塞的定义。道路附近指原参考多边形距当前SH1≤100米，并不等于真实堵路。不同区重复源对象按source_index合并，未重复增加分母。

## 分区结果

{chr(10).join(rows)}

“可评价”指参考多边形内至少有1个通过质量筛查的10米网格中心像素，部分对象仍只剩2–4个像素，计数不能等同于完整对象评估。质量要求为两期源数据有效、SENCLOUD12均为clear、SCL均为4/5/7、地形数值有效。水、云、阴影、雪冰等被排除，排除区域不是模型判定安全。

南端SH1-EXP-01道路周边仅5.4%通过联合筛查。两期道路周边12波段源有效覆盖实际上都是100%，主要是灾后云/卷云等质量问题，不是未下载影像。该区32个近路对象中27个无可评价像素，不能计作漏检。

## 汇总与小滑坡问题

- 近路：10/28检出（35.7%）；小滑坡9/27（33.3%）；大滑坡1/1，样本不足。
- 新区全部参考对象589个，其中430个可评价；123个有任一正像素，116个至少10%可评价面积重叠，92个至少50%重叠。这些不是模型准确率。
- 新区可评价参考滑坡像素覆盖为47.24%；道路100米周边参考像素覆盖48.18%。两者都是条件重叠诊断，不是已验证召回率。
- 18个近路完全无重叠对象中，15个全部参考栅格像素通过质量筛查，因此不能把漏检全部归因于云或不完整影像。面积约214–4,305平方米；例：SH1-EXP-02的ID4884约3,320平方米，33个参考像素均可用，最大预测概率约0.00064。
- 10米质量掩膜侵蚀敏感性（预先声明，3×3方形邻域）后：近路9/27达到10%重叠，结论未改善。这不是为抬高成绩调整阈值。
- 在事后描述性的“全部栅格像素可评价”子集中，7/22检出。该子集不是预声明主评价，也不替代总体分母。

本次新区只找到2个近路大滑坡，另一个被质量筛查排除；不能据此声称BBUnet对大滑坡稳定。全AOI较远离道路的大滑坡可评价12个，10个达到10%重叠，仍不等于道路场景表现。

示例图的初步目视核查也提示参考标签需要解释：ID4884的多边形位置在灾前和灾后10米RGB图中都已有较明亮的裸露区域，并不像检出示例ID2593那样出现明显新增裸露。这不能直接证明标签错误（也可能再活动或尺度不足），但意味着“清单内没有预测重叠”不应一概当作已证实的真实灾害漏检。ID13006在此尺度也难独立确认灾后变化。示例图不是专业人工真值判读，35.7%更不能解释为真实召回率、其余64.3%也不是已验证漏检率。

## 为什么和首轮不同？

原区对照保留相同灾前/后归一化光学数组（逐元素验证相同），仅地形来源改变。同一质量可评价区域的概率平均绝对差约0.00699，二分类变化约0.47%。原区道路附近仍19/29达到10%重叠，首轮同诊断也为19/29；“任一像素命中”从20/29变成19/29。

因此，本轮新增区较差不能主要用地形来源变更解释。新样本更多为较小滑坡、沿线场景不同，首轮本来就是参考清单引导选出的单一区域，不能代表整个走廊。以上是数据支持的解释方向，不是已经证实的唯一原因。影像配准、清单精度、可见地表变化和训练覆盖仍可能影响结果，需要更高分辨率证据核查。

## 推荐门槛和限制

预测前设定的本地探索性门槛为：至少4区各有≥10个可评价近路滑坡；汇总≥70%近路对象至少10%重叠；大对象≥80%；有足够样本的单区不得低于50%。该门槛不是课程、作者或业务标准。

本轮未通过：只有1区满足样本量，汇总近路重叠35.7%也低于70%。所以既有测试支持不足，也有已观察到的小滑坡漏检问题，不能单纯以“样本少”掩盖性能下降。

仍不可报告precision、误报率、道路准确率：参考清单外的预测尚未独立证实真假。新区清单外有2,929个正像素（约0.293平方公里），道路100米周边107个（约0.0107平方公里），它们是待核查候选，不是已确认误报。无近路标签的SH1-EXP-03仍有5个道路周边候选正像素，也未直接当作误报。

全部来自同一次地震、同一影像对；与检查点训练区域是否重叠尚未核实。空间非重叠不等于独立事件测试；没有2016道路堵塞/可通行真值，模型阴性不能当作道路可通行。

## 数据和输出复核

8区已读回检查每个栅格、验证SHA256，重新从原shapefile栅格化共831个区域对象片段，并核对对象计数、像素计数、二值阈值和质量/侵蚀掩膜。8区各独立抽取原JP2源坐标，共4,800次波段像素比较。

其中{boundary}次处于GDAL默认近似重投影的源像元边界，直接坐标采样与近似变换选择了相邻像元；均确认处于0.125源像素容差内且取值来自相邻实际源像元。所有抽样值与相同明确重投影规则重新读取的结果一致。这不是4,800次精确坐标取值全部相同；最大直接反射率差为0.09。对很小目标，重投影和配准仍是限制，未做精确变换重跑敏感性或独立地理定位验证。

## 建议的项目路径

1. 保留BBUnet为可运行的候选风险提示层，诚实报告大范围下降；不推荐作为单独判定道路安全的主模型。
2. 优先逐一核查完整影像中仍漏检的近路对象，以及清单外的正预测：用更高分辨率影像或官方灾情/道路证据确认位置和可见性，避免直接把清单与模型差异都归咎于模型。
3. 若团队愿意进行局部微调，需新增区域级训练/验证/测试隔离和真实标签；不得用本轮测试标签调阈值后再称独立成绩。这项尚未执行。
4. 交付GIS风险图层时必须分开“疑似滑坡”“未检出”“数据不足”，叠加道路后只生成待核查路段，不能把未检出路段标为可通行。登陆点到目的地的可达性仍需要道路状态、起终点和路网分析。

## 文件

本报告及输出均位于：`{OUT}`。

- frozen_design.json：预测前冻结区域和规则；expanded_results.json：全区、汇总与对照结果。
- independent_validation.json：源像元抽样、重投影边界记录和输出重算。
- 各区域目录：BOA输入、归一化输入、地形、云/SCL、质量掩膜、预测概率、二值图、参考对象统计。
- expanded_overview.png：7区及原区对照总览；expanded_diagnostic_cases.png：事后挑选的2成功/2漏检示例，仅解释失败模式，不参与选区或改变汇总。
- 作者模型源自 https://github.com/links-ads/igarss-landslide-delineation ，本地固定提交ce34d3ac02b80d4db7e542203b7e534649a46352及完整检查点哈希见expanded_results.json。
'''
(OUT/'BBUnet扩大测试报告.md').write_text(text,encoding='utf-8')
print(str(OUT/'BBUnet扩大测试报告.md'),flush=True)
