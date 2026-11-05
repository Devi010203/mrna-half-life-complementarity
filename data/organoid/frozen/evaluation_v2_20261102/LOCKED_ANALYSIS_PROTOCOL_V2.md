# GSE336417 shCtrl 正式评价方案 v2

修订日期：2026-11-02。性质：**性能计算前、经用户明确批准的两项次要诊断增补**，不是基于结果的方案变更。

## 1. 继承的冻结协议

本 v2 由下列原始 v1 全文及本文件的限定修订共同构成。原始文件、原始冻结记录及数据集合全部原位保留，不覆盖：

- `../preflight_20261102/LOCKED_ANALYSIS_PROTOCOL.md`
- v1 SHA256：`625d9d63b9a3c8f15a22dc49d5ea9285b3902c401a95013ce4f86cce029fdb70`
- v1 冻结索引 `../preflight_20261102/FREEZE_SHA256.csv`
- v1 冻结索引 SHA256：`dbaeac44fad8ca045480f927d634b60d6d84a546cf96645b15f15b8ef098ef2e`

未被本文件明确补充的规则一律沿用 v1。v1 的“本轮停在 preflight”停止点，现按用户授权推进为“完成正式评价与独立统计复核后停止”。不修改手稿、历史正式结果、模型、GitHub 发布包，不上传任何文件。

## 2. 唯一新增的两项预先指定诊断

### 2.1 生化模型相对 LM 的优势

在不变的 4,890 基因 Primary 集合增加：

`Spearman(Bio-XGB, shCtrl half-life) − Spearman(LM, shCtrl half-life)`。

使用与既有主比较完全相同的 5,000 次配对 gene bootstrap 抽样索引，给出 95% percentile CI。它是次要描述性分析；不升级为共同主终点，不根据其结果选择模型。

### 2.2 五折方向一致性

使用原有人类 OOF `fold` 标识，在 Primary 集合内分别报告每个外层测试折的 n、Bio-XGB Spearman、Bio-Fusion Spearman、Bio-Fusion−Bio-XGB 差值，以及正/零/负差值折数。

仅给描述性点估计，不增加折内 bootstrap、折间显著性检验、折均值新主终点、模型选择或重新加权。主分析仍为全部 4,890 基因的 pooled paired 比较。五折不是五次独立 PDO 实验；只检查该 OOF 划分下的方向一致性。

## 3. 未变更的要点

- 唯一主要比较：**Bio-Fusion−Bio-XGB Spearman**，n=4,890。
- 五模型：LM、Bio-XGB、Bio-Fusion、Common-XGB（Common-Compact）、Common-Fusion；均取保存的 OOF 预测，不训练、推理、重拟合、调参或校准。
- 原有次要比较保留：Bio-Fusion−LM、Common-Fusion−Common-XGB、Common-Fusion−LM。
- 主目标用 GEO `hl_exp_ctrl` 小时原值；Spearman 不改变方向；辅助 Pearson 使用 log2 半衰期。无 R² performance，无 winsorization、插补、符号救回或子集搜索。
- QC 不变：有限正半衰期、合法且严格 >0.6 的 `r2_exp_ctrl`、已发表名单、精确一对一映射、五模型均可用。
- **R²_exp >0.6 仅表示满足原作者采用的控制组衰减拟合标准，不证明半衰期具有很高测量精度。**没有逐重复半衰期不确定性；原始峰值和拟合未重建。
- 保留全部 **251** 个 >12 h 的 Primary 半衰期估计并标注外推限制；不增加阈值敏感性分析。
- 4,801 基因三目标共同集合、三对目标相关、同集合五模型 Spearman、三个目标下 Bio-Fusion−Bio-XGB 配对差和预设覆盖分析均不变。跨目标融合增益差只作描述，并由同次重采样直接计算。
- 每个集合按冻结 ENSID 顺序单独初始化 `numpy.random.default_rng(42)`；5,000 次等 n、有放回 gene bootstrap；模型与目标在集合内共用索引；每次重复样本重新排名，ties=average；95% 区间取 0.025/0.975、linear percentile。未定义统计量不丢弃或补抽，出现时暂停报告。
- 次要结果不用于正式多重检验结论，不以一个条件下正向另一个不显著推断交互。CI 条件于保存预测、目标估计和可观察基因，不含训练、拟合或患者间变异。
- 结论限定于单一 MSK107Li shCtrl PDO 背景，不称为未见基因泛化、多患者临床验证或因果情境效应。

## 4. 执行和验收

在读取任何性能结果前，生成 `PROTOCOL_V2_FREEZE.json`，记录本文件、v1、v1 冻结索引及两个集合清单的 SHA256 和 UTC 时间。正式程序在启动和结束时验证这些哈希及 v1 全部输入。

正式输出独立写入本目录 `results/`；保存配对 bootstrap 抽样索引流哈希和统计数组，便于逐次复核。独立统计程序不得导入正式评价程序的相关性、bootstrap 或连接函数，需从只读来源重新构造数据，使用不同实现复核所有点估计、5,000 次抽样、区间、fold 与覆盖分组。

同类结果合并为多 sheet XLSX，同时保留机器可读表。审查完成后形成报告及最终文件哈希；任何非数值实现修复必须记录，不得因结果修正规则。完成即停止。
