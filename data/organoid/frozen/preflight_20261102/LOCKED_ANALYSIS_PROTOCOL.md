# GSE336417 shCtrl：新增外部人类测量验证方案 v1

日期：2026-11-02。状态：**分析规则与基因集合已在性能计算前锁定；本轮只完成 preflight，尚未执行正式评价。**

## 1. 问题和实验单位

检验既有人类 OOF 预测与 MSK107Li 患者来源结直肠癌肝转移类器官的 shCtrl 半衰期是否一致，以及 Bio-Fusion 相对 Bio-XGB 的增益是否在这一测量背景中保留。生物学背景为一个 PDO 系；基因是统计重采样单位，不是独立患者。shCtrl 是接受实验流程的非靶向 shRNA 对照，不称为未经处理的正常组织。无 4sU 样本用于背景校正，不单独作为半衰期目标。

这是已知基因上的新测量/新情境评价，不是未见基因泛化、多患者临床验证或癌症机制的因果检验。不得把新目标并入 External-PC1，不改变已有论文、正式结果、模型或发布包。

## 2. 来源及精度优先级

- 主数值来源：`sources/GSE336417_SLAMseq_halflife.csv.gz` 的 `hl_exp_ctrl`（小时）和 `r2_exp_ctrl`（单指数衰减拟合优度）。GEO 数值原样保留。
- 核对来源：`sources/41586_2026_10890_MOESM7_ESM.xlsx` 的 `Supplementary Table 5e`，Excel 第 2 行为字段名。两份表的 5,888 个符号集合完全一致，但顺序和数值精度不完全相同。按符号连接，不按行号连接。不得在两份来源之间逐基因挑选值。
- 5,888 个已发表基因的 t=2 h 达峰身份来自 GEO 的处理说明及原文 Methods；**没有独立的峰时列、逐时间点转换率或逐重复半衰期**。可恢复发表名单和作者处理结果，不能独立重算或声称验证了峰值调用及衰减拟合。
- 控制组数值跨来源比较及全部输入 SHA256 见 `audit/`。敲低列仅用于来源一致性核查，不进入本研究目标、筛选或模型选择。

## 3. 事先固定的集合与映射

### Primary：CONTROL_QC_OOF

以下条件取交集，不能根据性能增删：

1. 属于 GEO 已发表的 5,888 行名单；不扩大到未提供有效拟合依据的其他基因。
2. `hl_exp_ctrl` 是有限实数且严格大于 0。
3. `r2_exp_ctrl` 是有限值、在 [0, 1] 内且**严格 > 0.6**。此阈值来自论文的对照测量质量标准，并在本研究中独立采用为控制组质量门槛；不是根据模型效果选定。
4. `gene_name` 与冻结人类 OOF 表 `gene_symbol` 大小写敏感、精确、一对一匹配，由后者取得 ENSID。空值、重名、一对多均排除并记账；不临时查询别名救回，不按预测值挑选映射。当前无空符号/重复符号/首尾空格。
5. 五个预列模型均有有限 OOF 预测；必须共享同一集合。当前五模型全部可用，没有因预测缺失而额外剔除。

锁定结果：5,888 → 控制 QC 5,369 → 唯一匹配且五模型可用 **4,890**。以 `audit/primary_gene_manifest.csv` 的 ENSID 升序作为正式顺序。

### COMMON_THREE_TARGETS

上述 4,890 个基因再与冻结 `data/external/matched.csv` 按 ENSID 取交集，得到 **4,801** 个。顺序见 `audit/three_target_gene_manifest.csv`。**不以 External-PC1 可用性限制 Primary**；另 89 个基因仍留在 Primary。

### 与原作者扰动分析的区别

本研究采用的是控制测量质量条件，不是“ZFP36L2 依赖性基因集”。不要求敲低组拟合通过，不按敲低半衰期、组间增幅、P 值、GO/HyperTRIBE/ActD 阳性或结合靶点筛选。不使用 Table 5f 的富集输入集。与作者某一步筛选在成员上重合，不等于使用作者的扰动终点；两者分别记录。

不新增基于结果的 QC 阈值扫描或最佳子集搜索。若以后需要修改映射、阈值或增加敏感性集合，应保留本方案与本轮结果，并明确标记为后续探索，不能替换 Primary。

## 4. 目标、变换和无效值

- Primary Spearman：预测与原始 `hl_exp_ctrl` 小时值；方向固定为越大越稳定。秩计算对 ties 取 average rank。
- Secondary Pearson：预测与 `log2(hl_exp_ctrl)`；仅对有限正值取对数，不加伪计数。
- Saluki 原标签和 External-PC1 保持已冻结方向与数值，不按新标签翻转符号、重新标准化或重拟合 PC1。
- 不报告跨目标数值尺度不一致的 R²，不以新标签拟合截距、斜率、融合系数或校准函数。
- 缺失、非数字、±Inf、≤0 的半衰期不插补；不把缺失变 0。拟合 R² 越界/缺失直接列入无效原因。当前对照列均为有限正值/合法 R²。
- 不截尾、winsorize、任意去除极端值。合格半衰期即使超过最后 chase 时间仍保留，同时描述其数量和范围。这些长半衰期是估计值，未提供其置信区间，不能当作高精度直接观测或擅自判为右删失。
- Primary 中有 251 个估计半衰期 >12 h；全体 Primary 范围 0.973843235–43.37544277 h。上述标记不参与筛选。

## 5. 模型、主比较与统计

冻结输入根目录：`../../release_preparation/github_migration_20261031/execution/working_repository/`（相对本文件）。

所有模型只从 `data/human/oof.csv` 读取各基因原有 held-out 预测：LM、Bio-XGB、Bio-Fusion、Common-XGB、Common-Fusion。这里 Common-XGB 指既定 Common-Compact XGB。保留原 fold。**不得用五个人类模型对同一人类基因平均预测代替 OOF**，不得重新训练、推理、调参、选模型、early stopping、重新拟合融合或用新标签择优。

唯一 Primary 比较：

`Δρ = Spearman(Bio-Fusion, shCtrl half-life) − Spearman(Bio-XGB, shCtrl half-life)`，在全部 4,890 个 Primary 基因上评价。

预列 Secondary：五模型 Spearman；五模型对 log2 半衰期的 Pearson；Bio-Fusion−LM、Common-Fusion−Common-XGB、Common-Fusion−LM 的 Spearman 差。Secondary 仅作带区间的描述性支持，不进行选择性显著性声明，不替代主比较。

沿用已有 gene bootstrap 定义：5,000 次等样本量、有放回基因重采样，`numpy.random.default_rng(42)`；模型共用每次抽样索引；重复抽中的基因在每次 Spearman 中重新排名，ties 用 average；95% percentile CI 使用 0.025/0.975、linear 插值。差值在每次配对重采样中直接计算，禁止相减两个独立 CI。集合排序固定后再抽样。不得因 CI 不利改 seed 或次数。

若出现常量样本等造成未定义统计量，不静默丢弃或补抽；保存诊断并暂停该区间的正式报告。所有 CI 均条件于固定模型、固定目标估计和可观测基因，不包含训练、半衰期拟合或患者间不确定性。CI 跨 0 表示方向未被精确确定，不等于无效或等效。未预设最小实用增益，不夸大很小但正向的差值。

## 6. 同一基因上的三目标比较

只用 COMMON_THREE_TARGETS 的 4,801 个基因：

1. 比较原 Saluki target、既有 External-PC1 与新 shCtrl 半衰期的三对 Spearman；辅助 Pearson 中仅新半衰期取 log2。
2. 对五个原有 OOF 模型分别计算三个目标的 Spearman，避免把不同覆盖集合的变化解释为情境变化。
3. 报告主模型对比 Bio-Fusion−Bio-XGB 在三个目标下的差值及配对 CI。跨目标增益变化只作描述性比较，不作为因果物种/肿瘤/平台效应。
4. 此共同集合单独以 seed 42、5,000 次重采样，并在每次抽样中对三个目标和五模型使用同一索引。如报告两个目标之间的增益差，也必须由同次重采样直接相减，不能相减独立 CI。

## 7. 覆盖偏倚（正式评价时执行，preflight 只锁定分组）

在全部 12,981 个人类基准基因中分组：

- Primary included：4,890。
- 在发表名单精确匹配、但控制 QC 未过：425。
- 未精确匹配到发表名单：7,666。
- 合并后未纳入 Primary：8,091。

末组只说明未进入这个可恢复的处理名单，不能称为“未表达”或“测不到”；可能涉及发表筛选、覆盖或符号变化。

使用冻结 `data/human/coverage_features.csv`，报告上述组的原 Saluki 标签、`TOTAL_LEN_SEQ`、UTR5/CDS/UTR3 长度及各区段 GC 的 n、缺失数、中位数和 IQR；不臆造未保存的总体 GC。报告既有 External-PC1 覆盖情况及 `n_measurements/n_datasets/n_cell_lines`，明确这些是**旧外部人类数据的覆盖**，不是该 SLAM-seq 实验的 reads/expression。

比较各组原 OOF 对原 Saluki 标签的 Spearman（描述性），以检验纳入集是否本就较易预测。不计算未纳入基因对新目标的“外部表现”。另单列 4,801 个三目标共有基因与另外 89 个 Primary 基因的覆盖构成。若缺乏新实验 read-depth、表达量或逐重复质量数据，报告 unavailable，不以半衰期或 fit R² 冒充深度。

## 8. 复核与停止点

正式运行前验证 `audit/input_manifest.csv` 和本目录 `FREEZE_SHA256.csv`；任何不一致先停下，不覆盖结果使其通过。正式运行另建目录，不覆盖 preflight。

运行后复核：ENSID 唯一性、顺序、5 模型配对、目标方向、包含/排除原因、系数/OOF来源、无新标签拟合、统计实现、CI、全部负向结果和输出完整性。原始效应与区间完整报告，不按模型表现裁剪基因或图轴。

本轮到预执行核查和锁定停止；下一轮获准后才进行指标和 bootstrap。最终只讨论单一 MSK107Li 对照背景中互补性的保留/减弱/不确定性，不能宣称多患者临床效度、未见基因泛化或归因于某一种生物机制。
