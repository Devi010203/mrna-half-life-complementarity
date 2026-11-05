# GSE336417 对照半衰期：正式评价前核查

日期：2026-11-02。结论：**可以开展基于公开 processed 结果的单 PDO 外部测量验证；本轮未计算任何模型性能。**

本轮仅保存官方半衰期 CSV、Supplementary Table 5 工作簿和 GEO series/sample 元数据，共 7,514,409 bytes（约 7.51 MB）。未下载 reads、BAM、SRA 或全套 supplementary。工作簿因 Table 5e 与其他 sheets 位于同一 XLSX 中而整体保存，其他实验不进入评价。

## 1. 核查结果

| 项目 | 状态 | 证据与含义 |
|---|---|---|
| 对照半衰期 | AVAILABLE | GEO `hl_exp_ctrl`；5,888 个唯一 gene_name；均为有限正值，单位小时 |
| 对照拟合质量 | AVAILABLE | `r2_exp_ctrl`；5,369 个严格 >0.6，519 个未过 |
| 2 小时达峰基因名单 | AVAILABLE AS PUBLISHED | GEO 与 Table 5e 的 5,888 个符号集合相同；GEO 处理说明与论文将该数量明确关联于 t=2 h 达峰名单 |
| 独立重做峰值/QC拟合 | UNAVAILABLE | 两份结果表没有逐时间点 T>C 数据、峰时列或逐重复拟合；本轮不处理原始 reads |
| GEO ↔ Table 5e 数值一致性 | PASS WITH PRECISION DIFFERENCES | 顺序不同，按符号一对一核对；控制质量筛选完全一致。不是逐字节或逐浮点精确相同 |
| 已有五模型 OOF | AVAILABLE | 全部 12,981 基因五模型预测有限；唯一 ENSID、唯一非空符号；使用冻结预测，无新推理 |
| Primary 匹配 | AVAILABLE | 控制 QC 合格的 5,369 个中，4,890 个精确匹配；479 个未精确匹配。未启用别名救回 |
| Saluki/External-PC1/新半衰期共有集 | AVAILABLE | 4,801 个；其余 89 个 Primary 基因不因缺 External-PC1 而剔除 |
| 覆盖偏倚所需旧基准特征 | AVAILABLE | 原 Saluki 标签、长度、分段 GC、既有外部测量覆盖字段均有冻结输入 |
| 新实验基因级深度/表达/逐重复不确定性 | UNAVAILABLE IN THESE FILES | 不据此声称恢复全部测量误差，也不把 3 次实验重复当 3 位患者 |
| 来源独立性 | NO OVERLAP IDENTIFIED IN DOCUMENTED INVENTORIES | 对 accession、样本/SRA experiment 标识和研究名称的既有来源清单核对未发现重叠；不是 reads 级去重证明 |

## 2. 原文、CSV 和 Supplement 的关系

论文方法说明，Matrigel 中残余标记造成延迟峰值，相关半衰期按 2、5、12 h 拟合；原作者对照/敲低比较另使用控制拟合 R² >0.6。[原文 Methods，SLAM-seq 部分](https://www.nature.com/articles/s41586-026-10890-0)

GEO 元数据当前已链接 PMID 42557330。28 个样本可分为 shCtrl/敲低各 12 个标记样本及各 2 个未标记背景样本；本研究只用处理表的 shCtrl 目标。[GEO GSE336417](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE336417)

**本研究集合的定义不沿用“扰动阳性”判断。**仅独立采用控制测量 QC，不以敲低结果、差异显著性、HyperTRIBE/ActD 或富集集合入选。全部 5,888 行并未先按 R²>0.6 过滤，因此可以显式恢复本研究的质量筛选。

Table 5e 与 GEO 控制半衰期最大绝对差约 **4.93×10⁻⁸ h**，控制 fit R² 最大绝对差约 **5.00×10⁻¹⁰**，没有跨越阈值造成的集合差异。GEO 为主数值来源，Table 5e 为交叉核查，不替换任何值。敲低列存在更大的舍入差异（含巨大外推半衰期），不参与本研究分析；详细列级比较在审计工作簿中，不以敲低列的差异决定控制基因去留。

## 3. 已锁定事项

- Primary 为 **4,890** 基因，`Bio-Fusion − Bio-XGB` Spearman 是唯一主比较。
- 控制半衰期有限且 >0、fit R² 合法且 >0.6；符号精确一对一；五模型共同可用。
- Spearman 用原始小时值；辅助 Pearson 用 log2 小时。不得对新标签再拟合、校准或选择模型。
- 正式阶段按 5,000 次 paired gene bootstrap、seed 42 评价；本轮未执行。
- 三目标同集合比较固定 **4,801** 基因，避免将集合构成差异混同为新情境效应。
- Primary 中 **251** 个半衰期 >12 h 的估计保留并注明外推范围，不为了好看或提高性能截尾。
- 覆盖分析分为 Primary 4,890、发表但 QC 未过 425、未精确进入发表名单 7,666；后两者合计 8,091，不当作新测量上的已知失败基因。

完整目标、排除、统计、覆盖和解释规则见 [LOCKED_ANALYSIS_PROTOCOL.md](LOCKED_ANALYSIS_PROTOCOL.md)。

## 4. 交付和复核

- `audit/SOURCE_QC_AND_MAPPING_AUDIT.xlsx`：同类核查合并为 8 个 sheets，包含所有控制行、12,981 基因覆盖分组、跨来源精度、28 个样本和来源重叠检查。
- `audit/primary_gene_manifest.csv`：4,890 个控制目标、QC、映射及固定顺序；无模型性能。
- `audit/three_target_gene_manifest.csv`：4,801 个共同目标集合；无模型性能。
- `audit/input_manifest.csv`：8 个既有只读输入和 4 个新官方来源的 SHA256。
- `audit/preflight_summary.json`：机器可读计数及版本。
- `download_sources.py` / `audit_sources.py`：小文件下载、来源/映射审计；没有训练、推理、相关系数或 bootstrap 计算函数。
- `FREEZE_SHA256.csv`：本轮文件冻结清单（不包含清单自身，避免自引用）。

程序已检查：表行数/唯一性、源符号集合、列结构、控制精度差异与 QC 一致性、预测可用性、匹配集合、28 个元数据样本、既有来源清单交叉检查、输入运行前后哈希不变。下载总 metadata 的一次尝试触及预设大小上限后未保存，改为仅请求 GSM metadata 成功；没有因此扩大下载范围。

旧仓库和正式数据没有被改动。本轮没有模型表现、目标相关性、覆盖组内部表现、bootstrap 或新图。没有修改手稿或上传仓库。下一步应在本方案冻结基础上单独执行，不把 preflight 的覆盖计数当作结果支持。

## 5. 方法辅助工具披露

本次使用 `scientific-critical-thinking` 技能帮助区分测量 QC、扰动选择、实验单位和解释范围；未将其当作自动质量评级，也没有更改论文参考文献。辅助工具参考：Kassis T, Agarwal V, He Y, Patel D, Brueckner AM (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。作者与版本信息于本轮核查（当前 v2，2026-09-02）。
