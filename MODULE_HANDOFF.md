# Module 1 → Module 2 数据交接文档

## Module 1 当前状态

### 已实现功能

Module 1（multi-omics standardization pipeline）已完成全部 6 步流水线：

```
Raw files → ① Parse → ② Select Strategy → ③ Impute → ④ Normalize → ⑤ Batch Correct → ⑥ Evaluate → Processed data
```

- **9 种文件格式** → AnnData：`.h5ad`, `.csv`, `.tsv`, `.fcs`, `.mzML`, `.biom`, `.fastq`, `.bam`
- **6 种组学模态**：scrna, bulk_rna, proteomics, metabolomics, atac, microbiome
- **智能算法选择**：GMM 模态检测 + RF 策略推荐（已训练模型在 `config/models/`）
- **混合存储**：MinIO + SQLite/DM8 + Neo4j（暂未启用，本地文件 fallback）

### 已知限制

- Windows 上 R/Bioconductor 需手动安装（TMM/DESeq2/VSN/Scran 的 R 路径不可用，走 Python fallback）
- pysam 在 Windows 上需 Visual Studio Build Tools 编译（BAM/SAM 解析暂不可用）
- 运行 Pipeline 前需设置 `$env:R_HOME = "$env:CONDA_PREFIX\Lib\R"`
- combined.h5mu 在部分情况下写入失败（各模态独立文件始终正常）

---

## Module 2 输入：Module 1 的输出

### 数据位置

```
d:\Database\omics_standardization\data\processed\
├── combined.h5mu                  # 汇总 MuData（合并所有模态，可能不存在）
├── scrna/scrna_expression.h5mu     # 各模态独立文件（始终存在）
├── atac/atac_peaks.h5mu
├── bulk_rna/bulk_rna_counts.h5mu
├── metabolomics/metabolomics_intensities.h5mu
├── microbiome/otu_table.h5mu
└── proteomics/proteomics_sample.h5mu
```

### 数据形状

```
scrna:          500 obs  ×  8,000 vars   (单细胞转录组)
atac:           300 obs  × 20,000 vars   (染色质可及性)
bulk_rna:        50 obs  ×  2,000 vars   (散装转录组)
metabolomics:    80 obs  ×    200 vars   (代谢组)
microbiome:      30 obs  ×    450 vars   (微生物组)
proteomics:     100 obs  ×     50 vars   (蛋白质组)
```

### AnnData 结构

```python
adata
├── .X                     # 原始表达矩阵 (CSR sparse, n_obs × n_vars)
├── .layers["imputed"]     # 插补后矩阵
├── .layers["normalized"]  # 归一化后矩阵
├── .obsm["X_corrected"]   # 批次校正后矩阵 ← Module 2 主要用这个
├── .obs                   # 样本元数据
│   ├── "batch"            # 批次标签 (字符串)
│   ├── "time"             # 时间点 (数值, 小时或天)
│   ├── "condition"        # 实验条件 (字符串)
│   └── "time_unit"        # 时间单位 (字符串, "hour"/"day", 仅 h5ad 模态)
├── .var                   # 特征名
└── .uns["standardization"] # 处理溯源
    ├── "parser"           # 解析来源
    ├── "strategy"         # 检测到的模态 + 推荐的方法组合
    ├── "imputation"       # {method, ...}
    ├── "normalization"    # {method, ...}
    ├── "batch_correction" # {method, device, mode_collapse_risk, ...}
    └── "evaluation"       # {metrics: {mmd, wasserstein, batch_silhouette, ...}}
```

### 实验条件详情

| 模态 | 条件 | 时间点 | 时间单位 | 批次 |
|------|------|--------|----------|------|
| scrna | LPS_stimulation | 0, 6, 12, 24, 48 | hour | donor_A, donor_B |
| atac | hypoxia_exposure | 0, 24, 72 | hour | tissue_lung, tissue_liver |
| bulk_rna | drug_treatment | 0, 3, 6, 12, 24 | hour | control |
| metabolomics | high_fat_diet | 0, 4, 12, 48 | hour | run_01 |
| microbiome | probiotics_intervention | 0, 7, 30 | day | stool, oral |
| proteomics | cytokine_induction | 0, 8, 24, 72 | hour | panel_v1, panel_v2 |

---

## Module 2 读取方式

### 读取独立文件

```python
from mudata import read

# 方式 1：读汇总文件（如存在）
m = read("data/processed/combined.h5mu")
adata_scrna = m.mod["scrna"]

# 方式 2：读独立文件（始终可用）
m2 = read("data/processed/scrna/scrna_expression.h5mu")
adata = m2.mod["data"]  # 单模态文件内键名为 "data"
```

### 提取 Module 2 所需数据

```python
# 批次校正后的特征矩阵（推荐用于动态对齐）
X = adata.obsm["X_corrected"]  # shape: (n_obs, latent_dim)

# 或归一化后的特征矩阵
X = adata.layers["normalized"]  # shape: (n_obs, n_vars)

# 时间信息
time_points = adata.obs["time"].values          # 数值数组
conditions = adata.obs["condition"].values       # 字符串数组
batch_labels = adata.obs["batch"].values         # 字符串数组

# 查看处理使用的方法
trace = adata.uns["standardization"]
print(trace["normalization"]["method"])          # "scran" / "quantile" / ...
print(trace["batch_correction"]["method"])       # "harmony" / "combat" / ...
```

---

## Module 1 项目结构 (供引用)

```
omics_standardization/
├── src/                        # Python 包根目录 (packages = ["src"] 扁平布局)
│   ├── parsers/                # 数据解析
│   ├── _selectors/             # 智能算法选择（下划线避免与 stdlib 冲突）
│   ├── imputers/               # 缺失值插补
│   ├── normalizers/            # 尺度归一化
│   ├── batch_correctors/       # 批次校正
│   ├── preprocessing/          # pp 命名空间 (impute/normalize/batch_correct)
│   ├── tools/                  # tl 评估工具
│   ├── pipeline/               # StandardizationPipeline
│   ├── storage/                # 混合存储
│   ├── _logging.py             # 日志系统
│   └── _settings/              # 配置单例
├── config/default.yaml         # 默认配置
├── config/models/              # 已训练的 GMM + RF 模型
├── data/raw/                   # 原始数据（Module 2 不需要）
├── data/processed/             # 处理后数据（Module 2 输入）
├── train.py                    # 独立模型训练脚本
├── scripts/                    # demo 生成和验证
└── environment.yml             # Conda 环境 (Python 3.10+)
```

### 导入约定

```python
# 正确：绝对导入（扁平包结构）
from parsers import parse_file
from pipeline import StandardizationPipeline
import _logging as logg
from _settings import settings

# 错误：不要用相对导入
# from .. import logging  ← 会报 ImportError
```

### 环境

- Conda 环境名：`omics-std`
- 运行前设 `$env:R_HOME = "$env:CONDA_PREFIX\Lib\R"`（PowerShell）
- `data/raw/` 下有 demo 数据可直接用 Pipeline 生成 processed 输出

---

## 生成新的 processed 数据

如需重新生成或使用不同参数：

```powershell
# 生成 demo 原始数据（已有则跳过）
python scripts/generate_demo_data.py

# 训练模型（如已训练则跳过）
D:\Anaconda3\envs\omics-std\python.exe train.py

# 运行 Pipeline（需设置 R_HOME）
$env:R_HOME = "D:\Anaconda3\envs\omics-std\Lib\R"
python run_all.py
# 或 Python API:
# from pipeline import StandardizationPipeline
# StandardizationPipeline().run("data/raw/", "data/processed/")
```
