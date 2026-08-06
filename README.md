# omics_feature_alignment

**多组学动态特征对齐与集成模块 (Module 2)**

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

在空间免疫多组学数据库体系构建过程中，不同组学模态（转录组、蛋白组、代谢组、微生物组等）虽然在格式上已统一为特征向量，但在时间尺度、变化节奏及跨模态响应上仍存在显著不一致。**omics_feature_alignment** 从时间/过程对齐和特征空间校正两个层面构建动态特征对齐机制，使不同数据的变化过程具备可比性。

---

## 架构概览

```
Module 1 (omics_standardization)
  │  标准化 .h5mu 输出
  ▼
┌─────────────────────────────────────────────┐
│  Module 2 (omics_feature_alignment)          │
│                                              │
│  阶段一：多形态动态时间与过程对齐               │
│  ┌──────────────────────────────────────┐    │
│  │ ① DTW 动态时间规整 (tslearn)          │    │
│  │ ② 时间插值与统一映射                   │    │
│  │ ③ 伪时间排序与阶段映射 (scanpy DPT)    │    │
│  │ ④ 时间延迟建模与相关性对齐             │    │
│  └──────────────────────────────────────┘    │
│                    ↓                         │
│  阶段二：特征空间补充校正                      │
│  ┌──────────────────────────────────────┐    │
│  │ ① MNN 互为最近邻 (scanpy)            │    │
│  │ ② CCA/rCCA 共享潜在空间              │    │
│  │ ③ 最优传输 GW/FGW/COOT (POT)        │    │
│  │ ④ 流形对齐                          │    │
│  └──────────────────────────────────────┘    │
│                    ↓                         │
│  三维度评估 → 结构化动态特征矩阵输出            │
└─────────────────────────────────────────────┘
```

## 快速开始

### 环境

复用 Module 1 的 `omics-std` Conda 环境，额外安装以下依赖：

```bash
pip install tslearn POT pyrcca rich
```

### CLI 使用

```bash
# 全流程（自动选择方法）
omics-align config/default.yaml

# 仅时间对齐
omics-align config/default.yaml --only temporal

# 手动指定方法
omics-align config/default.yaml --temporal dtw --feature-space optimal_transport

# 详细输出
omics-align config/default.yaml -v

# 评估已有结果
omics-align config/default.yaml --evaluate-only data/aligned/aligned.h5mu
```

### Python API

```python
from readers import Module1Reader
import preprocessing as da
import tools as tl

# 读取 Module 1 输出
reader = Module1Reader("../omics_standardization/data/processed/")
mdata = reader.read_all()

# 一键运行两阶段对齐
mdata = da.align(mdata)

# 或分步操作
mdata = da.temporal(mdata, method="dtw")        # 阶段一
mdata = da.feature_space(mdata, method="cca")   # 阶段二

# 三维度评估
result = tl.run_full_evaluation(mdata)
print(f"Overall Score: {result['overall_score']:.4f}")

# 导出
tl.export_to_csv(mdata, "output/")
tl.export_report(mdata, "output/report.json")
```

### Pipeline（一键式）

```python
from pipeline import DynamicAlignmentPipeline

pipeline = DynamicAlignmentPipeline("config/default.yaml")
mdata = pipeline.run(
    input_path="../omics_standardization/data/processed/",
    output_path="data/aligned/",
)
```

## 项目结构

```
omics_feature_alignment/
├── src/
│   ├── readers/              # Module1Reader — 读取 Module 1 输出
│   ├── temporal/             # 阶段一：时间/过程对齐
│   │   ├── _dtw.py           #   DTW (tslearn)
│   │   ├── _interpolation.py #   时间插值
│   │   ├── _pseudotime.py    #   伪时间 (scanpy DPT)
│   │   ├── _lag_modeling.py  #   延迟建模
│   │   └── _selector.py      #   自动选择器
│   ├── feature_space/        # 阶段二：特征空间校正
│   │   ├── _mnn.py           #   MNN (scanpy)
│   │   ├── _cca.py           #   CCA/rCCA
│   │   ├── _optimal_transport.py  # GW/FGW/COOT (POT)
│   │   ├── _manifold.py      #   流形对齐
│   │   └── _selector.py      #   自动选择器
│   ├── evaluation/           # 三维度评估
│   ├── preprocessing/        # da 命名空间（用户 API）
│   ├── tools/                # tl 命名空间（评估/导出）
│   ├── plotting/             # pl 命名空间（可视化）
│   ├── pipeline/             # DynamicAlignmentPipeline
│   └── utils/                # 工具函数
├── config/default.yaml       # 配置文件
├── tests/                    # 31 tests
├── requirements.txt
└── pyproject.toml
```

## 支持的模态

| 模态 | 观测数 | 特征数 | 时间尺度 |
|------|--------|--------|----------|
| scrna (单细胞转录组) | 500 | 8,000 | hour |
| atac (染色质可及性) | 300 | 20,000 | hour |
| bulk_rna (散装转录组) | 50 | 2,000 | hour |
| metabolomics (代谢组) | 80 | 200 | hour |
| microbiome (微生物组) | 30 | 450 | day |
| proteomics (蛋白质组) | 100 | 50 | hour |

## 配置

编辑 `config/default.yaml` 调整所有参数：

```yaml
temporal:
  method: auto          # auto | dtw | interpolation | pseudotime | lag
  time_normalization:
    enabled: true
    target_unit: hour

feature_space:
  method: auto          # auto | mnn | cca | optimal_transport | manifold
  optimal_transport:
    variant: fused_gromov_wasserstein
    alpha: 0.5          # 0=pure GW, 1=pure Wasserstein
```

## 运行测试

```bash
cd omics_feature_alignment
pytest tests/ -v
```

## 依赖

**核心新增依赖**（`omics-std` 环境基础之上）：

- `tslearn>=0.6` — DTW / SoftDTW / GAK
- `POT>=0.9` — 最优传输 GW / FGW / COOT
- `pyrcca>=1.0` — 正则化 CCA
- `rich` — 终端输出美化

**可选依赖：**

- `mofapy2` — MOFA 多组学因子分析
- `GPy` — 高级高斯过程核
- `torch` — 深度学习后端
- `scvi-tools` — scVI 潜在模型

## 参考

- [muon](https://github.com/scverse/muon) — 多组学 MuData 框架
- [POT](https://pythonot.github.io/) — Python Optimal Transport
- [tslearn](https://tslearn.readthedocs.io/) — 时间序列机器学习
- [scanpy](https://scanpy.readthedocs.io/) — 单细胞分析工具包

## License

BSD-3-Clause
