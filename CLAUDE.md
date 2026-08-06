# CLAUDE.md — omics_feature_alignment

多组学动态特征对齐与集成模块 (Module 2)，构建在 Module 1 (omics_standardization) 之上。

## 项目概述

两阶段动态特征对齐流水线：

```
Module 1 .h5mu 输出
  → 阶段一：时间/过程对齐 (DTW/插值/伪时间/延迟建模)
  → 阶段二：特征空间校正 (MNN/CCA/最优传输/流形)
  → 三维度评估 → 结构化动态特征矩阵
```

## 常用命令

```powershell
# 激活环境
conda activate omics-std

# 运行全流程（自动选择方法）
omics-align config/default.yaml

# 指定方法
omics-align config/default.yaml --temporal dtw --feature-space optimal_transport

# 仅时间对齐
omics-align config/default.yaml --only temporal

# 运行测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_temporal/test_dtw.py -v
```

## 包结构（扁平布局）

`src/` 是包根目录（`packages = ["src"]`），所有内部导入使用绝对导入：

```
omics_feature_alignment/
├── src/
│   ├── readers/              # Module1Reader
│   ├── temporal/             # 阶段一：时间对齐 (DTW/插值/伪时间/延迟)
│   ├── feature_space/        # 阶段二：特征空间校正 (MNN/CCA/OT/Manifold)
│   ├── evaluation/           # 三维度评估
│   ├── preprocessing/        # da 命名空间 — 用户 API
│   ├── tools/                # tl 命名空间
│   ├── plotting/             # pl 命名空间
│   ├── pipeline/             # DynamicAlignmentPipeline
│   ├── utils/                # 工具函数
│   ├── _settings/            # 设置单例
│   ├── _logging.py           # 自定义日志 (HINT 级别)
│   └── main.py               # CLI
├── config/default.yaml
└── tests/
```

## 导入约定

```python
# ✅ 正确：绝对导入（扁平包结构）
from readers import Module1Reader
from temporal._base import BaseTemporalAligner
from feature_space._selector import FeatureSpaceSelector
from evaluation._time_consistency import evaluate_time_consistency
from utils._time_utils import normalize_time_scales
from pipeline import DynamicAlignmentPipeline
import _logging as logg
from _settings import settings

# ❌ 错误：不要用相对导入
# from ..utils import ...  ← ImportError
# from ._base import ...   ← 在非 __init__.py 中不要用
```

## 核心设计模式

### 1. run() 方法模式
所有处理器遵循 `run(data, **kwargs) -> data` 契约，原地修改并返回：
```python
aligner = DTWAligner(window_type="sakoechiba")
mdata = aligner.run(mdata, time_key="time")
```

### 2. Selector + Strategy 模式
每个子系统有选择器自动判断最优方法：
```python
selector = TemporalSelector()
mdata = selector.run(mdata)           # 自动选择
mdata = selector.run(mdata, method="dtw")  # 显式指定
```

### 3. 命名空间 (scanpy 风格)
```python
import preprocessing as da    # da.temporal(), da.feature_space(), da.align()
import tools as tl            # tl.run_full_evaluation(), tl.export_to_csv()
import plotting as pl         # pl.plot_warping_paths(), pl.plot_integrated_embedding()
```

### 4. 结果存储
对齐结果存储在 `.uns["alignment"]`，与 Module 1 的 `.uns["standardization"]` 并列：
- `mdata.uns["alignment"]["temporal"]` — 时间对齐溯源
- `mdata.uns["alignment"]["feature_space"]` — 特征空间溯源
- `mdata.uns["alignment"]["evaluation"]` — 评估结果
- `adata.obsm["X_temporal_aligned"]` — 时间对齐后矩阵
- `adata.obsm["X_feature_aligned"]` — 特征空间校正后矩阵

### 5. 降级策略
每个方法有 1-2 层 fallback：
- DTW: tslearn → scipy fallback
- CCA: pyrcca/rcca → sklearn CCA
- OT: POT GW/FGW/COOT → mean_shift fallback
- MNN: scanpy mnn_correct → batch mean-centering

## 新增依赖 (vs Module 1)

- `tslearn>=0.6` — DTW/SoftDTW/GAK
- `POT>=0.9` — GW/FGW/COOT 最优传输
- `pyrcca>=1.0` — 正则化 CCA (import as `rcca`)
- `rich` — CLI 美化

所有已存在于 `omics-std` 环境的包无需重新安装。

## 数据流

Module 1 输出位于 `d:\Database\omics_standardization\data\processed\`：
- `combined.h5mu` — 汇总文件（6 模态）
- `{modality}/{modality}_*.h5mu` — 各模态独立文件

Module 2 读取时优先使用 `combined.h5mu`，不存在则逐个读取独立文件。

## 测试

Mock fixtures 在 `tests/conftest.py`：
- `mock_mdata` — 3 模态 MuData (scrna + metabolomics + proteomics)
- `mock_mdata_with_day` — 包含 hour + day 双时间尺度
- `mock_mdata_no_time` — 无时间列
