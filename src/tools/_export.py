"""
导出工具

将对齐结果导出为 CSV 特征矩阵和 HTML 报告。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import _logging as logg

if TYPE_CHECKING:
    from mudata import MuData


def export_to_csv(
    mdata: MuData,
    output_dir: str | Path,
    layer_key: str = "X_feature_aligned",
    modalities: list[str] | None = None,
    max_dims: int | None = None,
) -> list[Path]:
    """将对齐后的特征矩阵导出为 CSV 文件。

    每行对应一个观测（cell/sample），每列对应特征向量维度。

    Args:
        mdata: 已对齐的 MuData
        output_dir: 输出目录
        layer_key: 要导出的矩阵键名
        modalities: 导出的模态列表（None=全部）
        max_dims: 最大导出维度数（None=不截断）

    Returns:
        导出的文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    mods = modalities or list(mdata.mod.keys())

    for mod_name in mods:
        if mod_name not in mdata.mod:
            continue
        adata = mdata.mod[mod_name]

        if layer_key not in adata.obsm:
            logg.warning(f"[{mod_name}] 缺少 '{layer_key}'，跳过 CSV 导出")
            continue

        X = np.asarray(adata.obsm[layer_key])
        n_dims = X.shape[1] if max_dims is None else min(X.shape[1], max_dims)

        if max_dims is not None and X.shape[1] > max_dims:
            logg.warning(f"[{mod_name}] 特征维度 {X.shape[1]} > max_dims={max_dims}，截断到 {max_dims} 维")

        # 构建 DataFrame
        df_data = {}
        for k in range(n_dims):
            df_data[f"dim_{k + 1}"] = X[:, k]

        df = pd.DataFrame(df_data)

        # 添加元数据列
        for col in ["aligned_time", "time", "condition", "batch", "is_interpolated",
                     "pseudotime", "process_stage"]:
            if col in adata.obs.columns:
                df[col] = adata.obs[col].values

        # 添加 obs_names
        df.insert(0, "sample", adata.obs_names.values)
        df.insert(1, "modality", mod_name)

        # 保存
        file_path = output_dir / f"{mod_name}_aligned_features.csv"
        df.to_csv(file_path, index=False)
        exported.append(file_path)
        logg.hint(f"导出: {file_path} ({df.shape[0]} 行 × {df.shape[1]} 列)")

    return exported


def export_time_indexed_matrix(
    mdata: MuData,
    output_dir: str | Path,
    layer_key: str = "X_feature_aligned",
    time_key: str = "aligned_time",
    stage_key: str = "process_stage",
    modalities: list[str] | None = None,
    max_dims: int | None = None,
) -> list[Path]:
    """导出以时间/过程为索引的结构化特征矩阵。

    按时间点或过程阶段聚合（均值），满足需求文档：
    "每一行对应一个时间点或阶段位置，每一列对应特征向量维度"。

    Args:
        mdata: 已对齐的 MuData
        output_dir: 输出目录
        layer_key: 对齐后矩阵的 obsm 键名
        time_key: 时间列名（用于时间索引聚合）
        stage_key: 阶段列名（用于伪时间阶段聚合）
        modalities: 导出的模态列表
        max_dims: 最大维度数

    Returns:
        导出的文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    mods = modalities or list(mdata.mod.keys())

    for mod_name in mods:
        if mod_name not in mdata.mod:
            continue
        adata = mdata.mod[mod_name]

        if layer_key not in adata.obsm:
            continue

        X = np.asarray(adata.obsm[layer_key])
        n_dims = X.shape[1] if max_dims is None else min(X.shape[1], max_dims)

        # 确定索引列：优先用时间，其次用阶段
        df = pd.DataFrame()
        use_time = time_key in adata.obs.columns and adata.obs[time_key].nunique() > 1
        use_stage = not use_time and stage_key in adata.obs.columns

        if use_time:
            df["time_index"] = adata.obs[time_key].values
            # 按四舍五入到有效数字来分组
            df["time_bin"] = np.round(adata.obs[time_key].values.astype(float), decimals=2)
            group_col = "time_bin"
            index_name = "time_point"
        elif use_stage:
            df["stage_index"] = adata.obs[stage_key].values
            group_col = "stage_index"
            index_name = "process_stage"
        else:
            # 无时间/阶段列，生成观测级 CSV（不聚合）
            logg.warning(f"[{mod_name}] 无时间/阶段列，跳过时间索引导出")
            continue

        # 添加特征列
        for k in range(n_dims):
            df[f"dim_{k + 1}"] = X[:, k]

        # 按索引列分组聚合（均值）
        grouped = df.groupby(group_col).mean()

        # 添加元数据
        grouped.index.name = index_name
        grouped["modality"] = mod_name
        grouped["n_observations"] = df.groupby(group_col).size().values

        # 保存
        file_path = output_dir / f"{mod_name}_time_indexed_features.csv"
        grouped.to_csv(file_path)
        exported.append(file_path)
        logg.hint(f"导出时间索引矩阵: {file_path} ({grouped.shape[0]} 个时间点 × {grouped.shape[1]} 列)")

    return exported


def export_integrated_matrix(
    mdata: MuData,
    output_dir: str | Path,
    max_dims: int | None = None,
) -> Path | None:
    """导出跨模态统一拼接特征矩阵。

    从 mdata.uns["alignment"] 中读取集成后的矩阵并导出。

    Args:
        mdata: 已对齐的 MuData
        output_dir: 输出目录
        max_dims: 最大维度数

    Returns:
        导出文件路径，若无集成矩阵则返回 None
    """
    alignment = mdata.uns.get("alignment", {})

    # 尝试从 alignment 中读取集成矩阵
    integrated = alignment.get("X_pca_integrated")
    if integrated is None:
        logg.warning("无集成矩阵可导出（请先运行 integrated_embedding）")
        return None

    if hasattr(integrated, "toarray"):
        integrated = integrated.toarray()
    X = np.asarray(integrated)
    n_dims = X.shape[1] if max_dims is None else min(X.shape[1], max_dims)

    df_data = {}
    for k in range(n_dims):
        df_data[f"integrated_dim_{k + 1}"] = X[:, k]
    df = pd.DataFrame(df_data)

    # 添加元数据（从 alignment 中获取模态顺序，标注每行来源）
    modality_sources = alignment.get("integrated_modality_order", list(mdata.mod.keys()))
    modality_labels = []
    for mod_name in modality_sources:
        if mod_name in mdata.mod:
            n = mdata.mod[mod_name].n_obs
            modality_labels.extend([mod_name] * n)

    # 截断到实际行数
    if len(modality_labels) > len(df):
        modality_labels = modality_labels[:len(df)]
    elif len(modality_labels) < len(df):
        modality_labels.extend(["unknown"] * (len(df) - len(modality_labels)))

    df["modality"] = modality_labels

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "integrated_features.csv"
    df.to_csv(file_path, index=False)
    logg.hint(f"导出集成矩阵: {file_path} ({df.shape[0]} 行 × {df.shape[1]} 列)")

    return file_path


def export_report(
    mdata: MuData,
    output_path: str | Path,
) -> Path:
    """生成对齐报告（JSON 格式，可作为 HTML 的数据源）。

    Args:
        mdata: 已评估的 MuData
        output_path: 输出文件路径（.json 或 .html）

    Returns:
        输出文件路径
    """
    output_path = Path(output_path)

    # 提取评估结果
    evaluation = mdata.uns.get("alignment", {}).get("evaluation", {})
    temporal_info = mdata.uns.get("alignment", {}).get("temporal", {})
    feature_space_info = mdata.uns.get("alignment", {}).get("feature_space", {})

    report = {
        "pipeline_version": mdata.uns.get("alignment", {}).get("pipeline_version", "0.1.0"),
        "steps_executed": mdata.uns.get("alignment", {}).get("steps_executed", []),
        "n_modalities": len(mdata.mod),
        "modalities": list(mdata.mod.keys()),
        "temporal": {
            "method": temporal_info.get("method"),
            "dtw_distance_matrix": temporal_info.get("dtw_distance_matrix"),
        },
        "feature_space": {
            "method": feature_space_info.get("method"),
            "distribution_shift": feature_space_info.get("distribution_shift"),
        },
        "evaluation": {
            "overall_score": evaluation.get("overall_score"),
            "time_consistency": {
                "sequence_similarity": evaluation.get(
                    "time_consistency", {}
                ).get("sequence_similarity_score"),
            },
            "distribution_consistency": {
                "mmd": evaluation.get("distribution_consistency", {}).get("mmd_score"),
                "silhouette": evaluation.get("distribution_consistency", {}).get("silhouette_score"),
            },
            "cross_modality_correlation": {
                "cca": evaluation.get(
                    "cross_modality_correlation", {}
                ).get("mean_canonical_correlation"),
            },
        },
    }

    if output_path.suffix == ".html":
        html = _render_html_report(report)
        output_path.write_text(html, encoding="utf-8")
    else:
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    logg.info(f"报告已生成: {output_path}")
    return output_path


def _render_html_report(report: dict) -> str:
    """渲染简洁的 HTML 报告"""
    ev = report.get("evaluation", {})
    overall = ev.get("overall_score", 0)
    score_color = "#27ae60" if (isinstance(overall, (int, float)) and overall > 0.5) else "#e74c3c"
    score_display = f"{overall:.4f}" if isinstance(overall, (int, float)) else str(overall)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Dynamic Alignment Report</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ color: #333; }} h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
        .score {{ font-size: 48px; font-weight: bold; color: {score_color}; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>Multi-Omics Dynamic Feature Alignment Report</h1>
    <p>Modalities: {', '.join(report.get('modalities', []))}</p>
    <p>Temporal: {report.get('temporal', {}).get('method', 'N/A')} | Feature Space: {report.get('feature_space', {}).get('method', 'N/A')}</p>

    <h2>Overall Score</h2>
    <div class="score">{score_display}</div>

    <h2>Evaluation Details</h2>
    <table>
        <tr><th>Dimension</th><th>Metric</th><th>Value</th></tr>
        <tr><td>Time Consistency</td><td>Sequence Similarity</td><td>{ev.get('time_consistency', {}).get('sequence_similarity', 'N/A')}</td></tr>
        <tr><td>Distribution</td><td>MMD</td><td>{ev.get('distribution_consistency', {}).get('mmd', 'N/A')}</td></tr>
        <tr><td>Distribution</td><td>Silhouette</td><td>{ev.get('distribution_consistency', {}).get('silhouette', 'N/A')}</td></tr>
        <tr><td>Cross-Modality</td><td>CCA</td><td>{ev.get('cross_modality_correlation', {}).get('cca', 'N/A')}</td></tr>
    </table>
</body>
</html>"""
