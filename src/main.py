"""
CLI 入口：omics-align

多组学动态特征对齐与集成命令行工具。

用法:
    omics-align config/default.yaml                      # 全流程
    omics-align config/default.yaml --only temporal       # 仅时间对齐
    omics-align config/default.yaml -v                    # 详细输出
    omics-align config/default.yaml -i path/to/processed -o path/to/aligned
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 日志级别
    from _settings import settings, Verbosity
    if args.verbose:
        settings.verbosity = Verbosity.debug

    # 加载配置
    config_path = args.config or "config/default.yaml"
    try:
        settings.load_config(Path(config_path))
    except Exception as e:
        print(f"配置加载失败: {e}")
        return 1

    # 导入 Pipeline
    from pipeline import DynamicAlignmentPipeline

    # 确定输入输出路径
    input_path = args.input or settings.input.get("data_dir")
    output_path = args.output or "data/aligned"

    # 创建并运行 Pipeline
    pipeline = DynamicAlignmentPipeline(config_path)

    try:
        if args.evaluate_only:
            # 仅评估模式
            from mudata import read as read_mudata
            eval_path = Path(args.evaluate_only)
            if not eval_path.exists():
                print(f"文件不存在: {eval_path}")
                return 1
            mdata = read_mudata(str(eval_path))
            from tools._evaluation import run_full_evaluation
            eval_result = run_full_evaluation(mdata)
            print(f"\n综合评分: {eval_result['overall_score']:.4f}")
            return 0

        if args.only:
            # 分步模式
            from readers import Module1Reader

            reader = Module1Reader(
                data_dir=Path(input_path),
                layer=settings.input.get("alignment_layer", "X_corrected"),
                time_key=settings.input.get("time_key", "time"),
                condition_key=settings.input.get("condition_key", "condition"),
                batch_key=settings.input.get("batch_key", "batch"),
            )
            mdata = reader.read_all(args.modalities)

            if args.only == "temporal":
                from temporal import TemporalSelector
                from utils._time_utils import normalize_time_scales

                mdata = normalize_time_scales(mdata)
                selector = TemporalSelector()
                mdata = selector.run(mdata)

            elif args.only == "feature_space":
                from feature_space import FeatureSpaceSelector

                selector = FeatureSpaceSelector()
                mdata = selector.run(mdata)

            elif args.only == "evaluate":
                from tools._evaluation import run_full_evaluation

                result = run_full_evaluation(mdata)
                print(f"\n综合评分: {result['overall_score']:.4f}")

            # 保存
            if args.only != "evaluate":
                output_dir = Path(output_path)
                output_dir.mkdir(parents=True, exist_ok=True)
                mdata.write(str(output_dir / "aligned.h5mu"))
                print(f"已保存: {output_dir / 'aligned.h5mu'}")

        else:
            # 全流程
            mdata = pipeline.run(
                input_path=Path(input_path) if input_path else None,
                output_path=Path(output_path) if output_path else None,
                modalities=args.modalities,
            )
            print(f"\nPipeline 完成: {len(mdata.mod)} 个模态已对齐")

        return 0

    except Exception as e:
        print(f"运行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器"""
    parser = argparse.ArgumentParser(
        prog="omics-align",
        description="多组学动态特征对齐与集成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  omics-align config/default.yaml
  omics-align config/default.yaml --only temporal
  omics-align config/default.yaml -i ../omics_standardization/data/processed -o data/aligned
  omics-align config/default.yaml -v
  omics-align config/default.yaml --evaluate-only data/aligned/aligned.h5mu
        """,
    )

    parser.add_argument(
        "config", type=str, nargs="?",
        help="配置文件路径（YAML），默认 config/default.yaml",
    )
    parser.add_argument(
        "--input", "-i", type=str,
        help="Module 1 处理后数据目录（覆盖配置文件中的设置）",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="输出目录",
    )
    parser.add_argument(
        "--modalities", nargs="+",
        help="要处理的模态列表（如 scrna atac bulk_rna）",
    )
    parser.add_argument(
        "--only", choices=["temporal", "feature_space", "evaluate"],
        help="仅运行指定阶段",
    )
    parser.add_argument(
        "--temporal", dest="temporal_method",
        choices=["dtw", "interpolation", "pseudotime", "lag"],
        help="时间对齐方法（覆盖配置文件中的设置）",
    )
    parser.add_argument(
        "--feature-space", dest="feature_space_method",
        choices=["mnn", "cca", "optimal_transport", "manifold"],
        help="特征空间校正方法（覆盖配置文件中的设置）",
    )
    parser.add_argument(
        "--evaluate-only", type=str,
        help="仅评估已有的对齐结果文件",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细输出",
    )
    parser.add_argument(
        "--version", action="version",
        version="omics-align 0.1.0",
    )

    return parser


if __name__ == "__main__":
    sys.exit(main())
