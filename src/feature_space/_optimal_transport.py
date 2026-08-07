"""
OptimalTransportAligner — 最优传输对齐器

基于 POT 实现：
- Gromov-Wasserstein (GW)：纯结构对齐
- Fused Gromov-Wasserstein (FGW)：特征+结构联合对齐
- Co-Optimal Transport (COOT)：同时对齐样本和特征
- Unbalanced OT：处理部分观测/异常值
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg
from ._base import BaseFeatureAligner

if TYPE_CHECKING:
    from anndata import AnnData


class OptimalTransportAligner(BaseFeatureAligner):
    """最优传输分布对齐器。

    通过最小化分布之间的传输代价，实现群体层面的特征分布对齐。
    传输计划用于实际的数据变换（barycentric projection）。
    """

    def __init__(
        self,
        variant: str = "fused_gromov_wasserstein",
        alpha: float = 0.5,
        epsilon: float = 0.01,
        max_iter: int = 1000,
        solver: str = "pgd",
        reg_m: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.variant = variant
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.solver = solver
        self.reg_m = reg_m
        self._method_name = "OptimalTransportAligner"

    def run(
        self,
        mdata: MuData,
        variant: str | None = None,
        alpha: float | None = None,
        epsilon: float | None = None,
        max_iter: int | None = None,
        **kwargs,
    ) -> MuData:
        variant = variant or self.variant
        alpha = alpha if alpha is not None else self.alpha
        epsilon = epsilon if epsilon is not None else self.epsilon
        max_iter = max_iter or self.max_iter

        mod_names = list(mdata.mod.keys())
        if len(mod_names) < 2:
            logg.warning("OT: 需要至少 2 个模态")
            return mdata

        logg.info(f"最优传输对齐: variant={variant}, alpha={alpha}, epsilon={epsilon}")

        # 提取特征矩阵并测量对齐前分布
        matrices = {}
        for mod_name in mod_names:
            matrices[mod_name] = self._get_feature_matrix(mdata.mod[mod_name])

        # 成对 OT 对齐
        ot_results: dict[str, dict] = {}
        aligned_matrices: dict[str, np.ndarray] = {mod_name: matrices[mod_name].copy()
                                                     for mod_name in mod_names}

        # 选择参考模态（观测数最多的）
        ref_mod = max(mod_names, key=lambda m: matrices[m].shape[0])

        for i in range(len(mod_names)):
            for j in range(i + 1, len(mod_names)):
                mod_a, mod_b = mod_names[i], mod_names[j]
                try:
                    result = self._run_ot(
                        matrices[mod_a], matrices[mod_b],
                        variant, alpha, epsilon, max_iter,
                    )
                    pair_key = f"{mod_a}_{mod_b}"
                    ot_results[pair_key] = result

                    # 用传输计划做 barycentric projection
                    if "plan" in result and result["plan"] is not None:
                        plan = result["plan"]
                        # 将 mod_b 投影到 mod_a 的空间
                        n_a = matrices[mod_a].shape[0]
                        n_b = matrices[mod_b].shape[0]
                        plan_rescaled = plan * n_b  # 行归一化后的传输
                        aligned_matrices[mod_b] = plan_rescaled @ matrices[mod_a]

                    logg.hint(
                        f"  {pair_key}: GW distance={result.get('gw_dist', 0):.4f}, "
                        f"variant={result.get('variant', 'unknown')}"
                    )

                except Exception as e:
                    logg.error(f"  {mod_a}-{mod_b} OT 失败: {e}")

        # 将对齐后的矩阵写入各模态
        distribution_shift: dict[str, dict] = {}
        for mod_name in mod_names:
            adata = mdata.mod[mod_name]
            X_before = matrices[mod_name]
            X_after = aligned_matrices[mod_name]

            adata.obsm["X_feature_aligned"] = X_after
            distribution_shift[mod_name] = self._measure_distribution_shift(
                X_before, X_after
            )

        self._store_trace(
            mdata,
            method="optimal_transport",
            params={
                "variant": variant,
                "alpha": alpha,
                "epsilon": epsilon,
                "max_iter": max_iter,
                "solver": self.solver,
            },
            extra={
                "reference_modality": ref_mod,
                "ot_results": ot_results,
                "distribution_shift": distribution_shift,
                "stored_in_obsm": "X_feature_aligned",
            },
        )

        logg.info(f"OT 对齐完成: {len(ot_results)} 对模态, ref={ref_mod}")
        return mdata

    def _run_ot(
        self, X_a: np.ndarray, X_b: np.ndarray,
        variant: str, alpha: float, epsilon: float, max_iter: int,
    ) -> dict:
        """执行最优传输：返回传输计划和距离"""
        try:
            import ot

            n_a, n_b = X_a.shape[0], X_b.shape[0]
            a = np.ones(n_a) / n_a
            b = np.ones(n_b) / n_b

            # 样本间特征距离矩阵
            M = ot.dist(X_a, X_b, metric="euclidean")
            M /= M.max() if M.max() > 0 else 1.0

            # 模态内结构距离矩阵
            C_a = ot.dist(X_a, X_a, metric="euclidean")
            C_b = ot.dist(X_b, X_b, metric="euclidean")
            C_a /= C_a.max() if C_a.max() > 0 else 1.0
            C_b /= C_b.max() if C_b.max() > 0 else 1.0

            if variant in ("gw", "gromov_wasserstein"):
                plan = ot.gromov.entropic_gromov_wasserstein(
                    C_a, C_b, a, b, epsilon=epsilon, max_iter=max_iter,
                )
                gw_dist = ot.gromov.gromov_wasserstein2(C_a, C_b, a, b)
                return {"plan": plan, "gw_dist": float(gw_dist), "variant": "gw"}

            elif variant in ("fused_gw", "fused_gromov_wasserstein"):
                plan = ot.gromov.entropic_fused_gromov_wasserstein(
                    M, C_a, C_b, a, b, alpha=alpha, epsilon=epsilon, max_iter=max_iter,
                )
                fgw_dist = ot.gromov.fused_gromov_wasserstein2(
                    M, C_a, C_b, a, b, alpha=alpha,
                )
                return {"plan": plan, "gw_dist": float(fgw_dist), "variant": "fgw"}

            elif variant == "coot":
                try:
                    plan_s, plan_f = ot.coot.co_optimal_transport(
                        X_a, X_b, epsilon=epsilon, max_iter=max_iter,
                    )
                    return {
                        "plan": plan_s, "feature_plan": plan_f,
                        "gw_dist": 0.0, "variant": "coot",
                    }
                except Exception:
                    logg.warning("COOT 不可用，回退到 FGW")
                    return self._run_ot(X_a, X_b, "fused_gw", alpha, epsilon, max_iter)

            elif variant == "unbalanced":
                # Unbalanced OT：使用 KL 散度惩罚边际不匹配
                try:
                    plan = ot.unbalanced.sinkhorn_knopp_unbalanced(
                        a, b, M, epsilon, reg_m=self.reg_m,
                    )
                    return {"plan": plan, "gw_dist": float(np.sum(plan * M)), "variant": "unbalanced"}
                except Exception:
                    logg.warning("Unbalanced OT 不可用，回退到 entropic OT")
                    plan = ot.sinkhorn(a, b, M, epsilon)
                    return {"plan": plan, "gw_dist": float(np.sum(plan * M)), "variant": "sinkhorn"}

            else:
                plan = ot.sinkhorn(a, b, M, epsilon)
                return {"plan": plan, "gw_dist": float(np.sum(plan * M)), "variant": "sinkhorn"}

        except ImportError:
            logg.warning("POT 未安装，使用简化 OT fallback")
            return self._fallback_ot(X_a, X_b)

    def _fallback_ot(self, X_a: np.ndarray, X_b: np.ndarray) -> dict:
        """简化 OT fallback：均值平移"""
        shift = np.mean(X_a, axis=0) - np.mean(X_b, axis=0)
        n_min = min(X_a.shape[0], X_b.shape[0])
        plan = np.eye(n_min) / n_min
        return {
            "plan": plan,
            "gw_dist": float(np.linalg.norm(shift)),
            "variant": "mean_shift_fallback",
        }

    @staticmethod
    def _measure_distribution_shift(X_before: np.ndarray, X_after: np.ndarray) -> dict:
        """测量对齐前后的分布差异"""
        n_max = min(500, min(X_before.shape[0], X_after.shape[0]))

        def rbf_mmd(X, Y, sigma=1.0):
            """RBF 核 MMD"""
            XX = np.mean(np.exp(-np.sum((X[:n_max, None] - X[None, :n_max]) ** 2, axis=-1) / (2 * sigma)))
            YY = np.mean(np.exp(-np.sum((Y[:n_max, None] - Y[None, :n_max]) ** 2, axis=-1) / (2 * sigma)))
            XY = np.mean(np.exp(-np.sum((X[:n_max, None] - Y[None, :n_max]) ** 2, axis=-1) / (2 * sigma)))
            return float(XX + YY - 2 * XY)

        wass = float(np.linalg.norm(
            np.mean(X_before[:n_max], axis=0) - np.mean(X_after[:n_max], axis=0)
        ))

        return {"mmd": rbf_mmd(X_before, X_after), "wasserstein_approx": wass}
