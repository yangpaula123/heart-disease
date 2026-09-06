"""Bootstrap 乐观校正内部验证（Harrell 方法）。"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score

from .evaluation import calibration_parameters


def bootstrap_optimism_correction(
    model,
    features,
    target,
    n_iterations=1000,
    confidence_level=0.95,
    random_state=42,
):
    """估计最终模型的拟合乐观性，并给出校正后的指标。

    原理（Efron 1983 / Harrell 1996）：

        Corrected = Apparent − mean(Apparent_boot − Test_boot)

    每次重抽样用 clone 得到未拟合的同一 Pipeline（含预处理器），
    在 bootstrap 样本上重新拟合并评估：
    - apparent_boot：bootstrap 样本本身上的表现
    - test_boot：同一模型在原始样本上的表现

    无惩罚（MLE）逻辑回归在拟合数据上的表观校准斜率恒为 1、
    截距恒为 0（得分方程的直接推论）。本项目默认 C=1 带 L2
    惩罚，系数被压缩后预测概率欠分散，表观斜率会大于 1；
    乐观校正同样适用，校正后斜率直接量化过拟合收缩量。

    注意：校正只在传入的开发样本（本项目为训练集）上进行，
    测试集必须保持独立，不参与任何一步。
    """
    n = len(target)
    rng = np.random.default_rng(random_state)
    records = []

    for _ in range(n_iterations):
        indices = rng.integers(0, n, n)
        sampled_target = target.iloc[indices]
        if sampled_target.nunique() < 2:
            continue  # 极端重抽样只含一个类别，无法计算 AUC

        model_boot = clone(model)
        model_boot.fit(features.iloc[indices], sampled_target)
        probability_boot = model_boot.predict_proba(features.iloc[indices])[:, 1]
        probability_full = model_boot.predict_proba(features)[:, 1]
        intercept_boot, slope_boot = calibration_parameters(
            sampled_target, probability_boot
        )
        intercept_test, slope_test = calibration_parameters(
            target, probability_full
        )

        records.append(
            {
                "auc_apparent": roc_auc_score(sampled_target, probability_boot),
                "auc_test": roc_auc_score(target, probability_full),
                "brier_apparent": brier_score_loss(sampled_target, probability_boot),
                "brier_test": brier_score_loss(target, probability_full),
                "slope_apparent": slope_boot,
                "slope_test": slope_test,
                "intercept_apparent": intercept_boot,
                "intercept_test": intercept_test,
            }
        )

    if not records:
        raise ValueError("所有重抽样都只包含一个类别，无法估计乐观性。")

    records = pd.DataFrame(records)

    probability_final = model.predict_proba(features)[:, 1]
    apparent = {
        "auc": roc_auc_score(target, probability_final),
        "brier": brier_score_loss(target, probability_final),
        "slope": calibration_parameters(target, probability_final)[1],
        "intercept": calibration_parameters(target, probability_final)[0],
    }

    column_pairs = {
        "auc": ("auc_apparent", "auc_test"),
        "brier": ("brier_apparent", "brier_test"),
        "slope": ("slope_apparent", "slope_test"),
        "intercept": ("intercept_apparent", "intercept_test"),
    }
    alpha = 1 - confidence_level
    mean_optimism = {}
    corrected = {}
    corrected_ci = {}
    for metric, (apparent_column, test_column) in column_pairs.items():
        mean_optimism[metric] = (
            records[apparent_column] - records[test_column]
        ).mean()
        corrected[metric] = apparent[metric] - mean_optimism[metric]
        # 校正值的置信区间用各次 test 表现的百分位区间近似
        corrected_ci[metric] = tuple(
            np.quantile(records[test_column], [alpha / 2, 1 - alpha / 2])
        )

    return {
        "apparent": apparent,
        "mean_optimism": mean_optimism,
        "corrected": corrected,
        "corrected_ci": corrected_ci,
        "records": records,
        "n_resamples": len(records),
    }


def optimism_summary_table(result):
    """把乐观校正结果整理成指标 × 数值的表格。"""
    rows = []
    for metric in ("auc", "brier", "slope", "intercept"):
        ci_lower, ci_upper = result["corrected_ci"][metric]
        rows.append(
            {
                "metric": metric,
                "apparent": result["apparent"][metric],
                "mean_optimism": result["mean_optimism"][metric],
                "corrected": result["corrected"][metric],
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
    return pd.DataFrame(rows)


def plot_optimism_distribution(records):
    """绘制 AUC 乐观性与校准斜率（test）的分布直方图。"""
    auc_optimism = records["auc_apparent"] - records["auc_test"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(
        auc_optimism,
        bins=25,
        color="#2a78d6",
        edgecolor="white",
        linewidth=0.6,
    )
    axes[0].axvline(
        auc_optimism.mean(),
        color="#1c5cab",
        linewidth=1.5,
        label=f"Mean = {auc_optimism.mean():.4f}",
    )
    axes[0].set_xlabel("AUC optimism (apparent − test)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Bootstrap optimism in AUC")
    axes[0].legend()

    axes[1].hist(
        records["slope_test"],
        bins=25,
        color="#2a78d6",
        edgecolor="white",
        linewidth=0.6,
    )
    axes[1].axvline(
        1.0,
        color="#898781",
        linewidth=1.5,
        linestyle="--",
        label="Ideal slope = 1",
    )
    axes[1].axvline(
        records["slope_test"].mean(),
        color="#1c5cab",
        linewidth=1.5,
        label=f"Mean = {records['slope_test'].mean():.4f}",
    )
    axes[1].set_xlabel("Calibration slope on full sample")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Bootstrap test calibration slope")
    axes[1].legend()

    return figure, axes
