"""最终 Logistic Regression 的 SHAP 可解释性分析和图形。"""

import matplotlib.pyplot as plt
import numpy as np
import shap

from .config import BINARY_FEATURES, CA_FEATURES, CONTINUOUS_FEATURES, MULTICLASS_FEATURES

def explain_model(model, background_features, features):
    """用训练集作为背景，计算待解释数据的 Logistic Regression SHAP 值。"""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    background_transformed = preprocessor.transform(background_features)
    transformed_features = preprocessor.transform(features)
    feature_names = preprocessor.get_feature_names_out().tolist()

    explainer = shap.LinearExplainer(estimator, background_transformed)
    shap_values = explainer(transformed_features)
    shap_values.feature_names = feature_names

    return shap_values


def aggregate_shap_values(shap_values):
    """将 One-Hot 编码后的 SHAP 值汇总回原始变量。"""
    encoded_names = list(shap_values.feature_names)
    groups = {}
    original_names = []

    for name in encoded_names:
        if name.startswith("multiclass__"):
            encoded_name = name.removeprefix("multiclass__")
            original_name = next(
                variable
                for variable in MULTICLASS_FEATURES
                if encoded_name.startswith(f"{variable}_")
            )
        elif name.startswith("binary__"):
            original_name = name.removeprefix("binary__")
        elif name.startswith("numeric__"):
            original_name = name.removeprefix("numeric__")
        else:
            original_name = name

        if original_name not in groups:
            groups[original_name] = []
            original_names.append(original_name)
        groups[original_name].append(encoded_names.index(name))

    aggregated_values = np.column_stack(
        [shap_values.values[:, indices].sum(axis=1) for indices in groups.values()]
    )
    aggregated_data = np.column_stack(
        [shap_values.data[:, indices].sum(axis=1) for indices in groups.values()]
    )

    return shap.Explanation(
        values=aggregated_values,
        base_values=shap_values.base_values,
        data=aggregated_data,
        feature_names=original_names,
    )


def plot_global_importance(shap_values, max_display=15):
    """绘制全局平均绝对 SHAP 值重要性图。"""
    figure = plt.figure()
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title("Global SHAP Feature Importance")
    return figure


def plot_directional_importance(shap_values, max_display=15):
    """绘制 SHAP beeswarm 图，展示特征值和风险方向。"""
    figure = plt.figure()
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Directional Summary")
    return figure


def plot_individual_explanation(shap_values, sample_index=0, max_display=15):
    """绘制单个样本的 SHAP waterfall 图。"""
    figure = plt.figure()
    shap.plots.waterfall(
        shap_values[sample_index],
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Explanation for Sample {sample_index}")
    return figure
