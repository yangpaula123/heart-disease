"""最终测试集评价：区分能力、校准、分类指标和 DCA。"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(
    model,
    features,
    target,
    threshold=0.5,
    bootstrap_iterations=2000,
    random_state=42,
):
    """计算最终测试集指标，并返回预测概率和类别。"""
    probability = model.predict_proba(features)[:, 1]
    prediction = (probability >= threshold).astype(int)
    auc = roc_auc_score(target, probability)
    auc_ci = bootstrap_auc_ci(
        target,
        probability,
        iterations=bootstrap_iterations,
        random_state=random_state,
    )

    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    calibration_intercept, calibration_slope = calibration_parameters(target, probability)
    classification_ci = bootstrap_classification_ci(
        target,
        prediction,
        iterations=bootstrap_iterations,
        random_state=random_state,
    )

    return {
        "auc": auc,
        "auc_ci_lower": auc_ci[0],
        "auc_ci_upper": auc_ci[1],
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "brier_score": brier_score_loss(target, probability),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "classification_ci": classification_ci,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "probability": probability,
        "prediction": prediction,
    }


def bootstrap_auc_ci(target, probability, iterations=2000, confidence_level=0.95, random_state=42):
    """使用 bootstrap 计算 ROC-AUC 的百分位 95% 置信区间。"""
    target = np.asarray(target)
    probability = np.asarray(probability)
    rng = np.random.default_rng(random_state)
    auc_values = []

    for _ in range(iterations):
        indices = rng.integers(0, len(target), len(target))
        sampled_target = target[indices]
        if np.unique(sampled_target).size < 2:
            continue
        auc_values.append(roc_auc_score(sampled_target, probability[indices]))

    if not auc_values:
        return np.nan, np.nan

    alpha = 1 - confidence_level
    return tuple(np.quantile(auc_values, [alpha / 2, 1 - alpha / 2]))


def calibration_parameters(target, probability):
    """估计校准截距和校准斜率。理想值分别为 0 和 1。"""
    target = np.asarray(target)
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    logit_probability = np.log(probability / (1 - probability))

    def negative_log_likelihood(parameters):
        intercept, slope = parameters
        linear_predictor = intercept + slope * logit_probability
        log_likelihood = target * linear_predictor - np.logaddexp(0, linear_predictor)
        return -log_likelihood.sum()

    result = minimize(
        negative_log_likelihood,
        x0=np.array([0.0, 1.0]),
        method="BFGS",
    )
    return tuple(result.x)


def bootstrap_classification_ci(
    target,
    prediction,
    iterations=2000,
    confidence_level=0.95,
    random_state=42,
):
    """使用 bootstrap 计算四个分类指标的置信区间。"""
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    rng = np.random.default_rng(random_state)
    metric_values = {
        "sensitivity": [],
        "specificity": [],
        "ppv": [],
        "npv": [],
    }

    for _ in range(iterations):
        indices = rng.integers(0, len(target), len(target))
        sampled_target = target[indices]
        sampled_prediction = prediction[indices]
        if np.unique(sampled_target).size < 2:
            continue

        tn, fp, fn, tp = confusion_matrix(
            sampled_target,
            sampled_prediction,
            labels=[0, 1],
        ).ravel()
        metric_values["sensitivity"].append(tp / (tp + fn) if tp + fn else np.nan)
        metric_values["specificity"].append(tn / (tn + fp) if tn + fp else np.nan)
        metric_values["ppv"].append(tp / (tp + fp) if tp + fp else np.nan)
        metric_values["npv"].append(tn / (tn + fn) if tn + fn else np.nan)

    alpha = 1 - confidence_level
    return {
        metric: tuple(np.nanquantile(values, [alpha / 2, 1 - alpha / 2]))
        for metric, values in metric_values.items()
    }


def plot_roc_curve(target, probability):
    """绘制 ROC 曲线并返回图形对象。"""
    false_positive_rate, true_positive_rate, _ = roc_curve(target, probability)
    auc = roc_auc_score(target, probability)
    figure, axis = plt.subplots()
    axis.plot(false_positive_rate, true_positive_rate, label=f"Model (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], "--", label="Chance")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend()
    return figure, axis


def plot_calibration_curve(target, probability, n_bins=5):
    """绘制预测概率校准曲线并返回图形对象。"""
    fraction_positive, mean_predicted_value = calibration_curve(
        target,
        probability,
        n_bins=n_bins,
        strategy="uniform",
    )
    figure, axis = plt.subplots()
    axis.plot(mean_predicted_value, fraction_positive, "o-", label="Model")
    axis.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed fraction positive")
    axis.set_title("Calibration Curve")
    axis.legend()
    return figure, axis


def decision_curve_data(target, probability, thresholds=None):
    """计算模型、Treat-all 和 Treat-none 的 DCA 净获益。"""
    target = np.asarray(target)
    probability = np.asarray(probability)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.80, 100)

    prevalence = target.mean()
    model_net_benefit = []
    treat_all_net_benefit = []
    treat_none_net_benefit = np.zeros(len(thresholds))

    for threshold in thresholds:
        predicted_positive = probability >= threshold
        true_positive = np.sum(predicted_positive & (target == 1))
        false_positive = np.sum(predicted_positive & (target == 0))
        model_net_benefit.append(
            true_positive / len(target)
            - false_positive / len(target) * threshold / (1 - threshold)
        )
        treat_all_net_benefit.append(
            prevalence - (1 - prevalence) * threshold / (1 - threshold)
        )

    return {
        "thresholds": np.asarray(thresholds),
        "model": np.asarray(model_net_benefit),
        "treat_all": np.asarray(treat_all_net_benefit),
        "treat_none": treat_none_net_benefit,
    }


def plot_decision_curve(target, probability, thresholds=None):
    """绘制决策曲线并返回图形对象。"""
    curve = decision_curve_data(target, probability, thresholds)
    figure, axis = plt.subplots()
    axis.plot(curve["thresholds"], curve["model"], label="Model")
    axis.plot(curve["thresholds"], curve["treat_all"], "--", label="Treat all")
    axis.plot(curve["thresholds"], curve["treat_none"], ":", label="Treat none")
    axis.set_xlabel("Threshold probability")
    axis.set_ylabel("Net benefit")
    axis.set_title("Decision Curve Analysis")
    axis.legend()
    return figure, axis
