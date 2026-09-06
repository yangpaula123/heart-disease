"""统一实验入口：按顺序调用各个模块。"""

import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from heart_disease.data import (
    create_target,
    load_data,
    split_features_target,
    split_train_test,
)
from heart_disease.config import FIGURES_DIR, RESULTS_DIR
from heart_disease.preprocessing import (
    check_missing_values,
    check_preprocessed_data,
    remove_missing_rows,
)
from heart_disease.model_selection import (
    cross_validate_logistic_regression,
    fit_final_logistic_regression,
    tune_lasso_logistic_regression,
    tune_random_forest,
    tune_xgboost,
)
from heart_disease.evaluation import (
    evaluate_model,
    plot_calibration_curve,
    plot_decision_curve,
    plot_roc_curve,
)
from heart_disease.bootstrap_validation import (
    bootstrap_optimism_correction,
    optimism_summary_table,
    plot_optimism_distribution,
)

from heart_disease.explainability import (
    aggregate_shap_values,
    explain_model,
    plot_directional_importance,
    plot_global_importance,
    plot_individual_explanation,
)

def main():
    data = load_data()
    data = create_target(data)

    X, y = split_features_target(data)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    check_missing_values(X_train)

    X_train, y_train = remove_missing_rows(X_train, y_train)
    X_test, y_test = remove_missing_rows(X_test, y_test)

    print("删除缺失样本后：")
    print("训练集：", X_train.shape)
    print("测试集：", X_test.shape)

    print("\n=== 训练集质量检查 ===")
    check_preprocessed_data(X_train, y_train)

    print("\n=== 测试集质量检查 ===")
    check_preprocessed_data(X_test, y_test)

    print("\n=== Logistic Regression 五折交叉验证 ===")
    cv_result = cross_validate_logistic_regression(X_train, y_train)
    print("每折 ROC-AUC：")
    for fold_number, fold_auc in enumerate(cv_result["fold_auc"], start=1):
        print(f"第 {fold_number} 折：{fold_auc:.4f}")
    print(f"平均 ROC-AUC：{cv_result['mean_auc']:.4f}")
    print(f"ROC-AUC 标准差：{cv_result['std_auc']:.4f}")
    print(f"平均 Brier score：{cv_result['mean_brier']:.4f}")
    print(f"Brier score 标准差：{cv_result['std_brier']:.4f}")

    print("\n=== 其他模型五折参数搜索 ===")
    lasso_result = tune_lasso_logistic_regression(X_train, y_train)
    random_forest_result = tune_random_forest(X_train, y_train)
    xgboost_result = tune_xgboost(X_train, y_train)
    model_results = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "best_params": json.dumps({}, ensure_ascii=False),
                "mean_auc": cv_result["mean_auc"],
                "std_auc": cv_result["std_auc"],
                "mean_brier": cv_result["mean_brier"],
                "std_brier": cv_result["std_brier"],
            },
            {
                "model": "lasso_logistic_regression",
                "best_params": json.dumps({"C": lasso_result["best_c"]}),
                "mean_auc": lasso_result["best_mean_auc"],
                "std_auc": lasso_result["best_std_auc"],
                "mean_brier": lasso_result["best_mean_brier"],
                "std_brier": lasso_result["best_std_brier"],
            },
            {
                "model": "random_forest",
                "best_params": json.dumps(random_forest_result["best_params"], default=str),
                "mean_auc": random_forest_result["best_mean_auc"],
                "std_auc": random_forest_result["best_std_auc"],
                "mean_brier": random_forest_result["best_mean_brier"],
                "std_brier": random_forest_result["best_std_brier"],
            },
            {
                "model": "xgboost",
                "best_params": json.dumps(xgboost_result["best_params"], default=str),
                "mean_auc": xgboost_result["best_mean_auc"],
                "std_auc": xgboost_result["best_std_auc"],
                "mean_brier": xgboost_result["best_mean_brier"],
                "std_brier": xgboost_result["best_std_brier"],
            },
        ]
    ).sort_values("mean_auc", ascending=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(model_results[["model", "mean_auc", "mean_brier"]].to_string(index=False))
    print("已保存：", RESULTS_DIR / "model_comparison.csv")

    print("\n=== 使用全部训练集拟合最终 Logistic Regression ===")
    final_model = fit_final_logistic_regression(X_train, y_train)
    print("拟合样本数：", len(X_train))
    print("最终模型：", final_model.named_steps["model"])

    print("\n=== Bootstrap 乐观校正内部验证（训练集） ===")
    validation_result = bootstrap_optimism_correction(final_model, X_train, y_train)
    validation_table = optimism_summary_table(validation_result)
    print(f"有效重抽样次数：{validation_result['n_resamples']} / 1000")
    print(validation_table.round(4).to_string(index=False))
    validation_table.to_csv(RESULTS_DIR / "bootstrap_validation.csv", index=False)
    validation_result["records"].to_csv(
        RESULTS_DIR / "bootstrap_validation_records.csv",
        index=False,
    )
    print("已保存：", RESULTS_DIR / "bootstrap_validation.csv")
    print("已保存：", RESULTS_DIR / "bootstrap_validation_records.csv")

    print("\n=== 测试集最终评价 ===")
    evaluation_result = evaluate_model(final_model, X_test, y_test)
    print(f"ROC-AUC：{evaluation_result['auc']:.4f}")
    print(
        "ROC-AUC 95% CI："
        f"({evaluation_result['auc_ci_lower']:.4f}, "
        f"{evaluation_result['auc_ci_upper']:.4f})"
    )
    print(f"Sensitivity：{evaluation_result['sensitivity']:.4f}")
    print(f"Specificity：{evaluation_result['specificity']:.4f}")
    print(f"PPV：{evaluation_result['ppv']:.4f}")
    print(f"NPV：{evaluation_result['npv']:.4f}")
    print(f"Brier score：{evaluation_result['brier_score']:.4f}")
    print(
        "校准截距："
        f"{evaluation_result['calibration_intercept']:.4f}"
    )
    print(f"校准斜率：{evaluation_result['calibration_slope']:.4f}")
    print("分类指标 95% CI：")
    for metric, interval in evaluation_result["classification_ci"].items():
        print(f"{metric}: ({interval[0]:.4f}, {interval[1]:.4f})")
    print("Confusion matrix：", evaluation_result["confusion_matrix"])

    print("\n=== 测试集评价图 ===")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_figures = {
        "roc_curve.png": plot_roc_curve(y_test, evaluation_result["probability"])[0],
        "calibration_curve.png": plot_calibration_curve(
            y_test,
            evaluation_result["probability"],
            n_bins=5,
        )[0],
        "decision_curve.png": plot_decision_curve(
            y_test,
            evaluation_result["probability"],
        )[0],
        "bootstrap_optimism.png": plot_optimism_distribution(
            validation_result["records"],
        )[0],
    }
    for filename, figure in evaluation_figures.items():
        figure.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)
        print("已保存：", FIGURES_DIR / filename)

    print("\n=== SHAP 可解释性分析 ===")
    shap_values = explain_model(final_model, X_train, X_test)
    aggregated_shap_values = aggregate_shap_values(shap_values)
    shap_figures = {
        "shap_global_importance.png": plot_global_importance(aggregated_shap_values),
        "shap_directional_summary.png": plot_directional_importance(aggregated_shap_values),
        "shap_individual_sample_0.png": plot_individual_explanation(
            aggregated_shap_values,
            sample_index=0,
        ),
    }
    for filename, figure in shap_figures.items():
        figure.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)
        print("已保存：", FIGURES_DIR / filename)


if __name__ == "__main__":
    main()