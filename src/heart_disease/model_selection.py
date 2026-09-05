"""训练集上的模型选择和交叉验证。"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from .models import (
    build_lasso_logistic_regression,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)


def fit_final_logistic_regression(features, target):
    """使用全部清理后的训练集拟合最终逻辑回归 Pipeline。"""
    model = build_logistic_regression()
    model.fit(features, target)
    return model


def cross_validate_logistic_regression(
    features,
    target,
    n_splits=5,
    random_state=42,
):
    """五折评估逻辑回归，同时记录 AUC 和 Brier score。"""
    model = build_logistic_regression()
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    results = cross_validate(
        estimator=model,
        X=features,
        y=target,
        cv=cv,
        scoring={"auc": "roc_auc", "brier": "neg_brier_score"},
        return_train_score=False,
    )
    fold_brier = -results["test_brier"]
    return {
        "fold_auc": results["test_auc"],
        "mean_auc": results["test_auc"].mean(),
        "std_auc": results["test_auc"].std(),
        "fold_brier": fold_brier,
        "mean_brier": fold_brier.mean(),
        "std_brier": fold_brier.std(),
    }


def tune_lasso_logistic_regression(
    features,
    target,
    c_values=None,
    n_splits=5,
    random_state=42,
):
    """使用五折交叉验证搜索 L1 逻辑回归的最佳 C。"""
    if c_values is None:
        c_values = [0.01, 0.1, 1, 10, 100]

    model = build_lasso_logistic_regression()
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    search = GridSearchCV(
        estimator=model,
        param_grid={"model__C": c_values},
        scoring={"auc": "roc_auc", "brier": "neg_brier_score"},
        cv=cv,
        refit=False,
        return_train_score=False,
    )
    search.fit(features, target)
    best_index = search.cv_results_["mean_test_auc"].argmax()

    return {
        "search": search,
        "c_values": c_values,
        "best_c": search.cv_results_["params"][best_index]["model__C"],
        "best_mean_auc": search.cv_results_["mean_test_auc"][best_index],
        "best_std_auc": _score_std(search.cv_results_, "auc", best_index),
        "best_mean_brier": -search.cv_results_["mean_test_brier"][best_index],
        "best_std_brier": _score_std(search.cv_results_, "brier", best_index),
        "results": search.cv_results_,
    }


def tune_random_forest(
    features,
    target,
    n_splits=5,
    random_state=42,
):
    """使用五折交叉验证搜索随机森林参数。"""
    model = build_random_forest()
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    param_grid = {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring={"auc": "roc_auc", "brier": "neg_brier_score"},
        cv=cv,
        refit=False,
        return_train_score=False,
    )
    search.fit(features, target)
    best_index = search.cv_results_["mean_test_auc"].argmax()

    return {
        "search": search,
        "param_grid": param_grid,
        "best_params": search.cv_results_["params"][best_index],
        "best_mean_auc": search.cv_results_["mean_test_auc"][best_index],
        "best_std_auc": _score_std(search.cv_results_, "auc", best_index),
        "best_mean_brier": -search.cv_results_["mean_test_brier"][best_index],
        "best_std_brier": _score_std(search.cv_results_, "brier", best_index),
        "results": search.cv_results_,
    }


def tune_xgboost(
    features,
    target,
    n_splits=5,
    random_state=42,
):
    """使用五折交叉验证搜索 XGBoost 参数。"""
    model = build_xgboost()
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    param_grid = {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.1],
    }
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring={"auc": "roc_auc", "brier": "neg_brier_score"},
        cv=cv,
        refit=False,
        return_train_score=False,
    )
    search.fit(features, target)
    best_index = search.cv_results_["mean_test_auc"].argmax()

    return {
        "search": search,
        "param_grid": param_grid,
        "best_params": search.cv_results_["params"][best_index],
        "best_mean_auc": search.cv_results_["mean_test_auc"][best_index],
        "best_std_auc": _score_std(search.cv_results_, "auc", best_index),
        "best_mean_brier": -search.cv_results_["mean_test_brier"][best_index],
        "best_std_brier": _score_std(search.cv_results_, "brier", best_index),
        "results": search.cv_results_,
    }


def _score_std(results, metric, index):
    """计算最佳参数组合在五折中的指标标准差。"""
    fold_scores = [
        results[f"split{fold}_test_{metric}"][index]
        for fold in range(5)
    ]
    if metric == "brier":
        fold_scores = [-score for score in fold_scores]
    return sum((score - sum(fold_scores) / len(fold_scores)) ** 2 for score in fold_scores) ** 0.5 / len(fold_scores) ** 0.5
