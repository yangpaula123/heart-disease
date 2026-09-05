"""四种分类模型。"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def build_logistic_regression():
    """构造标准 Logistic Regression Pipeline。"""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def build_lasso_logistic_regression():
    """构造使用 L1 正则化的 Logistic Regression Pipeline。"""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)),
        ]
    )


def build_random_forest():
    """构造 Random Forest Pipeline。"""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )


def build_xgboost():
    """构造 XGBoost Pipeline。"""
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError(
            "使用 XGBoost 前请安装依赖：pip install xgboost"
        ) from error
    
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("model", XGBClassifier(n_estimators=200, max_depth=3, random_state=42, eval_metric="logloss")),
        ]
    )
