"""缺失值处理、特征编码和特征缩放。"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BINARY_FEATURES,
    CA_FEATURES,
    CONTINUOUS_FEATURES,
    MULTICLASS_FEATURES,
)


def check_missing_values(data):
    """检查数据中的缺失值数量和比例。"""
    print("缺失值数量：")
    print(data.isna().sum())

    print("缺失值比例（%）：")
    print((data.isna().mean() * 100).round(2))


def remove_missing_rows(features, target):
    """删除特征存在缺失值的样本，并保持特征和目标对齐。"""
    valid_rows = ~features.isna().any(axis=1)
    features = features.loc[valid_rows]
    target = target.loc[valid_rows]
    return features, target


def check_preprocessed_data(features, target):
    """检查删除缺失样本后的数据质量。"""
    print("=== 数据形状 ===")
    print("特征数据：", features.shape)
    print("目标变量：", target.shape)

    print("=== 缺失值数量 ===")
    print(features.isna().sum())

    print("=== target 分布 ===")
    print(target.value_counts())
    print("target 比例：")
    print(target.value_counts(normalize=True).round(4))

    print("=== 重复样本数量 ===")
    print(features.duplicated().sum())

    print("=== 特征与目标变量行数是否一致 ===")
    print(len(features) == len(target))


def build_preprocessor(scale_numeric=True):
    """构造适用于当前模型的编码和缩放流程。"""
    numeric_features = CONTINUOUS_FEATURES + CA_FEATURES
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    multiclass_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("binary", "passthrough", BINARY_FEATURES),
            ("numeric", numeric_transformer, numeric_features),
            ("multiclass", multiclass_transformer, MULTICLASS_FEATURES),
        ]
    )

    


