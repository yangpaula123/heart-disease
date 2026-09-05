"""数据读取和数据集拆分。"""

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import COLUMNS, DATA_PATH


def load_data():
    """读取 Cleveland 原始数据。"""
    data = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values="?",
    )

    return data


def create_target(data):
    """根据原始疾病程度 num 创建二分类目标 target。"""
    data = data.copy()
    data["target"] = (data["num"] > 0).astype(int)
    return data


def split_features_target(data):
    """将数据拆分为特征变量 X 和目标变量 y。"""
    features = data.drop(columns=["num", "target"])
    target = data["target"]

    return features, target


def split_train_test(features, target, test_size=0.2, random_state=42):
    """按 8:2 划分训练集和测试集，并按 target 分层抽样。"""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
