import argparse

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind

from heart_disease.config import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, TARGET
from heart_disease.data import create_target, load_data


def show_basic_information(data):
    """查看数据规模、类型、描述统计和目标分布。"""
    print("数据形状：", data.shape)
    data.info()
    print(data.describe())
    print(data["num"].value_counts())
    print(data[TARGET].value_counts())


def analyze_missing_values(data):
    """分析缺失值数量、比例、模式及其与其他变量的关系。"""
    print("=== 缺失值数量 ===")
    print(data.isna().sum())
    print("=== 缺失值比例（%） ===")
    print((data.isna().mean() * 100).round(2))

    missing_summary = data.isna().sum()
    missing_columns = missing_summary[missing_summary > 0].index
    print("=== 存在缺失值的变量 ===")
    print(missing_summary[missing_summary > 0])
    print("=== 存在缺失值的样本数 ===")
    print(data.isna().any(axis=1).sum())
    print("=== 缺失模式 ===")
    print(data.isna().sum(axis=1).value_counts().sort_index())

    print("=== 缺失情况与其他变量的关系 ===")
    for missing_column in missing_columns:
        missing_flag = data[missing_column].isna().astype(int)
        print(f"\n--- {missing_column} 的缺失情况与 target 的关系 ---")
        print(pd.crosstab(missing_flag, data[TARGET]))
        print("按缺失状态的 target 比例（%）：")
        print(pd.crosstab(missing_flag, data[TARGET], normalize="index").mul(100).round(2))

        print(f"--- {missing_column} 的缺失情况与连续变量的关系 ---")
        print(data[CONTINUOUS_FEATURES].groupby(missing_flag).agg(["count", "mean", "median"]).round(2))

        print(f"--- {missing_column} 的缺失情况与分类变量的关系 ---")
        for variable in CATEGORICAL_FEATURES:
            print(f"{variable}:")
            print(pd.crosstab(missing_flag, data[variable], normalize="index", dropna=False).mul(100).round(2))


def analyze_continuous_variables(data):
    """分析连续变量的描述统计、分布和组间差异。"""
    print("=== 连续变量描述统计 ===")
    print(data[CONTINUOUS_FEATURES].describe())
    grouped_stats = data.groupby(TARGET)[CONTINUOUS_FEATURES].agg(["mean", "std", "median", "min", "max"])
    print("=== 不同疾病状态下的连续变量比较 ===")
    print(grouped_stats)

    print("=== 连续变量分布图 ===")
    for variable in CONTINUOUS_FEATURES:
        plt.figure()
        plt.hist(data[variable].dropna(), bins=20)
        plt.title(f"Distribution of {variable}")
        plt.xlabel(variable)
        plt.ylabel("Frequency")
        plt.show()

    print("=== age 独立样本 t 检验 ===")
    age_no_disease = data.loc[data[TARGET] == 0, "age"].dropna()
    age_has_disease = data.loc[data[TARGET] == 1, "age"].dropna()
    t_stat, p_value = ttest_ind(age_no_disease, age_has_disease, equal_var=False)
    print(f"t-statistic: {t_stat}, p-value: {p_value}")

    print("=== Mann-Whitney U 检验 ===")
    for variable in ["trestbps", "chol", "thalach", "oldpeak"]:
        group_0 = data.loc[data[TARGET] == 0, variable].dropna()
        group_1 = data.loc[data[TARGET] == 1, variable].dropna()
        u_stat, p_value = mannwhitneyu(group_0, group_1, alternative="two-sided")
        print(f"\n变量：{variable}")
        print(f"无患病组中位数：{group_0.median():.3f}")
        print(f"患病组中位数：{group_1.median():.3f}")
        print(f"U-statistic={u_stat:.3f}, p-value={p_value:.6g}")


def analyze_categorical_variables(data):
    """分析分类变量的频数及其与 target 的关系。"""
    print("=== 分类变量频数统计 ===")
    for variable in CATEGORICAL_FEATURES:
        print(f"\n--- {variable} ---")
        print(data[variable].value_counts(dropna=False))

    print("=== 按疾病状态分组的分类变量统计 ===")
    for variable in CATEGORICAL_FEATURES:
        print(f"\n--- {variable} ---")
        count_table = pd.crosstab(data[variable], data[TARGET], dropna=False)
        print("频数：")
        print(count_table)
        proportion_table = pd.crosstab(data[variable], data[TARGET], normalize="columns", dropna=False) * 100
        print("比例（%）：")
        print(proportion_table.round(2))


def test_categorical_variables(data):
    """使用卡方检验分析分类变量与 target 的关系。"""
    print("=== 分类变量的卡方检验 ===")
    for variable in CATEGORICAL_FEATURES:
        table = pd.crosstab(data[variable], data[TARGET])
        chi2, p_value, _, _ = chi2_contingency(table)
        print(f"{variable}: Chi-square = {chi2:.3f}, p = {p_value:.6g}")


def parse_arguments():
    """解析命令行参数，决定运行哪个分析模块。"""
    parser = argparse.ArgumentParser(description="心脏病数据探索性分析")
    parser.add_argument(
        "--analysis",
        choices=["basic", "missing", "continuous", "categorical", "chi2", "all"],
        default="basic",
        help="选择要运行的分析模块，默认只运行 basic",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    data = create_target(load_data())
    analysis_functions = {
        "basic": show_basic_information,
        "missing": analyze_missing_values,
        "continuous": analyze_continuous_variables,
        "categorical": analyze_categorical_variables,
        "chi2": test_categorical_variables,
    }

    if arguments.analysis == "all":
        for analysis_function in analysis_functions.values():
            analysis_function(data)
    else:
        analysis_functions[arguments.analysis](data)


if __name__ == "__main__":
    main()
