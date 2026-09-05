"""项目配置：数据路径、字段名称和变量分组。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
FIGURES_DIR = PROJECT_ROOT / "figure"
RESULTS_DIR = PROJECT_ROOT / "results"

COLUMNS = [
	"age",
	"sex",
	"cp",
	"trestbps",
	"chol",
	"fbs",
	"restecg",
	"thalach",
	"exang",
	"oldpeak",
	"slope",
	"ca",
	"thal",
	"num",
]

CONTINUOUS_FEATURES = [
	"age",
	"trestbps",
	"chol",
	"thalach",
	"oldpeak",
]

BINARY_FEATURES = [
	"sex",
	"fbs",
	"exang",
]

CA_FEATURES = ["ca"]

MULTICLASS_FEATURES = [
	"cp",
	"restecg",
	"slope",
	"thal",
]

CATEGORICAL_FEATURES = BINARY_FEATURES + MULTICLASS_FEATURES
NUMERIC_FEATURES = CONTINUOUS_FEATURES + CA_FEATURES

TARGET = "target"
