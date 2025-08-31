from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DUCKDB_PATH = RAW_DATA_DIR / "index.duckdb"
RAW_RUREVIEWS_PATH = RAW_DATA_DIR / "RuReviews.csv"
WB_RAW_PATH = RAW_DATA_DIR / "raw_WB.csv"
WB_CLEANED_PATH = PROCESSED_DATA_DIR / "WB_cleaned.csv"
RUREVIEWS_PATH = PROCESSED_DATA_DIR / "RuReviews_cleaned.csv"

TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"