from pathlib import Path

# Project root = kidney-disease-classification/
ROOT_DIR = Path(__file__).resolve().parents[3]

CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"

print("CONFIG:", CONFIG_FILE_PATH)
print("PARAMS:", PARAMS_FILE_PATH)
