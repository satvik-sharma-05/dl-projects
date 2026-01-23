import os
import sys
import logging
import yaml
import json
import joblib
import base64

from pathlib import Path
from typing import Iterable, List, Any   # ✅ FIX HERE
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from kidney_disease_classification import logger



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read YAML file and return ConfigBox"""

    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)

            if content is None:
                raise BoxValueError("YAML file is empty")

            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e




def create_directories(path_to_directories: Iterable[Path], verbose: bool = True):
    """Create list of directories"""

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Save JSON file"""

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load JSON file"""

    with open(path, "r") as f:
        content = json.load(f)

    logger.info(f"JSON file loaded from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save binary file"""

    joblib.dump(data, path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary file"""

    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Get file size in KB"""

    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring: str, fileName: str):
    """Decode base64 image"""

    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)


def encodeImageIntoBase64(imagePath: str) -> bytes:
    """Encode image to base64"""

    with open(imagePath, "rb") as f:
        return base64.b64encode(f.read())

import os
import zipfile
import gdown
from kidney_disease_classification import logger
from pathlib import Path


def download_file_from_google_drive(url: str, output_path: Path):
    """
    Downloads file from Google Drive
    """
    os.makedirs(output_path.parent, exist_ok=True)

    logger.info(f"Downloading data from: {url}")
    gdown.download(url, str(output_path), quiet=False)
    logger.info(f"Downloaded file saved at: {output_path}")


def unzip_file(zip_file_path: Path, unzip_dir: Path):
    """
    Unzips a zip file
    """
    logger.info(f"Unzipping file: {zip_file_path}")
    os.makedirs(unzip_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(unzip_dir)

    logger.info(f"Unzipped data to: {unzip_dir}")
