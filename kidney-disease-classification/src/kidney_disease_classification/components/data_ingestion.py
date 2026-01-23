import os
import zipfile
import gdown
from pathlib import Path
from kidney_disease_classification import logger
from kidney_disease_classification.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> None:
        """
        Download data from Google Drive
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_path = self.config.local_data_file

            os.makedirs(zip_download_path.parent, exist_ok=True)

            logger.info(f"Downloading data from {dataset_url}")

            # ✅ CORRECT way (convert Path → str)
            gdown.download(
                url=dataset_url,
                output=str(zip_download_path),
                quiet=False,
                fuzzy=True
            )

            logger.info(f"Downloaded file saved at: {zip_download_path}")

        except Exception as e:
            logger.exception(e)
            raise e

    def extract_zip_file(self) -> None:
        """
        Extract zip file
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            logger.info(f"Extracting zip file: {self.config.local_data_file}")

            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info(f"Extraction completed at: {unzip_path}")

        except Exception as e:
            logger.exception(e)
            raise e
