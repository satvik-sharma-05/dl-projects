import os
import sys
import logging

LOGGING_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

LOG_DIR = "logs"
LOG_FILE = "running_logs.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("kidneyDiseaseLogger")
