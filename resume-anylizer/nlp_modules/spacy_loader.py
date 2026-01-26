import spacy
import subprocess
import sys
from spacy.util import is_package

def load_spacy_model(model_name="en_core_web_sm"):
    if not is_package(model_name):
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", model_name]
        )
    return spacy.load(model_name)
