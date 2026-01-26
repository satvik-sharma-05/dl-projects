import spacy
from spacy.util import is_package

def load_spacy_model():
    """
    Loads spaCy model safely for Streamlit Cloud.
    Assumes model is already available in the environment.
    """
    return spacy.blank("en") if not is_package("en_core_web_sm") else spacy.load("en_core_web_sm")
