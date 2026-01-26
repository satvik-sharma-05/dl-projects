import spacy
import os

def load_spacy_model():
    model_path = os.path.join("spacy_models", "en_core_web_sm")
    return spacy.load(model_path)
