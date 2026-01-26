import subprocess
import sys
import nltk

def setup_spacy():
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except:
        subprocess.run([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])

def setup_nltk():
    nltk_packages = [
        "punkt",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger"
    ]
    for pkg in nltk_packages:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)

def setup_all():
    setup_spacy()
    setup_nltk()
