import nltk

def setup_nltk():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
    for r in resources:
        try:
            nltk.data.find(r)
        except LookupError:
            nltk.download(r)
