import string

from cli.load_data import get_stopwords
from nltk.stem import PorterStemmer

stopwords = get_stopwords()
stemmer = PorterStemmer()

def process_str(data: str) -> list[str]:
    data = data.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = list(filter(lambda x: x != " ", data.split()))
    tokens = list(filter(lambda x: x not in stopwords, tokens))
    tokens = list(map(lambda x: stemmer.stem(x), tokens))
    return tokens
