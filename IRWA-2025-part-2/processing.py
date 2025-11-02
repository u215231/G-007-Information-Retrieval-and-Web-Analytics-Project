import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download("punkt")
nltk.download("stopwords")

STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update({"nan", "none", "null", ""})

def build_terms(line: str) -> list[str]:
    """Function to process the text of a document."""
    line = line.lower()
    line = re.sub(r"[^a-zA-Z-]", " ", line) # extra
    line = line.split(" ")
    line = [word for word in line if word not in STOP_WORDS]
    line = [STEMMER.stem(word) for word in line]
    line = list(set(line)) # extra
    return line