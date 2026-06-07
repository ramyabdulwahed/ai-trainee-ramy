import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOPWORDS = set (stopwords.words ("english"))
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile (r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>") #finds html tags
EMAIL_RE = re.compile(r"\S+@\S+")#finds emails

def clean_text (text:str, remove_stopwords: bool= True) -> str:
    text =text.lower()

    #replace matched pattern with empty text
    text = URL_RE.sub ("", text)
    text = HTML_RE.sub("", text)
    text = EMAIL_RE.sub("", text)

    #deletes punctuation characters ex. '
    text = text.translate (str.maketrans("","",string.punctuation))

    #splits text into tokens based on spaces
    tokens = word_tokenize (text)

    #only keep alphateic tokens and remove else. ex. "10"
    tokens = [token for token in tokens if token.isalpha()] 

    if remove_stopwords: #if remove_stopword is on  
        tokens = [token for token in tokens if token not in STOPWORDS]

    #Lemmatizes each token. ex: miles -> mile
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    return " ".join(tokens) #turn token list to string




if __name__ == "__main__":
    sample = "Mrs. Smith's email is smith@example.com. She ran <b>10 miles</b> yesterday!"
    print(clean_text(sample))

