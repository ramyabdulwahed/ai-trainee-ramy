from sklearn.feature_extraction.text import TfidfVectorizer #Turns raw text into TF-IDF number features
from sklearn.linear_model import LogisticRegression #Ml classifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



#Create one object called pipe that has two steps inside it:
    #Those two steps are: Step 1: "tfidf" Raw text -> numbers 
    # Step 2: "clf" Numbers → prediction
pipe = Pipeline ([
    #TF-IDF settings
    #ngram_range=(1, 2) : use single or two word phrases. ex.this matters since good isnt same as not good
    #min_df=2 : ignore words that appear in less than 2 documents
    #max_df =0.95 : ignore words/phrases that appear in more than 95% of documents

    ##linear regression settings
    #max_iter=1000 : This gives the model more time/iterations to learn
    #n_jobs=-1 : tell scikit to use all available cpu cores
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)), #convert text into numbers. tfidf is just a label
    ("clf", LogisticRegression(max_iter=1000, n_jobs=-1)), #When the classifier eventually trains, use these training settings
])