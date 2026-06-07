import spacy


nlp = spacy.load ("en_core_web_sm", disable=["ner", "parser"]) #loads english spacy model into nlp, whihc will be our text machine

def clean_text_spacy (text: str, remove_stopwords:bool = True) -> str:
    #lower text and send to nlp and we get a spacy doc object
    doc = nlp (text.lower()) 

    #token.lemma_ = word base form ex. run
    tokens = [token.lemma_ for token in doc if token.is_alpha and (not remove_stopwords or not token.is_stop)] #checks token isnt stopword
    return " ".join(tokens)




if __name__ == "__main__":
    sample = "Mrs. Smith's email is smith@example.com. She ran <b>10 miles</b> yesterday!"
    print(clean_text_spacy(sample))