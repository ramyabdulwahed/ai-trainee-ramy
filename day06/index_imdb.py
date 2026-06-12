#pipeline
# IMDB CSV file
#    ↓
# pandas reads it
#    ↓
# we create an Elasticsearch index
#    ↓
# we define mapping: text / sentiment / length
#    ↓
# we bulk insert 25,000 reviews
#    ↓
# we search them with BM25


# notes for case study:

# One IMDB review = one Elasticsearch document

# All reviews together = imdb_reviews index

# Review text = searchable field

# Sentiment = exact filter field

# Length = numeric field for aggregation/filtering

import pandas as pd #to read csv file
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm #progress bar
 
es = Elasticsearch("http://localhost:9200")

INDEX = "imdb_reviews"

if es.indices.exists (index=INDEX):
    es.indices.delete(index=INDEX)

es.indices.create (index=INDEX, mappings= { #schema of the index
    "properties": {
        "text":      {"type": "text",    "analyzer": "english"},
        "sentiment": {"type": "keyword"},
        "length":    {"type": "integer"},
    }
})

df = pd.read_csv("data/raw/imdb_train.csv")
df["length"] = df["text"].str.len() #calculate length of each string
#df ["sentiment"] = df["label"].map({0:"negative", 1: "positive"}) #Replace each value using this dictionary

#For every row in the DataFrame, prepare one Elasticsearch document in the format helpers.bulk expects
#df.itertuples(index=False) This loops through the DataFrame rows. Each row becomes a tuple-like object. So you can access columns like: row.text row.sentiment row.length index=False means: Do not include the DataFrame index as part of the row.
#tqdm(..., total=len(df)) Wraps the loop with a progress bar. len(df) is the number of rows. 
# Parentheses here create a generator expression. That means it does not build all 25,000 actions in memory at once. It produces them one by one as helpers.bulk needs them.
actions = (
    {
    "_index": INDEX, "_id": i,
     "_source": {"text": row.text, "sentiment": row.sentiment, "length": int(row.length) }
     }
for i, row in enumerate(tqdm (df.itertuples(index=False), total=len(df)))
)

#send rquest to es
helpers.bulk (es, actions, chunk_size=500)

es.indices.refresh(index=INDEX)
print ("Docs inserted: ",es.count(index=INDEX)["count"])