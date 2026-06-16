from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import pandas as pd

es = Elasticsearch("http://localhost:9200") #connects to the Elasticsearch server running locally

model = SentenceTransformer("all-MiniLM-L6-v2")


#load data
df = pd.read_csv("data/raw/imdb_train.csv")

df = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(10, random_state=42))
      .reset_index(drop=True)
)


docs = df["text"].tolist()
sentiments = [
    "positive" if l == 1 else "negative"
    for l in df["label"]
]

vectors = model.encode(
    docs,
    normalize_embeddings=True
).tolist()

#elastic search stuff

INDEX = "imdb_hybrid"

if es.indices.exists(index=INDEX):
    es.indices.delete(index=INDEX)


#create mapping
es.indices.create (index=INDEX, mappings={
    "properties": {
        "text" : {"type": "text", "analyzer": "english"},
        "sentiment": {"type": "keyword"},
        "embedding" :{"type": "dense_vector", #dense_vector: means this field stores a vector of numbers.
                      "dims": 384, 
                      "index": True, #index=true is importnat as it tells Elasticsearch: Build a vector index so I can run kNN search efficiently
                      "similarity": "cosine"},  

    }
})


def gen():
    for i, (text, sent, vec) in enumerate(zip(docs, sentiments, vectors)):
        yield {
            "_index": INDEX,
            "_id": i,
            "_source": {
                "text": text,
                "sentiment": sent,
                "embedding": vec
            }
        }


print("Indexing documents...")
helpers.bulk(es, gen(), chunk_size=500)

print("Refreshing index...")
es.indices.refresh(index=INDEX)

q_vec = model.encode(
    ["a film with breathtaking cinematography"],
    normalize_embeddings=True
)[0].tolist()

print ("\nPure knn results:")
r = es.search (index=INDEX, 
               knn={ #run vector nearest-neighbor search.
                   "field": "embedding",
                   "query_vector": q_vec,
                   "k" : 10,
                   "num_candidates": 100, #controls how many candidate vectors ES considers before choosing the final top k
               },
               size=5            
)


# r
#  └── hits
#       └── hits
#            └── list of documents
for hit in r["hits"]["hits"]:
    print(hit["_score"], hit["_source"]["text"][:120])


print("\nHybrid BM25 + kNN results:")
r = es.search(
    index=INDEX,
    query={"match": {"text": "breathtaking cinematography"}}, #normal Elasticsearch query
    knn={ #Find documents whose embedding is close to q_vec
        "field": "embedding",
        "query_vector": q_vec,
        "k": 10,
        "num_candidates": 100,
    },
    rank={
        "rrf": {"window_size": 50,"rank_constant": 60}}, #BM25 scores and vector scores are on different scales. So instead, RRF looks at rankings. RRF rewards documents that appear high in either/both lists
    size=10,
)

for hit in r["hits"]["hits"]:
    print(hit["_score"], hit["_source"]["text"][:120])


# Here's the manual version in case your stack doesn't support RRF natively:
#Example: result_lists = [ ["docA", "docB", "docC"],  # BM25 results ["docC", "docA", "docD"],  # kNN results ]
def rrf(result_lists, k=60, top_n=10):
    """result_lists: list of lists of doc_ids, each list in rank order"""
    scores = {}
    for ranks in result_lists:
        #loops through the documents inside one ranked list
        for rank, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_n]