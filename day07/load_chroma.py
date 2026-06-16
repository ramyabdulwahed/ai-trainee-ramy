#ChromaDB auto-handles embedding, persistence, metadata filters, and collections. 
# For pure-local work it's the fastest way to stand up a vector DB.

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

#client =connection to ChromaDB
#chromadb.PersistentClient(...) means create a Chroma client that saves data to disk. 
# Persistent = it does not disappear when the script ends. 
# path="day07/chroma_store" means Chroma will store its database files inside: day07/chroma_store/
client = chromadb.PersistentClient(path="day07/chroma_store")

ef = embedding_functions.SentenceTransformerEmbeddingFunction (
    model_name = "all-MiniLM-L6-v2"
)

#If the collection exists, open it. If it does not exist, create it. "imdb" is the collection name. embedding_function=ef connects our embedding model to this collection.
col = client.get_or_create_collection ("imdb", embedding_function=ef)

#load data
df = pd.read_csv ("data/raw/imdb_train.csv").head(100)

#inserts multiple documents at once.
col.add (
    ids=[f"doc{i}" for i in range(len(df))],
    documents=df["text"].tolist(), #turns pd dataframe text column to a list.
    metadatas=[{"sentiment": "positive" if l == 1 else "negative"} for l in df["label"]]
)

#search collection
res = col.query (
    query_texts=["a film with breathtaking cinematography"], #It is a list because Chroma allows multiple queries at once
    n_results=5,
    where={"sentiment": "positive"},
)
#[0] since chroma supports multiple queries and in our case one so index 0
for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(f"{dist:.3f}  {doc[:120]}")