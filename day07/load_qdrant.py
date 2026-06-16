# With Qdrant, it becomes:

# Python script
#   ↓ HTTP request
# Qdrant server running in Docker
#   ↓
# persistent vector database storage

from qdrant_client import QdrantClient #how python talks to qdrant
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer #imports the embedding model library.
import pandas as pd


model = SentenceTransformer ("all-MiniLM-L6-v2") 
qc = QdrantClient("http://localhost:6333") #connects Python to the Qdrant server running in Docker

df = pd.read_csv("data/raw/imdb_train.csv")

df = ( #parentheses allow us to split one long operation across multiple lines
    df.groupby ("label", group_keys=False) #group dataset by label
    .apply(lambda x: x.sample(2500, random_state=42)) #For each label group, run this function. So it runs once on the negative group and once on the positive group
    .reset_index (drop=True)
)

docs = df["text"].tolist() #.tolist() converts that column into a normal Python list

sentiments = [
    "positive" if l == 1 else "negative"
    for l in df["label"]
]

qc.recreate_collection(
    collection_name="imdb", #A Qdrant collection is like a vector table/index
    vectors_config=qm.VectorParams( #The vectors in this collection will look like this
        size=384,
        distance=qm.Distance.COSINE #compare vectors using cosine similarity/distance
    ),
)

#This takes the list of review texts and converts each review into an embedding
vectors = model.encode (
    docs,
    normalize_embeddings=True
).tolist()

#insert to qdrant
qc.upsert(
    collection_name="imdb",
    points=[
        qm.PointStruct(
            id=i,
            vector=v,
            payload={"text": d, "sentiment": s}
        )
        for i, (d, v, s) in enumerate(zip(docs, vectors, sentiments))
    ],
)

q = model.encode(
    ["a film with breathtaking cinematography"],
    normalize_embeddings=True
)[0].tolist() #Give me the embedding for the first query as we might have multiple thats why its a list first arg

#Search for reviews semantically close to: "a film with breathtaking cinematography" But only where: sentiment = positive Return top 5.
hits = qc.search(
    collection_name="imdb",
    query_vector=q,
    limit=5,
    query_filter=qm.Filter(
        must=[
            qm.FieldCondition( #payload["sentiment"] must equal "positive"
                key="sentiment",
                match=qm.MatchValue(value="positive")
            )
        ]
    ),
)

for h in hits:
    print(h.score, h.payload["text"][:120])