import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer ("all-MiniLM-L6-v2")

corpus =  [
    "A man is eating pasta.",
    "Someone is enjoying a bowl of spaghetti.",
    "The stock market dropped today.",
    "A woman plays the violin on stage.",
    "Plants convert sunlight into energy through photosynthesis.",
    "The company reported strong profits this quarter.",
]

#turns each corpus sentence into a vector
emb = model.encode (
    corpus,
    batch_size=64,
    show_progress_bar=True, #encode up to 64 texts at a time
    normalize_embeddings=True
)

#FAISS expects vectors as NumPy arrays with type: float32
emb = np.array (emb, dtype="float32")

#this creates the FAISS index.
index =faiss.IndexFlatIP(emb.shape[1]) #emb.shape[1] means: number of columns For MiniLM: 384

#stores all corpus embedding sinside faiss index
index.add (emb)
print("Vectors in index:", index.ntotal)

#embed query
query = model.encode(
    ["how do plants convert sunlight into energy?"],
    normalize_embeddings=True
).astype("float32")
#convert to np for faiss
#query = np.array(query, dtype="float32")

# Search the FAISS index using the query vector, Return top 3 closest vectors.
# k=3 means: give me 3 results FAISS returns two things: 
# D = distances/scores.  D contains inner product scores. Since vectors are normalized: D ≈ cosine similarity scores
# I = indexes/IDs For IndexFlatIP, I contains the positions in the corpus. Example: I = [[4, 1, 0]] Means: best match is corpus[4] second best is corpus[1] third best is corpus[0]
D, I = index.search(query, k=5)

for rank, (score, idx) in enumerate(zip(D[0], I[0])):
    print(f"{rank+1}. score={score:.3f} | {corpus[idx]}")

