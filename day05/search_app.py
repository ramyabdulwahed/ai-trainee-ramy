import pickle

import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("day05/wiki.faiss")

with open ("day05/corpus.pkl", "rb") as f:
    corpus = pickle.load (f)

def search(query, k=5):
    q = model.encode(
        [query],
        normalize_embeddings=True,
    ).astype("float32")

    D, I = index.search(q, k)

    results = []

#Why D[0] and I[0]? Because FAISS supports searching multiple queries at once. 
# Example: q = model.encode([ "what are fossil fuels?", "what is nuclear power?" ]) That would be 2 queries. 
# Then FAISS might return: D = [ [0.48, 0.41, 0.39],   # scores for query 1 [0.52, 0.44, 0.37],   # scores for query 2 ] I = [ [15, 200, 42],# passage indexes for query 1 [91, 8, 300],         # passage indexes for query 2 ] So: D[0] means: scores for the first query and: I[0] means: indexes for the first query    for score, idx in zip(D[0], I[0]):
        passage = corpus[idx][:400]
        results.append(f"**Score {score:.3f}**\n\n{passage}...")

        return "\n\n---\n\n".join(results)

#Gradio is a Python library that lets you turn a Python function into a simple web app like wrapper
gr.Interface(
    fn=search, #When user clicks submit, call the search function.
    inputs =[ #creates 2 ui components
        gr.Textbox(label="query"),
        gr.Slider(1, 20, value=5, step=1, label="Top-K"),
    ],
    outputs = [gr.Markdown (label="Results")],
    title="Day 5 Semantic Search",

).launch(share=True)