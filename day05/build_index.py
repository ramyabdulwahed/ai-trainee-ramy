# We are building a tiny semantic search engine.
#mental model

# Wikipedia dataset	gives us text passages
# SentenceTransformer	converts passages into embeddings
# FAISS	stores/searches embeddings
# corpus.pkl	stores original text passages
# wiki.faiss	stores FAISS vector index

# Notes: 

# FAISS does not create embeddings.
# SentenceTransformer creates embeddings.
# FAISS stores/searches embeddings.

import os
import pickle #to save/load python obkjects. ex.corpus
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

print ("loading dataset")

#load_dataset loads ds from hugging face. 
ds = load_dataset ("rag-datasets/rag-mini-wikipedia", #ds name
                    "text-corpus")["passages"] #passages is the split in the ds

print (ds)
