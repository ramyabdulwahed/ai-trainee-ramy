import re, requests, chromadb
from pypdf import PdfReader #PdfReader opens a PDF and gives us access to its pages
from sentence_transformers import SentenceTransformer


#loads embedding model that converst texts to vectors
EMB = SentenceTransformer ("BAAI/bge-small-en-v1.5")

#Persistent means it saves to disk. So Chroma will store its files inside: day08/chroma
client = chromadb.PersistentClient(path="day08/chroma")

#creates knowledge base collection
col = client.get_or_create_collection("kb")


def load_pdf(pdf: str) -> str:


#Try to make chunks around 800 characters. Carry over 100 characters between chunks.
def chunk(text: str, size: int = 800, overlap: int = 100):
