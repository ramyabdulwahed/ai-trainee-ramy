# The pipeline in one picture
# data/raw files
#    ↓
# SimpleDirectoryReader
#    ↓
# LlamaIndex Document objects
#    ↓
# VectorStoreIndex.from_documents(docs)
#    ↓
# chunks/nodes
#    ↓
# BAAI/bge-small-en-v1.5 embeddings
#    ↓
# vector index
#    ↓
# query engine
#    ↓
# retrieve top 5 chunks
#    ↓
# Ollama llama3.2:3b
#    ↓
# final answer


#Settings is where we set global defaults
##VectorStoreIndex is the main RAG index. Mental model: VectorStoreIndex = documents/chunks + embeddings + retrieval ability
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

#Configure embedding model and LLM

#This tells LlamaIndex: Whenever you need embeddings, use BAAI/bge-small-en-v1.5
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

#Whenever you need answer generation, use llama3.2:3b through Ollama
Settings.llm = Ollama (model="llama3.2:3b", request_timeout=120)

#load our data
docs = SimpleDirectoryReader ("data/raw").load_data() #docs is a list of LlamaIndex Document objects.

#this line below is important
# LlamaIndex does several things behind the scenes:
# documents
# → split into chunks/nodes
# → embed each chunk
# → store the chunk vectors
# → create a retrievable index
index = VectorStoreIndex.from_documents (docs)

#create query engine it hanldes everything below
# it handles the whole flow:

# 1. Embed the question
# 2. Retrieve top 5 similar chunks
# 3. Build a prompt with those chunks
# 4. Send prompt to llama3.2:3b
# 5. Return the final answer
qe = index.as_query_engine (similarity_top_k=5)

print (qe.query("What is the main claim of section 3?"))
