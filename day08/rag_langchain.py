from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser # converts the model output into a plain string.

#A Document usually has: # page_content, metadata
# Example:
# Document(
#     page_content="Text from page 1...",
#     metadata={"source": "data/raw/my_document.pdf", "page": 0}
# )
docs = PyPDFLoader ("data/raw/my_document.pdf").load()

#This takes the loaded documents and splits them into smaller documents. Important: The result is still a list of Document objects, not plain strings
chunks = RecursiveCharacterTextSplitter (chunk_size=800,chunk_overlap=100).split_documents (docs)

#embedding model wrapper
emb = HuggingFaceEmbeddings (model_name="BAAI/bge-small-en-v1.5")

#Create a Chroma vector database from these documents. emb is the model langchain uses to embed chunks
db = Chroma.from_documents (chunks, emb, persist_directory="day08/lc_chroma")

#A retriever is an object whose job is: Take a user question -> search the vector database -> return relevant documents/chunks
retr = db.as_retriever (search_kwargs={"k": 5})

#creates local llm object. temp=0 equals no randomness
llm =ChatOllama (model="llama3.2:3b", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """Answer only from context. If missing, say "I don't know based on the provided documents."
Context:
{context}
Question: {question}"""
)


#this function converts retrieved Document objects into plain text. Remember, LangChain retrieval returns Document objects.But the prompt needs text.
def format_docs(docs):
    texts = []

    for d in docs:
        texts.append(d.page_content)

    return "\n\n".join(texts)


#below explained
# chain.invoke(question)
#         │
#         ├── context branch:
#         │       question → retr → format_docs → context
#         │
#         └── question branch:
#                 question → RunnablePassthrough → same question

# Then:
# {context, question}
#         ↓
# prompt
#         ↓
# llm
#         ↓
# StrOutputParser
#         ↓
# final answer
#how does it wokr:
#When you later run: chain.invoke("What is the main claim of section 3?") that string: "What is the main claim of section 3?" gets passed into the chain.
chain = (
    {"context" : retr | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

print(chain.invoke("What is the difference between an image and a container?"))
