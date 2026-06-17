import re, requests, chromadb
from pypdf import PdfReader #PdfReader opens a PDF and gives us access to its pages
from sentence_transformers import SentenceTransformer


#loads embedding model that converst texts to vectors
EMB = SentenceTransformer ("BAAI/bge-small-en-v1.5")

#Persistent means it saves to disk. So Chroma will store its files inside: day08/chroma
client = chromadb.PersistentClient(path="day08/chroma")

#creates knowledge base collection
col = client.get_or_create_collection("knowledgebase")

#take a string of the file path. return a string of thr whole pdf
def load_pdf(path: str) -> str:
    reader = PdfReader (path)
    page_texts = []

    for p in reader.pages:
        #extract the text from the current page.
        text = p.extract_text()

        if text is None:
            text = ""
        page_texts.append(text)

    #combine all page texts with new lines between them
    full_text = "\n".join(page_texts)

    return full_text


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()



#The previous function did this: PDF file -> one big text string 
# Now this function does this: one big text string ->  list of smaller chunks
#Try to make chunks around 800 characters. Carry over 100 characters between chunks.
def chunk(text: str, size: int = 800, overlap: int = 100):

    #split wherever there are 2 or more newline characters. paras becomes a list of paragraphs
    paras = re.split(r"\n{2,}",text)

    chunks,buf = [], ""

    for p in paras:
        #If I add this paragraph to the current chunk, will the chunk still be under 800 characters
        if len(buf) + len(p) < size:
            buf += ("\n\n" + p) if buf else p #if buf is empty dont add new lines
        else:
            chunks.append(buf) #save current buf to final list
            buf = buf[-overlap:] + "\n\n" + p # take last 100 chars from last chunk and then add new paragraph
    if buf:
        chunks.append(buf)

    clean_chunks = []

    for c in chunks:
        cleaned = c.strip()

        if cleaned:
            clean_chunks.append(cleaned)

    return clean_chunks


#Before this function: PDF is just a file sitting on disk. 
# After this function: PDF chunks are inside Chroma with embeddings. 
# Now we can ask questions and retrieve relevant chunks.
def index_pdf(path: str):
    #text = load_pdf (path)
    text = load_text("data/raw/my_document.txt")
    chunks = chunk (text)
    #embedd all chunks to vectors
    embs = EMB.encode (chunks, normalize_embeddings=True).tolist() #.tolist() converts: NumPy array → Python list

    ids = [f"{path}:: {i}" for i in range(len(chunks))] #this will reutrn a list of paths for each chunk

    #col is the chroma collection we created.
    #the four arguments line up
    col.add (ids=ids, documents=chunks, embeddings=embs,
             metadatas= [{"source": path, "chunk": i } for i in range(len(chunks))])


#Before this function, we already indexed the PDF: PDF → chunks → embeddings → Chroma 
# Now the user asks a question: What is the main claim of section 3?
#k: int = 5 This means: Return the top 5 most relevant chunks

#return value: It returns a list of retrieved hits. Each hit is a tuple: (document_text, metadata, distance)
def retrieve(query: str, k: int = 5):
    #embeds the user question
    q = EMB.encode ([query], normalize_embeddings=True).tolist() #so after q looks like this: [ [0.12, -0.44, 0.09, ...] ]
    #take this query embedding. Compare it against all stored chunk embeddings. Return the top k closest chunks
    r = col.query (query_embeddings=q, n_results=k) #r will contain a dict of documents, distances and metadatas and there values will be nested

    #zip(...) by itself gives a zip object. To actually see/use it as a normal list so we wrap it with list
    return list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))


#prompt below:
# This is not a function. This is a template string.
# Its job is: Tell the LLM how to answer.
# Remember:
# retrieve() finds the chunks.
# PROMPT tells the LLM how to use those chunks.
# The LLM does not automatically know what to do with retrieved chunks. We have to tell it.


####Most important idea The LLM only sees what we put in the prompt. 
# It does not automatically see Chroma. It does not automatically see your PDF. 
# It only sees: instructions + retrieved chunks + question That is why this PROMPT is the bridge between retrieval and generation.

PROMPT = """You are a precise assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the provided documents."
Cite sources as [source_name chunk_id] inline.

Context:
{context}

Question: {question}

Answer:"""


# This function connects everything. It does the full question-answering pipeline:
# This is the function that finally calls the LLM.
def ask(question: str):
    hits = retrieve(question, k=5) #take the question, search chroma, return top 5 chunks, hits will be a list of tuples

    #context is the retrieved chunks formatted as text.
    formatted_chunks = []

    for doc, m, _ in hits:
        formatted = f"[{m['source']} chunk {m['chunk']}]\n{doc}"
        formatted_chunks.append(formatted)
    
    context = "\n\n---\n\n".join(formatted_chunks)
    #fills the prompt template with actual values
    prompt = PROMPT.format (context=context, question=question)

    #sends data to a ollama server
    #r is the HTTP response from Ollama. It contains JSON data. Example: { "model": "llama3.2:3b", "response": "The main claim of section 3 is that RAG improves factual grounding [data/raw/my_document.pdf chunk 3].", "done": true }
    r = requests.post ("http://localhost:11434/api/generate", 
                       json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}) #stream=false:Return the full answer at once
    return r.json()["response"]



if __name__ == "__main__":
    index_pdf("data/raw/my_document.pdf")
    print(ask("What is the main claim of section 3?"))