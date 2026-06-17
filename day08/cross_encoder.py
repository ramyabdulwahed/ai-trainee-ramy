from sentence_transformers import CrossEncoder 
#this imports the cross-encoder class. before, we used SentenceTransformer for embeddings. That was bi-encoder style: model.encode(text)
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Inputs:

# query = user's question
# candidates = chunks retrieved by the bi-encoder
# top_k = how many final chunks we want
def rerank(query, candidates, top_k=5):
    pairs  = [(query, c) for c in candidates] #create a tupel with the question query and each chunk.Because the cross-encoder needs to read both together.
    #This gives a relevance score for each pair.
    scores = ce.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return ranked[:top_k]