from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

es = Elasticsearch("http://localhost:9200")

INDEX = "imdb_hybrid"
model = SentenceTransformer("all-MiniLM-L6-v2")


def search_bm25(query, k=10):
    r = es.search(
        index=INDEX,
        size=k,
        query={
            "match": {
                "text": query
            }
        },
    )

    return [hit["_id"] for hit in r["hits"]["hits"]]


def search_semantic(query, k=10):
    qvec = model.encode(query).tolist()

    r = es.search(
        index=INDEX,
        size=k,
        knn={
            "field": "embedding",
            "query_vector": qvec,
            "k": k,
            "num_candidates": 100,
        },
    )

    return [hit["_id"] for hit in r["hits"]["hits"]]


def rrf(result_lists, k=60, top_n=10):
    scores = {}

    for ranks in result_lists:
        for rank, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    return [doc_id for doc_id, score in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]


def search_hybrid(query, k=10):
    bm25_ids = search_bm25(query, k=20)
    semantic_ids = search_semantic(query, k=20)

    return rrf([bm25_ids, semantic_ids], top_n=k)


#Build 5 test questions and remember the correct document ID for each one
def get_eval_queries():
    r = es.search(
        index=INDEX,
        size=5, #5 documents/reviews
        query={"match_all": {}}, #Give me any documents from the index
        _source=["text"],
    )

    eval_queries = []

    for hit in r["hits"]["hits"]:
        #for each review we get the doc id and text
        doc_id = hit["_id"]
        text = hit["_source"]["text"]

        #takes the first 10 words of that review
        query = " ".join(text.split()[:10])
        #we then store the query so we later use it
        eval_queries.append({
            "query": query,
            "target_id": doc_id,
        })
    #So get_eval_queries() returns something like: [ {"query": "A good picture is worth all the words", "target_id": "1746"}, {"query": "This movie was one of the funniest films", "target_id": "22"}, {"query": "The acting was terrible and the story was", "target_id": "913"}, ]
    return eval_queries


#Recall@10 tells us how many times the correct answer appeared somewhere in the top 10.
def recall_at_10(search_fn, eval_queries):
    found = 0

    for item in eval_queries:
        query = item["query"]
        target_id = item["target_id"]

        results = search_fn(query, k=10)

        if target_id in results:
            found += 1

    return found / len(eval_queries)


def mrr(search_fn, eval_queries):
    total = 0

    for item in eval_queries:
        query = item["query"]
        target_id = item["target_id"]

        results = search_fn(query, k=10)

        if target_id in results:
            rank = results.index(target_id) + 1
            total += 1 / rank

    return total / len(eval_queries)


eval_queries = get_eval_queries()

methods = {
    "BM25": search_bm25,
    "Semantic": search_semantic,
    "Hybrid": search_hybrid,
}

print("\nQuick Hybrid Search Bake-off\n")

for name, fn in methods.items():
    recall = recall_at_10(fn, eval_queries)
    score_mrr = mrr(fn, eval_queries)

    print(f"{name:10s} Recall@10: {recall:.2f}   MRR: {score_mrr:.2f}")