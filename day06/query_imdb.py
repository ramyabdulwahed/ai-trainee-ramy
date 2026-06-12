from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
INDEX = "imdb_reviews"


r = es.search(
    index=INDEX,
    query={
        "bool": { #bool lets us combine multiple conditions
            "must": [ #This condition must match and affects score.
                {"match": {"text": "best cinematography"}} #Search the review text for the terms best and cinematography.
            ],
            "filter": [ #Hard condition. Keep only documents that satisfy this. Do not use it to calculate score. term means exact match.
                {"term": {"sentiment": "positive"}}
            ]
        }
    },
    size=5
)

for h in r["hits"]["hits"]:
    print("Score:", h["_score"])
    print(h["_source"]["sentiment"], h["_source"]["length"])
    print(h["_source"]["text"][:300])
    print("-" * 80)




#query 2 
#Group reviews by sentiment. For each sentiment, calculate statistics about review length.

r = es.search(
    index=INDEX,
    size=0,
    aggs={
        "length_stats_by_sentiment": { #This is  our chosen name for the aggregation
            "terms": {
                "field": "sentiment" #This groups documents by sentiment. Because sentiment is keyword, this works well
            },
            "aggs": { #Inside each sentiment bucket, calculate stats for the length field. stats gives: count min max avg sum
                "length_stats": {
                    "stats": {
                        "field": "length"
                    }
                }
            }
        }
    }
)

for bucket in r["aggregations"]["length_stats_by_sentiment"]["buckets"]:
    print("Sentiment:", bucket["key"])
    print("Count:", bucket["doc_count"])
    print("Length stats:", bucket["length_stats"])
    print("-" * 80)



#Query 3 — multi_match query that boosts "brilliant" 2x
#multi_match means: Search the same query across one or more fields. 
# Example if we had title and plot: "fields": ["title^2", "plot"] This would mean: Search both title and plot, but title is twice as important.
r = es.search(
    index=INDEX,
    query={
        "multi_match": {
            "query": "brilliant acting",
            "fields": ["text^2"]
        }
    },
    size=5
)

for h in r["hits"]["hits"]:
    print("Score:", h["_score"])
    print(h["_source"]["sentiment"], h["_source"]["length"])
    print(h["_source"]["text"][:300])
    print("-" * 80)


# Query 4 — Phrase query: exact phrase "waste of time"
#"match_phrase": { "text": "waste of time" } This means: Find reviews where this phrase appears in this order. 
# This is different from match. A normal match query: {"match": {"text": "waste of time"}} means: Find reviews related to the words waste and time. 
# But match_phrase means: Find the phrase waste of time together. So phrase query is stricter.
r = es.search(
    index=INDEX,
    query={
        "match_phrase": {
            "text": "waste of time"
        }
    },
    size=5
)

for h in r["hits"]["hits"]:
    print("Score:", h["_score"])
    print(h["_source"]["sentiment"], h["_source"]["length"])
    print(h["_source"]["text"][:300])
    print("-" * 80)