from elasticsearch import Elasticsearch, helpers #imports the main Python class that lets us connect to Elasticsearch. Elasticsearch = Python object that talks to the Elasticsearch server
es= Elasticsearch ("http://localhost:9200")
#asks elasticsearch to check if its avaialable
print (es.info())

INDEX = "movies_python"

#does the movies_python index already exist?
if es.indices.exists (index=INDEX):
    es.indices.delete(index=INDEX)

#create index
es.indices.create(
    index=INDEX,
    mappings={
        "properties": {
            "title":  {"type": "text", "analyzer": "english"}, #searhable text
            "genre":  {"type": "keyword"},#keyword
            "year":   {"type": "integer"},
            "plot":   {"type": "text", "analyzer": "english"},#searhable text
            "rating": {"type": "float"},
        }
    },
)

movies = [ #list of dicts
    {
        "title": "The Matrix",
        "genre": "SciFi",
        "year": 1999,
        "plot": "A hacker discovers reality is a simulation.",
        "rating": 8.7,
    },
    {
        "title": "Inception",
        "genre": "SciFi",
        "year": 2010,
        "plot": "A thief invades dreams to plant an idea.",
        "rating": 8.8,
    },
    {
        "title": "The Godfather",
        "genre": "Crime",
        "year": 1972,
        "plot": "A mafia family's dynastic struggle.",
        "rating": 9.2,
    },
]

for i, movie in enumerate(movies): #enumerate gives us pos + actual movie dict
    #This inserts one document into Elasticsearch
    es.index(index=INDEX, id=i, document=movie)

es.indices.refresh(index=INDEX) #This makes newly inserted documents searchable immediately.

print("Docs indexed:", es.count(index=INDEX)["count"]) #counts how many documents are in the index.

#send a search reuqest 
response = es.search(
    index=INDEX,
    query={ #Here is the search query
        "match": { #this is saying search the plot field for the words dream and hacking
            "plot": "dream hacking"
        }
    },
)

#Elasticsearch search results are inside: response["hits"]["hits"] weird
for hit in response["hits"]["hits"]:
    print(hit["_score"], hit["_source"]["title"])

#bulk indexing is key:
#### Because inserting documents one by one is slow for big data. 
# Better: Send documents in batches. That is what helpers.bulk does.

##if we want bulk indexing we use the below code insetad

# def gen (docs, index):
#     for i, d in enumerate (docs):
        #each yield item says:Put this document into this index with this ID and this source data
        #yield is like return, but it gives values one at a time
#         yield {
#             "_index":index,
#             "_id": i,
#             "_source": d
#         }

# helpers.bulk (es, gen(movies, INDEX))
# es.indices.refresh(index=INDEX)
