
## Day 1 - Environment Setup

### Self-Reflection Questions

#### 1. What is the difference between `pip` and `mamba`? When do you prefer one over the other?

`pip` is mainly used to install Python packages from PyPI, while `mamba` is used to manage Conda environments and install packages from Conda channels. I understand `mamba` as a faster version of Conda that is useful when setting up a full environment, especially for data science packages that may have more complex dependencies.

I would use `mamba` to create the main environment and install heavier scientific packages. I would use `pip` when a package is mainly a normal Python package or when the instructions specifically give a `pip install` command.

#### 2. Why do we pin versions in `requirements.txt`?

We pin versions so the project environment is repeatable. This helps avoid random bugs where the code works on one machine but breaks on another because a library version changed.

#### 3. What would `torch.cuda.is_available()` return inside a Docker container without `--gpus all`?

It would return `False` because the Docker container would not have access to the GPU. Even if the physical machine has an NVIDIA GPU, the container needs permission to see it. The `--gpus all` flag is what tells Docker to expose the GPU to the container. Without that, PyTorch inside the container would act like no CUDA GPU is available.


## Day 2


#### knowledge check

The NLTK and spaCy outputs differ because the NLTK function explicitly removes emails, HTML, punctuation, numbers, and stopwords before lemmatizing, while the spaCy function relies more on spaCy token attributes such as is_alpha, is_stop, and lemma_. spaCy lemmatizes ran to run better because it uses part-of-speech information, while the basic NLTK lemmatizer leaves ran unchanged. The NLTK pipeline kept mile because it removed the HTML tags first and then lemmatized miles, while the spaCy version did not explicitly clean HTML before tokenization.

#### TF-IDF
TF-IDF means: Give higher weight to words that are common in this document but not common everywhere. Good words for classification are usually words that help separate classes.


#### pipeline
The Pipeline pattern guarantees the same transformations are applied at train and inference time. Means: The way you convert text to numbers during training must be the same way you convert text to numbers during prediction. Why? Because the model learns based on a specific feature layout.

dataleakage: The test set should be unseen.

### Step 2.5 — Databases Primer

#### storage types 

1. flat files: used for datasets, exports. ex. csv, json. 
2. relational db: used for structured data
3. search engine ex. elasticsearch
4. vector db: search by meaning not exact words


### Self-Reflection Questions

#### Why is stemming rarely the right choice for English NLP compared to lemmatization?

Stemming cuts words using rough rules and can produce incorrect or non-real word forms. Lemmatization is usually better for English NLP because it converts words to real dictionary base forms, which preserves meaning more clearly. 

#### In what scenario would you keep stopwords during preprocessing?

I would keep stopwords in sentiment analysis because words like "not", "never" and "no" can completely change the meaning of a sentence, such as "not good" becoming "good" if stopwords are removed.

#### Explain in one sentence why TF-IDF’s IDF term is useful.

The IDF term is useful because it lowers the weight of words that appear in many documents and gives more importance to rarer words that better distinguish one document from another.


## Day 3


### Self-Reflection Questions

#### why does fp16=True help on the RTX 3060 but could hurt accuracy on older GPUs?
fp16=True makes training use 16-bit floating point numbers instead of the usual 32-bit numbers. This helps on newer GPUs like the RTX 3060 because they have hardware support for fast 16-bit math, which can reduce memory usage and speed up training. However, fp16 has less numerical precision than fp32. On older GPUs, such as Pascal-generation cards, fp16 may not be accelerated as effectively and can be less numerically stable, which may hurt training quality or accuracy.

#### What is an attention mask and what happens if you pass a random one during inference?

The attention mask protects the model from treating padding as real text. A random mask corrupts the input signal, so the model may ignore important words or attend to fake padding, producing wrong or unstable predictions.

#### Distinguish: pretraining, fine-tuning, instruction tuning, RLHF

Pretraining: The model learns general language patterns from a huge amount of text before being used for a specific task.
Fine-tuning: A pretrained model is trained further on a specific dataset or task, such as IMDB sentiment classification.
Instruction tuning: A model is trained on examples of instructions and good responses so it learns to follow user requests better.
RLHF: Reinforcement Learning from Human Feedback uses human preference judgments to make the model’s responses more helpful, safe, or aligned with what people prefer.


## day4

### Reflection questions explained

#### Why does temperature=0 still sometimes produce different outputs across hardware?
In theory: temperature=0 means always pick the most likely next token. So you expect deterministic output. But in practice, tiny differences can still happen because: different GPUs/CPUs may perform floating-point math slightly differently, libraries may use different optimized kernels, parallel computation can change tiny rounding behavior, tie-breaking between very close token scores can differ So the model may have two tokens with extremely close scores. Example: Token A score: 10.0000001 Token B score: 10.0000000 A tiny rounding difference can flip which one wins. Once one token differs, the rest of the generation can drift.

#### What is the KV-cache and why does it grow with conversation length?
A transformer uses attention. When generating text, it needs to look back at previous tokens. Without caching, every new token would require recalculating attention information for all previous tokens again and again. The KV-cache stores previous attention information. KV means: K = keys V = values. Thus KV-cache = saved memory of previous token computations used during generation. Why does it grow? Because longer conversation means more tokens. More tokens means the model has more previous context to attend to. So the cache stores more key/value vectors. short prompt = small KV-cache long conversation = large KV-cache. This is why long chats use more VRAM/RAM.

The KV-cache stores the transformer’s already-computed attention information for previous tokens, so when the model generates the next token, it can reuse previous context instead of recalculating everything from scratch; as the chat gets longer, this cache grows, which can make generation slower and use more memory.

#### When would you choose Ollama over llama.cpp-python, and vice versa?
Choose Ollama when we want: quick setup, easy local model running, simple CLI, REST API, fast prototyping, local chatbot apps, less manual model management. Choose llama.cpp-python when we want: more control over GGUF files, exact quantization choice, custom Python integration, manual CPU/GPU offloading, more low-level experimentation


#### What is Q4_K_M vs fp16? 
Q4_K_M and fp16 are different precision/storage choices. fp16 stores weights using 16-bit floating-point numbers, which gives better precision but uses more memory. Q4_K_M stores weights in a 4-bit quantized GGUF format, which greatly reduces memory usage with a small quality tradeoff.

## day5

### Reflection questions

#### If you embed "bank" (the river kind) and "bank" (the money kind) with a bi-encoder, why is the embedding the same? What class of models solves this?
    If I embed the word "bank" by itself, the embedding is the same for the river meaning and the money meaning because the model receives the exact same input text: bank. There is no surrounding context telling it whether I mean a financial bank or the side of a river. In our FAISS setup, we use a bi-encoder: the query and documents are encoded separately into vectors, then compared. This is fast because document embeddings can be precomputed, but it can struggle with ambiguous words when there is little context. A cross-encoder handles this better because it reads the query and candidate passage together and produces a relevance score based on their combined meaning.


#### 
IndexFlatIP is exact search, so it compares the query vector against every stored vector. At a small scale like 1,000 vectors, this is very fast because there is not much to scan and there is little overhead. At a huge scale like 10 million vectors, scanning every vector becomes expensive and slow. IndexHNSWFlat uses an approximate graph-based search structure, so it can skip many vectors and find good matches faster at large scale. However, at small scale, that extra structure can be unnecessary overhead, so IndexFlatIP can be faster.

#### 
If I do not L2-normalize before using inner-product search, the ranking can be affected by vector magnitude instead of just semantic direction. A document vector with a larger length may get a higher dot-product score even if it is not the most relevant result. This can cause FAISS to return results that win because their vectors are bigger, not because their meanings are closer to the query. L2-normalization fixes this by making every vector length 1, so inner product behaves like cosine similarity.


## day6

### Reflection questions

#### What happens if you `match` a single lower-case word against a `keyword` field?

A keyword field is stored as one exact value. It is not analyzed like a text field. If the lowercase query exactly matches the stored keyword value, it can match. For example, matching "positive" against a sentiment keyword field works if the stored value is exactly "positive". However, keyword fields are not full-text searchable. A partial word or different casing may not match. For example, searching "scifi" may not match a stored keyword value "SciFi" unless a lowercase normalizer was configured. So the main idea is: keyword fields are for exact matching, filtering, and aggregations, not flexible full-text search.

#### Name two analyzer differences between `english` and `standard`.

The standard analyzer is a general-purpose analyzer. It tokenizes text and lowercases terms, but it does not do English-specific stemming. The english analyzer is designed for English text. Two important differences are: The english analyzer removes common English stopwords, such as "the", "and", or "is". The english analyzer performs stemming, so related forms like "movies" and "movie" can be reduced closer together.

#### In BM25, why does `idf` make rare words more influential?

idf stands for inverse document frequency. It makes rare words more influential because rare words are more informative. A common word appears in many documents, so matching it does not strongly prove that a document is relevant. For example, in IMDB reviews, the word "movie" appears very often, so it is not very specific. But a word like "cinematography" is rarer, so if a query contains "cinematography" and a review also contains it, that is stronger evidence of relevance. Therefore, BM25 gives rare terms more weight because they help narrow down the search results better than common words.

#### BM25 vs semnantic search
Day 5 semantic search uses embeddings, so it searches by meaning. Day 6 BM25 search uses Elasticsearch, so it searches mainly by exact words, rare terms, term frequency, and document length normalization. In general: BM25 wins when the query has exact keywords or exact phrases. Semantic search wins when the query is a paraphrase or describes an idea using different words.


## day7

### Reflection questions

#### In RRF, why does the choice of `k` matter less than one might expect?
k matters less because RRF mainly depends on rank order and overlap between result lists. With a typical k = 60, the score difference between nearby ranks is small, so changing k usually does not dramatically change the final ranking. The most important thing is whether a document appears high in multiple rankers.

#### Why might you choose Qdrant over ChromaDB for a 10M-document corpus?
I might choose Qdrant over ChromaDB for a 10M-document corpus because Qdrant is more suitable for production-scale vector search. ChromaDB is excellent for local experiments and quick prototypes, but Qdrant is designed with stronger support for large indexes, efficient approximate nearest-neighbor search, metadata filtering, persistence, deployment, and scaling. At 10 million documents, performance, reliability, and index tuning become much more important.

#### When does kNN fail completely that BM25 handles fine (and vice versa)?
kNN can fail when the search depends on exact keywords, names, titles, IDs, or rare terms. For example, a query like "Citizen Kane" or an error code may be handled better by BM25 because BM25 directly matches exact terms. BM25 can fail when the user query is a paraphrase or describes meaning without using the same words as the document. For example, "rich man dies whispering a name" may not share exact words with a review about Citizen Kane, but semantic search can still find it because kNN searches by meaning. This is why hybrid search is useful: BM25 handles exact matching, while kNN handles semantic similarity.


## day9

#### Why do we double quantize in NF4?

In QLoRA, NF4 compresses the base model weights into 4-bit values to save memory. But quantization also requires extra metadata/scales to reconstruct the values during computation. Double quantization means we also quantize those scale/metadata values. It does not mean quantizing the whole model twice. The purpose is to reduce memory overhead and make QLoRA even more GPU-memory efficient without significantly hurting quality.


#### Why does packing=True matter so much for short-sample instruction datasets?

 Instruction datasets often contain short examples, but training uses a fixed max sequence length like 1024 or 2048 tokens. Without packing, each short example may waste most of the sequence on padding. packing=True combines multiple short examples into one full sequence, so the GPU trains on more real tokens and less padding. This makes training faster and more memory-efficient.


#### What would break if lora_alpha is much larger than 2 × r?

In LoRA, r controls the adapter rank/capacity, while lora_alpha controls the strength of the adapter update. The LoRA scaling is roughly lora_alpha / r. If lora_alpha is much larger than 2 × r, the adapter update becomes too strong and can overpower the base model. This can cause unstable training, worse generations, overfitting, repetition, or loss of the model’s original behavior. The code may still run, but model quality can break.