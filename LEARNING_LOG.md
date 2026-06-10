
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