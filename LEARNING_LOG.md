
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