(ai-trainee) PS C:\Users\ramia\ai-trainee> ollama run llama3.2:3b "Explain attention in transformers in 3 sentences."
In transformer architectures, attention is a mechanism used to weigh the importance of different elements within a sequence or vector space, 
allowing the model to focus on the most relevant parts when making predictions or generating text. This is achieved through self-attention 
mechanisms that compute weights for each pair of input tokens, enabling the model to selectively attend to certain parts and ignore others. By 
using attention, transformers can effectively capture long-range dependencies and relationships within sequences, which is particularly useful for 
tasks such as machine translation, natural language processing, and text generation.



(ai-trainee) PS C:\Users\ramia\ai-trainee> python day04/ollama_api_test.py
In Transformers, attention is a mechanism that allows the model to weigh the importance of each input element (such as words or tokens) when generating an output. It does this by computing a score for each possible target position based on similarities between input and target sequences, allowing the model to selectively focus on relevant parts of the input data. This attention mechanism is typically computed using a learnable matrix, known as the self-attention matrix, which captures complex relationships between input elements.
(ai-trainee) PS C:\Users\ramia\ai-trainee> 

(ai-trainee) PS C:\Users\ramia\ai-trainee> python day04/llamacpp_test.py
C:\Users\ramia\ai-trainee
A context window is a graphical user interface (GUI) component that displays a limited portion of the content of a larger document or application, typically used for previewing, editing, or navigating within a specific section.
(ai-trainee) PS C:\Users\ramia\ai-trainee> 


[transformers] Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
/usr/local/lib/python3.12/dist-packages/bitsandbytes/backends/cuda/ops.py:468: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
[transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer TokenizersBackend. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.
Retrieval-Augmented Generation (RAG) is a machine learning technique that combines two main tasks:

1. **Retrieval**: This involves searching a database to find relevant information (e.g., a text, image, or concept).
2. **Generation**: This involves generating new content (e.g., a text, image, or summary) based on the retrieved information.

In simple terms, RAG works by:

- Retrieving relevant information from a database.
- Using that information to generate new content, like a text or image.

This technique can be useful for tasks like text summarization, image generation, or even creative writing.