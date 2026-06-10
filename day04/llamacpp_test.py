from llama_cpp import Llama #lets python call the llama c++ engine
import os

print (os.getcwd())

# this loads the model. The object llm is now your local LLM
llm = Llama (    model_path="model/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_ctx=2048, #How much text the model can keep in mind at once
    verbose=False,)

#This asks the model to answer in chat format.
response = llm.create_chat_completion(messages = [
    {"role" : "system", "content": "You are a concise technical tutor."},  #This gives the model its behavior. Example: You are a concise technical tutor.
    {"role" : "user", "content": "In one sentence, what is a context window?"}
    ]
)

print (response ["choices"][0]["message"]["content"])