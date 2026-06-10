# This case study teaches the app architecture of a local LLM chatbot behind many apps:

# Terminal UI
#    ↓
# Python loop
#    ↓
# Conversation history list
#    ↓
# Ollama REST API
#    ↓
# Local LLM
#    ↓
# Streaming chunks
#    ↓
# Terminal output


import requests #to talk to ollamas api
import json #to parse text
import sys #to print streaming text without waiting for a full line


HOST = "http://localhost:11434" #ollama local server

#This asks Ollama: What models are downloaded on this machine? The endpoint: /api/tags returns downloaded models
def list_models():
    response = requests.get(f"{HOST}/api/tags")
    response.raise_for_status()

    data = response.json()
    return [model["name"] for model in data.get ("models", [])] #data is a dict, .get gets the values in the key "model" and if it doesnt eixst return []

#It takes: model = which Ollama model to use, messages = conversation history
def chat_stream (model, messages):
    with requests.post (f"{HOST}/api/chat", 
                             json= {
                                 "model": model,
                                 "messages": messages,
                                 "stream": True, #ask ollama to send answers in chunks ex. token by token
                             },
                             stream=True) as response: #Tell Python requests not to wait for the full response before giving us data
        
        response.raise_for_status() #if Ollama returns an error, stop and show the error
        full_reply = ""

        #This loops through each line Ollama sends back, Because streaming responses often arrive as many small JSON lines.
        #ex. {"message":{"content":"Quant"},"done":false}
        for line in response.iter_lines(): 
            #ignore empty lines
            if not line:
                continue
            #convert into python dict
            chunk = json.loads (line)
            #suppose chunk = { "message": { "content": "Quant" }, "done": False } Then: chunk.get("message", {}) returns: {"content": "Quant"} Then: .get("content", "") returns: "Quant"
            piece = chunk.get("message", {}).get("content", "")


            #This is what creates the streaming effect. 
            # Normal print() waits for newline behavior, sys.stdout.write() prints exactly the piece. 
            # flush() forces it to show immediately
            sys.stdout.write (piece)
            sys.stdout.flush()
            full_reply += piece #save peice

            #This checks whether Ollama says the answer is finished
            if chunk.get("done"):
                print()
                return full_reply
def main():
    print ("Available models: ", list_models())
    model = input("Model (e.g., llama3.2:3b): ").strip() or "llama3.2:3b" #or "llama3.2:3b" This is a shortcut for default value. If the user types nothing and just presses Enter, then:
    system = input ().strip("System prompt (Enter for default): ") or "You are a helpful, concise assistant."
    history = [{"role": "system", "content": system}]

    print (f"\n --- chatting with {model} Commands: /clear /exit /swap <model> ---")
    while True: #infinite loop
        try:
            user = input ("\nYou: ") #ask model for somethign
        except (EOFError, KeyboardInterrupt): #If the user interrupts the input, exit the chat loop
            break

        if user == "/exit":
            break
        #reset program history
        if user == "/cear": 
            history =  [{"role": "system", "content": system}] 
            continue
        if user.startswith("/swap "):
            model = user.splt (" ", 1)[1] #split once only when you see a space and then give me the 2nd item. ex.["/swap", "phi3.5:3.8b"] The 1 means: Only split once. Then: user.split(" ", 1)[1] gets the second item: "phi3.5:3.8b"
            print(f"Swapped to {model}."); continue
        
        #ex for history after 1 input
        #     history = [
        # {"role": "system", "content": "You are helpful."},
        # {"role": "user", "content": "Explain quantization in one sentence."}]
        history.append({"role": "user", "content": user})
        print("Assistant: ", end="")
        reply = chat_stream(model, history)


        #after the above function call we append again because:
        #History looks like this
        # history = [
        #     {"role": "system", "content": "You are helpful."},
        #     {"role": "user", "content": "Explain quantization."},
        #     {"role": "assistant", "content": "Quantization means storing model weights with fewer bits to save memory."}
        # ]

        # Why important?
        # Because next time you ask: # Can you give me an example? # the model receives the previous user question and assistant answer too.
        # So it understands what “it” refers to.

        history.append({"role": "assistant", "content": reply})
        
if __name__ == "__main__":
    main()
