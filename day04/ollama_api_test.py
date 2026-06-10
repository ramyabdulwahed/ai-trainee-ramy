import requests #lets python send HTTP requests, like a browser nstead of opening a website manually

url = "http://localhost:11434/api/chat" #adress we are sending request to. ex.Send a chat request to the local Ollama server running on my machine.

#payload below:
#Which model do you want?
# What message do you want to send?
# Do you want the answer all at once or streamed piece by piece?
payload = {
    "model": "llama3.2:3b",
    "messages": [
        {
            "role": "user",
            "content": "Explain attention in transformers in 3 simple sentences."
        }
    ],
    "stream": False #Wait until the model finishes the full answer, then send it back all at once
}

#sends post request to ollama,json=payload part tells Python: Send this dictionary as JSON data
response = requests.post (url, json=payload)

data = response.json() # converts that JSON into a Python dictionary so you can access its fields
#gets the message object from the response then the content
print (data ["message"] ["content"])