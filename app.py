#removed api key from code to prevent missuse of it
import openai from OpenAI
import requests
Client = OpenAI{
api_key="",
base_url="https://api.perplexity.ai"}
response=client.chat.completions.create{
model="sonar-pro",
messsages=[{"role":"system","content":"you are a helpfull assistant."},
           {"role":"user","content":"heloo!!, what can you do?"}
          ]
print(response.choices[0].message.content)


url="https://api.perplexity.ai/v1/chat/completions"
heades={"Authorization":f"Bearer YOUR_PERLEXITY_API_KEY",
        "Content-Type":"application/json"}
data={"model":"sonar-pro","messages":[{"role":"system","content":"you are a helpfull assistant."},
           {"role":"user","content":"heloo!!, what can you do?"}
          ]
     }
response= requests.post(url,headers=headers,json=data)
print(response.json()["choices"][0]["message"]["content"])
           

