import os
import json
import base64
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
MODEL = "gpt-4o"
    
client = AzureOpenAI(
    api_key = os.getenv("API_KEY"),  
    api_version = os.getenv("API_VERSION"), 
    azure_endpoint = os.getenv("ENDPOINT")
)

user_name = "Justin"
transcript = json.load(open("./json/{}.json".format(user_name), 'r'))["speech_score"]["transcript"]
user_message = open("./prompt/image_user_prompt").read().replace("{{transcript}}", transcript)
system_message = open("./prompt/image_system_prompt", 'r', encoding="utf-8").read()

print("SYSTEM")
print(system_message)
print("USER")
print(user_message)

with open("./image/{}.png".format(user_name), 'rb') as image_file:
    image_data = image_file.read()
    user_image = base64.b64encode(image_data).decode('utf-8')


chat_completion = client.chat.completions.create(
    model = MODEL,
    temperature = 0.7,
    messages = [
        { "role": "system", "content": system_message },
        { "role": "user", "content": [  
            { 
                "type": "text", 
                "text": user_message 
            },
            { 
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{user_image}"
                }
            }
        ] } 
    ],
    max_tokens=2000 
)


try:
    print(chat_completion.choices[0].message.content)
except Exception as e:
    print(e)
    print(chat_completion)