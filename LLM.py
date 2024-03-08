import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv('../AI_phone_agent_v1/.env.py')
user_input = "Hi there, what is the color of the sky?"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "system", "content": "You are a friendly AI assistant. Ask me anything!"},
        {"role": "user", "content": user_input}
    ]
)
print(response.choices[0].message.content)