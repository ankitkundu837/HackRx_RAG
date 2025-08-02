
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def ask_groq(question):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        temperature=0,
        top_p=1,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

questions = [
    "What is the purpose of the i3s (Idle Stop Start System) in the Super Splendor motorcycle?",
    "Explain quantum entanglement.",
    "How does photosynthesis work?",
    "What causes rainbows?"
]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(ask_groq, questions))

for q, a in zip(questions, results):
    print(f"\nQ: {q}\nA: {a}")
