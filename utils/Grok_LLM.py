import os
import re
import tiktoken
from typing import List
from concurrent.futures import ThreadPoolExecutor
from groq import Groq

# Set your Groq API key here or through an environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_key_here")
MODEL = "llama3-8b-8192"
MAX_INPUT_TOKENS = 2000

client = Groq(api_key=GROQ_API_KEY)

def get_encoding(model: str = "gpt-4") -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = get_encoding(model)
    return len(enc.encode(text))

def limit_chunks_by_token(chunks: List[str], max_tokens: int = MAX_INPUT_TOKENS) -> List[str]:
    enc = get_encoding()
    total = 0
    selected = []
    for chunk in chunks:
        tokens = len(enc.encode(chunk))
        if total + tokens > max_tokens:
            break
        selected.append(chunk)
        total += tokens
    return selected

def generate_response(question: str, relevant_chunks: List[str]) -> str:
    print(f"🔎 Generating response for question: {question}")
    context = "\n\n".join(limit_chunks_by_token(relevant_chunks))

    prompt = (
        "You are an expert assistant that answers user questions using only the provided contexts.\n"
        "If the answer is not in the document, reply with: 'I don't know based on the provided information.'\n"
        "Mention every relevant detail related to the question provided in context\n"
        "Answer in the following format:\n"
        "Answer: your answer here\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            top_p=1,
            max_tokens=512
        )
    except Exception as e:
        print("❌ Groq API call failed:", e)
        return "ERROR"

    content = response.choices[0].message.content
    print("🧠 Raw LLM Response:\n", content, "\n")

    match = re.search(r"Answer:\s*(.*)", content)
    return match.group(1).strip() if match else "No answer found."

# 🧪 Example parallel usage
def run_parallel_queries(questions: List[str], relevant_chunks_list: List[List[str]]):
    with ThreadPoolExecutor(max_workers=min(len(questions), 4)) as executor:
        results = list(executor.map(generate_response, questions, relevant_chunks_list))
    return results