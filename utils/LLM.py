import os
import re
import requests
import tiktoken
from typing import List

API_URL = "https://67444dfd2e04.ngrok-free.app/api/generate"
MODEL = "llama3"


def get_encoding(model: str = "gpt-4") -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = get_encoding(model)
    return len(enc.encode(text))


def limit_chunks_by_token(chunks: List[str], max_tokens: int = 2000) -> List[str]:
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


def call_llama3_api(prompt: str) -> str:
    response = requests.post(
        API_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def generate_batch_responses(questions: List[str], relevant_chunks: List[List[str]]) -> List[str]:
    assert len(questions) == len(relevant_chunks), "Mismatch between questions and context lists"
    print(f"Generating individual responses for {len(questions)} questions")

    results = []
    for i, (question, chunks) in enumerate(zip(questions, relevant_chunks)):
        print(f"\n🔍 Processing Question {i + 1}")
        limited_chunks = limit_chunks_by_token(chunks, max_tokens=2000)
        # for i, chunk in enumerate(limited_chunks):
        #     print(f"\n🔹 Chunk {i + 1}:\n{chunk}")
        #     print("\n" + "-" * 50)

        context = "\n".join(limited_chunks)

        prompt = (
            "You are a precise assistant. You answer using only the information from the provided document context.\n"
            "Do not use any external knowledge or make assumptions.\n"
            "If the answer is not found in the context, say exactly:\n"
            "\"Not mentioned in the document.\" and nothing more.\n\n"
            "Provide a complete and detailed answer using all relevant facts from the context.\n"
            "Do not summarize briefly. Include every important point mentioned.\n"
            "Do not leave out any useful information.\n\n"
            "⚠️ Important formatting rules:\n"
            "- Respond in a single paragraph of plain text.\n"
            "- Do NOT use bullet points, numbering, line breaks, bold text, or markdown.\n"
            "- Do NOT repeat the question.\n\n"
            f"Context:\n{context.strip()}\n\n"
            f"Question:\n{question.strip()}\n\n"
            "Now provide the answer as a single paragraph of plain text:"
        )

        try:
            response = call_llama3_api(prompt)
            print("🧠 LLM Response:\n", response)
            results.append(response.strip())
        except Exception as e:
            print("❌ API call failed:", e)
            results.append("ERROR")

    return results