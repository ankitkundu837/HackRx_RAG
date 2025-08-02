import requests

MODEL = "phi3"  # or "llama3" or any model you have pulled and running locally
OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n".join(context_chunks)
    prompt = f"""You are an expert assistant that answers user questions using only the provided contexts.\n"
        "If the answer is not in the document, reply with: 'I don't know based on the provided information.'\n"
        "Mention every relevant detail related to the question provided in context\n"
        "Answer in the following format:\n"
        "Answer: your answer here\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"""
    return prompt


def call_local_llm(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Local LLM response: {data}")
        return data.get("response", "").strip()
    except requests.RequestException as e:
        print(f"Error calling local model: {e}")
        return "[Error generating response]"


def generate_batch_responses(questions: list[str], relevant_chunks: list[list[str]]) -> list[str]:
    responses = []

    for question, chunks in zip(questions, relevant_chunks):
        prompt = generate_prompt(question, chunks)
        response = call_local_llm(prompt)
        responses.append(response)

    return responses