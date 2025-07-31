import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()

token = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

client = OpenAI(base_url=endpoint, api_key=token)


def get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text, model=model):
    enc = get_encoding(model)
    return len(enc.encode(text))


def limit_chunks_by_token(chunks, model=model, max_tokens=2000):
    enc = get_encoding(model)
    total = 0
    selected = []

    for chunk in chunks:
        tokens = len(enc.encode(chunk))
        if total + tokens > max_tokens:
            break
        selected.append(chunk)
        total += tokens

    return selected


def generate_response(question, relevant_chunks):
    print(f"Generating response for question: {question}")
    context = "\n\n".join(limit_chunks_by_token(relevant_chunks))

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant that answers user questions using retrieved documents "
                "from a knowledge base. Your answers must strictly rely on the provided context. "
                "If the answer is not in the context, respond with 'I don't know based on the provided information.' "
                "Be precise, concise, and do not add any extra information outside the context. "
                "Respond in a single, concise line with no extra commentary or formatting."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        top_p=1,
    )

    return response.choices[0].message.content.strip()


def generate_batch_responses(questions: list[str], relevant_chunks: list[list[str]]):
    assert len(questions) == len(relevant_chunks), "Mismatch between questions and context lists"
    print(f"Generating batch response for {len(questions)} questions")

    enc = get_encoding(model)
    max_tokens = 7500
    system_tokens = count_tokens(
        "You are an expert assistant that answers multiple user questions based on provided context...",
        model=model,
    )

    batch = []
    current_tokens = system_tokens
    responses = []
    indexes = []

    for i, (question, chunks) in enumerate(zip(questions, relevant_chunks)):
        limited = limit_chunks_by_token(chunks, model=model, max_tokens=2000)
        context = "\n".join(limited)
        entry = f"Context {i+1}:\n{context}\nQuestion {i+1}:\n{question}\n\n"
        entry_tokens = len(enc.encode(entry))

        if current_tokens + entry_tokens > max_tokens:
            responses += _run_batch(indexes, batch)
            batch = [entry]
            indexes = [i]
            current_tokens = system_tokens + entry_tokens
        else:
            batch.append(entry)
            indexes.append(i)
            current_tokens += entry_tokens

    if batch:
        responses += _run_batch(indexes, batch)

    return responses


def _run_batch(indexes, batch_entries):
    prompt = "".join(batch_entries) + "Provide your answers in the format:\n" + "\n".join(
        f"Answer {i + 1}:" for i in range(len(indexes))
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant that answers multiple user questions based on provided context. "
                "Each question is numbered and has a corresponding context. "
                "You must rely only on the corresponding context for each question. Be concise. "
                "If an answer is not in the context, say 'I don't know based on the provided information.' "
                "For each question respond in a single, concise line with no extra commentary or formatting, but mention every relevant detail given in the document don't miss anthing."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        top_p=1,
    )

    output = response.choices[0].message.content.strip()
    pattern = r"Answer (\d+):\s*(.*?)(?=(?:\nAnswer \d+:|$))"
    matches = re.findall(pattern, output, re.DOTALL)

    answer_map = {int(num): ans.strip() for num, ans in matches}
    return [answer_map.get(i + 1, "No answer found.") for i in range(len(indexes))]