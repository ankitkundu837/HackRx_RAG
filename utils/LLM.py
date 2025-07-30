import os
import re
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

token = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

def generate_response(question, relevant_chunks):
    print(f"Generating response for question: {question}")
    context = "\n\n".join(relevant_chunks)

    response = client.chat.completions.create(
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant that answers user questions using retrieved documents "
                    "from a knowledge base. Your answers must strictly rely on the provided context. "
                    "If the answer is not in the context, respond with 'I don't know based on the provided information.' "
                    "Be precise, concise, and do not add any extra information outside the context. "
                    "Respond in a **single, concise line** with no extra commentary or formatting."
                )
            },
            {
                "role": "user",
                "content": (
                    "Context:\n" + context + "\n\n"
                    "Question:\n" + question
                )
            }
        ],
        temperature=0,
        top_p=1,
        model=model
    )

    answer = response.choices[0].message.content
    print("==== Generated Answer ====")
    return answer

def generate_batch_responses(questions: list[str], relevant_chunks: list[list[str]]):
    assert len(questions) == len(relevant_chunks), "Mismatch between questions and chunks"

    print(f"Generating batch response for {len(questions)} questions")

    # Only include specific numbered context and questions
    batch_prompt = ""
    for i, (q, chunks) in enumerate(zip(questions, relevant_chunks), start=1):
        context = "\n".join(chunks)
        batch_prompt += f"Context {i}:\n{context}\nQuestion {i}:\n{q}\n\n"

    batch_prompt += "Provide your answers in the following format:\nAnswer 1: ...\nAnswer 2: ... etc.\n"

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert assistant that answers multiple user questions based on provided context. "
                    "Each question is numbered and has a corresponding context. "
                    "You must rely only on the corresponding context for each question. Be concise. "
                    "If an answer is not in the context, say 'I don't know based on the provided information.'"
                    "For each question respond in a **single, concise line** with no extra commentary or formatting."
                )
            },
            {
                "role": "user",
                "content": batch_prompt
            }
        ],
        model=model,
        temperature=0,
        top_p=1,
    )

    raw_output = response.choices[0].message.content.strip()

    # Use regex to robustly extract numbered answers
    pattern = r"Answer (\d+):\s*(.*?)(?=(?:\nAnswer \d+:|$))"
    matches = re.findall(pattern, raw_output, re.DOTALL)

    # Map answers by index
    answer_map = {int(num): ans.strip() for num, ans in matches}

    # Return answers in input order
    answers = [answer_map.get(i + 1, "No answer found.") for i in range(len(questions))]
    
    return answers




