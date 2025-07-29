import os
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
    return answer


