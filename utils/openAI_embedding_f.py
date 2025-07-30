import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("GITHUB_TOKEN")

client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=token,
)

class OpenAIEmbeddingFunction:
    def __call__(self, input):  
        response = client.embeddings.create(
            input=input,
            model="openai/text-embedding-3-small",
        )
        return [item.embedding for item in response.data]

    def name(self) -> str:
        return "openai-text-embedding-3-small"
