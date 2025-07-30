from utils.openAI_embedding_f import OpenAIEmbeddingFunction
embedding_function = OpenAIEmbeddingFunction()
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

def query_documents(questions: list[str]):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("pineconeidx")
    print(f"Querying documents for questions: {questions}")
    all_chunks = []

    # Step 1: Get embeddings for the questions using your class
    embeddings = embedding_function(questions)  # Calls __call__

    for question, vector in zip(questions, embeddings):
        # Sanity check
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        if not isinstance(vector, list) or len(vector) != 1536:
            raise ValueError(f"Invalid embedding for question '{question}': {type(vector)} / len={len(vector)}")

        result = index.query(
            namespace="default",
            top_k=3,
            include_metadata=True,
            vector=vector  # ✅ Proper dense embedding vector
        )

        chunks = [match["metadata"]["chunk_text"] for match in result.get("matches", [])]
        all_chunks.append(chunks)

    print("==== Returning relevant chunks ====")
    return all_chunks
