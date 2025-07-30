import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from utils.pdf_parser import chunks_from_pdf
from utils.openAI_embedding_f import OpenAIEmbeddingFunction  # Adjust path if needed
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "pineconeidx"

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# Initialize embedding function
embedding_function = OpenAIEmbeddingFunction()

def load_pinecone(filename, batch_size=50):
    chunks_split_texts = chunks_from_pdf(filename)
    print(f"Extracted {len(chunks_split_texts)} chunks from PDF.")

    ids = [str(i) for i in range(len(chunks_split_texts))]
    embeddings = embedding_function(chunks_split_texts)

    vectors = []
    for id_, chunk, emb in zip(ids, chunks_split_texts, embeddings):
        # Convert numpy array embeddings to list if needed
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()

        # Debug check for each vector
        if not isinstance(emb, list) or len(emb) != 1536:
            raise ValueError(f"Embedding for id {id_} has wrong format or dimension: {type(emb)} length={len(emb) if isinstance(emb, list) else 'NA'}")

        vectors.append({
            "id": id_,
            "values": emb,
            "metadata": {"chunk_text": chunk}
        })
    print(f"Prepared {len(vectors)} vectors for upsert.")

    # Batch upsert
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        print(f"Upserting batch {i // batch_size + 1} with {len(batch)} vectors...")
        index.upsert(vectors=batch, namespace="default")

    print(f"Upserted total {len(vectors)} text chunks into index '{INDEX_NAME}'.")

def delete_pinecone_index():
    print(f"Deleting index {INDEX_NAME}...")
    pc.delete_index(INDEX_NAME)
    print("Index deleted.")

if __name__ == "__main__":
    test_pdf = "./data/sample.pdf"  # Change path as needed
    load_pinecone(test_pdf)
