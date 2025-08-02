import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.pdf_parser import chunks_from_pdf
from typing import List, Tuple
import torch
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# GPU/CPU setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Local embedding model
embedding_model = SentenceTransformer('./all-MiniLM-L6-v2', device=DEVICE)

# Chroma collection
def get_chroma_collection(collection_name: str):
    return client.get_or_create_collection(name=collection_name)

# Embed a batch
def embed(texts: List[str]) -> List[List[float]]:
    return embedding_model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)

# TF-IDF chunk retriever
def retrieve_top_chunks(questions: List[str], chunks: List[str], top_k: int = 10) -> List[Tuple[str, str]]:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    chunk_vectors = vectorizer.fit_transform(chunks)
    question_vectors = vectorizer.transform(questions)

    relevant_chunk_indices = set()
    for q_vec in question_vectors:
        sims = cosine_similarity(q_vec, chunk_vectors).flatten()
        top_idxs = np.argsort(sims)[-top_k:]
        relevant_chunk_indices.update(top_idxs)

    filtered_chunks = [chunks[i] for i in relevant_chunk_indices]
    filtered_ids = [str(i) for i in relevant_chunk_indices]
    return filtered_ids, filtered_chunks

# Main loader using Query-First
def load_chroma(filename: str, questions: List[str], chroma_collection, top_k: int = 10, max_workers: int = 16):
    start = time.time()
    print("📄 Parsing PDF...")
    chunks = chunks_from_pdf(filename)
    print(f"📦 {len(chunks)} chunks extracted.")

    print("🔎 Using TF-IDF to find top relevant chunks for each question...")
    ids, selected_chunks = retrieve_top_chunks(questions, chunks, top_k=top_k)
    print(f"✅ Selected {len(selected_chunks)} chunks.")

    print("🧠 Embedding selected chunks...")
    t0 = time.time()
    batch_size = 64
    batches = [selected_chunks[i:i + batch_size] for i in range(0, len(selected_chunks), batch_size)]
    embeddings = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed, batch) for batch in batches]
        for future in as_completed(futures):
            embeddings.extend(future.result())
    print(f"✅ Embedding done in {round(time.time() - t0, 2)} sec")

    print("📥 Adding to ChromaDB...")
    def add_batch(batch_ids, batch_docs, batch_embs):
        chroma_collection.add(ids=batch_ids, documents=batch_docs, embeddings=batch_embs)

    batches = [
        (ids[i:i + batch_size], selected_chunks[i:i + batch_size], embeddings[i:i + batch_size])
        for i in range(0, len(selected_chunks), batch_size)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(add_batch, *batch) for batch in batches]
        for future in as_completed(futures):
            future.result()

    print(f"✅ ChromaDB load complete in {round(time.time() - start, 2)} sec")

# Drop collection
def drop_chroma_collection(collection_name: str):
    client.delete_collection(name=collection_name)
