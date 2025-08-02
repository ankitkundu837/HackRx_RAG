from sentence_transformers import SentenceTransformer
from multiprocessing import Pool, cpu_count
import chromadb

# Load model once globally in each process
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts):
    model = get_model()
    return model.encode(texts, show_progress_bar=False).tolist()

def parallel_embed(chunks, batch_size=100):
    with Pool(processes=cpu_count()) as pool:
        batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
        results = pool.map(embed_texts, batches)
    # Flatten list of lists
    return [vec for batch in results for vec in batch]
