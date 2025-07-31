
import chromadb
from utils.pdf_parser import chunks_from_pdf
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

client = chromadb.PersistentClient(path="./chroma_db")

embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# embedding_function = SentenceTransformerEmbeddingFunction()

def get_chroma_collection(collection_name: str):
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

def load_chroma(filename, chroma_collection):
    chunks_split_texts = chunks_from_pdf(filename)
    ids = [str(i) for i in range(len(chunks_split_texts))]
    chroma_collection.add(ids=ids, documents=chunks_split_texts)
    chroma_collection.count()

def drop_chroma_collection(collection_name: str):
    client.delete_collection(name=collection_name)
