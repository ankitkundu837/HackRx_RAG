import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from utils.pdf_parser import chunks_from_pdf


def get_chroma_collection(collection_name: str):
    client = chromadb.PersistentClient(path="./chroma_db") 

    # Ankit's Note: Can change the embedding function to openAI embedding function for more accuracy
    embedding_function = SentenceTransformerEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    return collection


def load_chroma(filename, chroma_collection):

    chunks_split_texts = chunks_from_pdf(filename)

    ids = [str(i) for i in range(len(chunks_split_texts))]
    chroma_collection.add(ids=ids, documents=chunks_split_texts)
    chroma_collection.count()