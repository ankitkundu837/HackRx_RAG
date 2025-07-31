
import chromadb
from utils.pdf_parser import chunks_from_pdf
from utils.openAI_embedding_f import OpenAIEmbeddingFunction

client = chromadb.PersistentClient(path="./chroma_db")

embedding_function = OpenAIEmbeddingFunction()
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
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)
        print(f"Collection [{collection_name}] deleted.")
    else:
        print(f"Collection [{collection_name}] not found.")
