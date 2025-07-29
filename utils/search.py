from utils.chroma_functions import (get_chroma_collection,load_chroma)

def query_documents(question, n_results=5):

    collection=get_chroma_collection("documents")

    load_chroma("./data/policy.pdf", collection)

    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks