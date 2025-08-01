from utils.chroma_functions import (get_chroma_collection,load_chroma)

def query_documents(questions: list[str]):
    collection = get_chroma_collection("documents")
    


    load_chroma("./data/policy.pdf", collection)

    print(f"Querying documents for questions: {questions}")
    
    # Query Chroma with the list of questions
    results = collection.query(query_texts=questions, n_results=6)

    relevant_chunks = results["documents"]

    print("==== Relevant Chunks Retrieved ====")
    return relevant_chunks
