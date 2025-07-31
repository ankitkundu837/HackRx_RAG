from utils.pdf_download import download_and_store_pdf
from utils.pinecone_functions import delete_pinecone_index, load_pdf_to_pinecone, load_pinecone, reset_index
from utils.search import query_documents
from utils.LLM import generate_response, generate_batch_responses
import os


question = "What is Accident?"
questions =  [
        "Is Non-infective Arthritis covered?", 
        "I renewed my policy yesterday, and I have been a customer for the last 6 years. Can I raise a claim for Hydrocele?", 
        "Is abortion covered?"
    ]

# print("Received documents URL:", payload.documents)
documents="https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D"
print("Received documents URL:", documents)
pdf_path = download_and_store_pdf(documents)
print("PDF downloaded and stored at:", pdf_path)

print("Loading PDF into Pinecone index...")
load_pdf_to_pinecone(pdf_path)

print("PDF loaded into Pinecone successfully.")

print(f"Querying Pinecone index for questions: {questions}")
relevant_chunks = query_documents(questions)

print("Generating batch responses for questions...")
answers = generate_batch_responses(questions, relevant_chunks)
print("Batch responses generated successfully.")
for idx, ans in enumerate(answers, 1):
    print(f"Answer {idx}: {ans}")

def cleanup():
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    # Optionally delete the Pinecone index if needed:
    delete_pinecone_index()

cleanup()