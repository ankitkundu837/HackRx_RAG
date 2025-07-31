from utils.pinecone_functions import delete_pinecone_index, load_pinecone
from utils.search import query_documents
from utils.LLM import generate_response, generate_batch_responses
import os


question = "What is Accident?"
questions =  [
    "What is the purpose of the i3s (Idle Stop Start System) in the Super Splendor motorcycle?",
    "What is the correct procedure for starting the Super Splendor engine?",
    "How should the rider check the engine oil level in the Super Splendor?",
    "What safety features are associated with the side stand in the Super Splendor?",
    "What are Hero MotoCorp’s recommendations for safe and eco-friendly riding?"
  ]

# print("Received documents URL:", payload.documents)
pdf_path = "./data/Super_Splendor_(Feb_2023).pdf"
print("PDF downloaded and stored at:", pdf_path)

print("Loading PDF into Pinecone index...")
load_pinecone(pdf_path)  # ⬅️ replaced Chroma
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