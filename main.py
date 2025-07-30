from utils.search import query_documents
from utils.LLM import generate_response, generate_batch_responses

question = "What is Accident?"
questions = [
    "waiting period for Tonsillectomy?",
    "The contact details of the Insurance Ombudsman offices in Rajasthan",
    "What is Accident?"
]

# relevant_chunks = query_documents(question)
# print("==== Relevant Chunks ====")
# print(relevant_chunks)

# answer = generate_response(question, relevant_chunks)
# print(answer)

relevant_chunks = query_documents(questions)

# Generate answers
answers = generate_batch_responses(questions, relevant_chunks)

# Print the answers
for i, ans in enumerate(answers, start=1):
    print(f"Answer {i}: {ans}")