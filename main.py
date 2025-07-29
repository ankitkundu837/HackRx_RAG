from utils.search import query_documents
from utils.LLM import generate_response

question = "What is Accident?"
relevant_chunks = query_documents(question)
# print("==== Relevant Chunks ====")
# print(relevant_chunks)
answer = generate_response(question, relevant_chunks)

print(answer)