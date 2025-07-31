from fastapi import FastAPI, HTTPException, Request, Header, Depends, status, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import os
import tempfile
from dotenv import load_dotenv
from utils.LLM import generate_batch_responses
from utils.pdf_download import download_and_store_pdf
from utils.pinecone_functions import load_pdf_to_pinecone, load_pinecone  # ⬅️ replaced chroma
from utils.search import query_documents            # ⬅️ replaced chroma
from utils.pinecone_functions import delete_pinecone_index  # Optional cleanup

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def verify_token(authorization: Optional[str] = Header(None)) -> bool:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must start with 'Bearer '"
        )

    token = authorization.removeprefix("Bearer ").strip()

    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid token"
        )
    
    return True


@app.post("/hackrx/run", response_model=QueryResponse)
def run_query(
    payload: QueryRequest,
    authorized: bool = Depends(verify_token),
    background_tasks: BackgroundTasks = None
):
    try:
        print("Received documents URL:", payload.documents)
        pdf_path = download_and_store_pdf(payload.documents)
        print("PDF downloaded and stored at:", pdf_path)

        print("Loading PDF into Pinecone index...")
        load_pdf_to_pinecone(pdf_path)  # ⬅️ replaced Chroma
        print("PDF loaded into Pinecone successfully.")

        print(f"Querying Pinecone index for questions: {payload.questions}")
        relevant_chunks = query_documents(payload.questions)
        
        print("Generating batch responses for questions...")
        answers = generate_batch_responses(payload.questions, relevant_chunks)
        print("Batch responses generated successfully.")
        for idx, ans in enumerate(answers, 1):
            print(f"Answer {idx}: {ans}")

        def cleanup():
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            # Optionally delete the Pinecone index if needed:
            delete_pinecone_index()

        background_tasks.add_task(cleanup)

        return QueryResponse(answers=answers)

    except Exception as e:
        print("Error occurred:", e)
        raise HTTPException(status_code=500, detail="Internal server error")
