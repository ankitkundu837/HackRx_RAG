from fastapi import FastAPI, HTTPException, Request, Header, Depends, status, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import os
import tempfile
import requests
from utils.chroma_functions import get_chroma_collection, load_chroma, drop_chroma_collection
from utils.LLM import generate_batch_responses
from dotenv import load_dotenv
from utils.pdf_download import download_and_store_pdf

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    success: bool
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

        print("Loading PDF into Chroma collection...")
        collection = get_chroma_collection("documents")
        load_chroma(pdf_path, collection)
        print("PDF loaded into Chroma collection successfully.")

        print(f"Querying documents for questions: {payload.questions}")
        results = collection.query(query_texts=payload.questions, n_results=3)
        relevant_chunks = results["documents"]
        print("==== Returning relevant chunks ====")

        print("Generating batch responses for questions...")
        answers = generate_batch_responses(payload.questions, relevant_chunks)
        print("Batch responses generated successfully.")

        def cleanup():
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            drop_chroma_collection("documents")

        background_tasks.add_task(cleanup)

        return QueryResponse(success=True, answers=answers)


    except Exception as e:
        print("Error occurred:", e)
        raise HTTPException(status_code=500, detail="Internal server error")


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
