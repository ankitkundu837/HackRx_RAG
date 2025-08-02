from fastapi import FastAPI, HTTPException, Request, Header, Depends, status, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import os
import tempfile
import requests
import time
from utils.chroma_functions import get_chroma_collection, load_chroma, drop_chroma_collection
# from utils.LLM import generate_batch_responses
from dotenv import load_dotenv
from utils.local_LLM import generate_batch_responses
from utils.pdf_download import download_and_store_pdf
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import threading

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


# Constants
TIMEOUT_SECONDS = 80

# Default fallback response
DEFAULT_RESPONSE = ["Answer not present in document"] * 3

@app.post("/hackrx/run", response_model=QueryResponse)
def run_query(
    payload: QueryRequest,
    authorized: bool = Depends(verify_token),
    background_tasks: BackgroundTasks = None
):
    def process_query():
        start_time = time.time()

        try:
            step_start = time.time()
            print("Received documents URL:", payload.documents)
            pdf_path = download_and_store_pdf(payload.documents)
            print("PDF downloaded and stored at:", pdf_path)
            print("⏱️ Time taken for PDF download:", round(time.time() - step_start, 4), "seconds")

            step_start = time.time()
            print("Loading PDF into Chroma collection...")
            collection = get_chroma_collection("documents")
            load_chroma(pdf_path, payload.questions, collection)
            print("⏱️ Time taken for loading PDF into Chroma:", round(time.time() - step_start, 4), "seconds")

            step_start = time.time()
            print(f"Querying documents for questions: {payload.questions}")
            results = collection.query(query_texts=payload.questions, n_results=4)
            relevant_chunks = results["documents"]
            for i, question in enumerate(payload.questions):
                print(f"\n🔹 Question {i+1}: {question}")
                chunks = relevant_chunks[i]
                for j, chunk in enumerate(chunks):
                    print(f"  - Chunk {j+1}: {chunk}")
            print("⏱️ Time taken for querying:", round(time.time() - step_start, 4), "seconds")

            step_start = time.time()
            print("Generating batch responses for questions...")
            answers = generate_batch_responses(payload.questions, relevant_chunks)
            print("⏱️ Time taken for answer generation:", round(time.time() - step_start, 4), "seconds")

            # Background cleanup
            def cleanup():
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                drop_chroma_collection("documents")
            background_tasks.add_task(cleanup)

            duration = time.time() - start_time
            print("⏱️ Total response time:", round(duration, 4))
            return QueryResponse(success=True, answers=answers)

        except Exception as e:
            print("Error occurred during processing:", e)
            raise

    # Use ThreadPoolExecutor to enforce timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process_query)
        try:
            return future.result(timeout=TIMEOUT_SECONDS)
        except FutureTimeout:
            print(f"⏰ Timeout: Processing exceeded {TIMEOUT_SECONDS} seconds")
            return QueryResponse(success=False, answers=["Answer not present in document"] * len(payload.questions))
        except Exception as e:
            print("Unhandled Exception:", e)
            raise HTTPException(status_code=500, detail="Internal server error")