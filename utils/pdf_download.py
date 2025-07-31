import os
import requests
from fastapi import HTTPException

def download_and_store_pdf(document_url: str, save_dir: str = "./data") -> str:
    response = requests.get(str(document_url))
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download document")

    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    filename = os.path.basename(str(document_url).split("?")[0])

    pdf_path = os.path.join(save_dir, filename)

    with open(pdf_path, "wb") as f:
        f.write(response.content)

    return pdf_path