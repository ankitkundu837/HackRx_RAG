import time
import fitz
import re
import subprocess
import os
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from tempfile import NamedTemporaryFile

TABULA_JAR_PATH = "/usr/share/tabula/tabula.jar"  # Adjust path if needed


# ---------- CLEANING ----------
def clean_lines(text):
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join([line for line in lines if len(line) > 20])


def clean_combined_text(text):
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ---------- TEXT EXTRACTION ----------
def extract_text_page(args):
    page_num, file_path = args
    doc = fitz.open(file_path)
    text = doc[page_num].get_text("text")
    return clean_lines(text) if text else ""


# ---------- TABLE EXTRACTION PER PAGE ----------
def extract_table_page(args):
    page_num, file_path = args
    try:
        dfs = pd.read_pdf(
            file_path,
            pages=str(page_num + 1),  # tabula uses 1-based indexing
            multiple_tables=True,
            lattice=True
        )
        tables = []
        for df in dfs:
            if isinstance(df, pd.DataFrame) and not df.empty:
                markdown = df.to_markdown(index=False)
                tables.append(markdown)
        return "\n\n".join(tables) if tables else ""
    except Exception as e:
        return f"# Table extraction failed on page {page_num + 1}: {e}"


# ---------- MAIN FUNCTION ----------
def chunks_from_pdf(file_path, model_name="gpt-4"):
    step_start = time.time()
    print(f"\n🔍 Processing PDF: {file_path}")

    doc = fitz.open(file_path)
    total_pages = len(doc)
    page_args = [(i, file_path) for i in range(total_pages)]

    # Parallel text extraction
    print(f"⚡ Extracting text with {cpu_count()} workers...")
    with Pool(processes=cpu_count()) as pool:
        page_texts = pool.map(extract_text_page, page_args)

    # Safe table extraction using external processes
    print(f"📊 Extracting tables safely...")
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        page_tables = list(executor.map(extract_table_page, page_args))

    # Combine text + tables
    combined = []
    for text, table in zip(page_texts, page_tables):
        combined_page = text
        if table:
            combined_page += "\n\n## Extracted Tables\n" + table
        combined.append(combined_page)

    # Clean full text
    full_text = "\n\n".join(filter(None, combined))
    cleaned_text = clean_combined_text(full_text)

    # Chunking
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
        chunk_size=256,
        chunk_overlap=50
    )
    chunks = splitter.split_text(cleaned_text)

    print("⏱️ Time taken for text and table extraction:", round(time.time() - step_start, 4), "seconds")

    print(f"🧩 Total Chunks Generated: {len(chunks)}")
    return [c for c in chunks if c and isinstance(c, str)]
