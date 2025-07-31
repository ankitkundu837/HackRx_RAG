import pdfplumber
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter

def chunks_from_pdf(file_path):
    print(f"Extracting text from {file_path}...")
    
    meaningful_texts = []
    footer_keywords = {"super splendor", "page", "hero motocorp"}
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if text:
                # Clean each page's text
                cleaned_lines = []
                for line in text.splitlines():
                    line = line.strip()
                    # Filter out short lines or common noise patterns
                    if len(line) < 20:
                        continue
                    if any(keyword in line.lower() for keyword in footer_keywords):
                        continue
                    cleaned_lines.append(line)
                
                if cleaned_lines:
                    meaningful_texts.append("\n".join(cleaned_lines))

    # Combine into one large text block
    full_text = "\n\n".join(meaningful_texts)

    # First split by characters (to keep semantic structure)
    character_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    char_chunks = character_splitter.split_text(full_text)

    # Then split by token length for model compatibility
    token_splitter = TokenTextSplitter(
        chunk_size=256,
        chunk_overlap=0
    )
    token_chunks = []
    for chunk in char_chunks:
        token_chunks.extend(token_splitter.split_text(chunk))

    print(f"Extracted {len(token_chunks)} meaningful chunks from {file_path}.")
    return token_chunks


def pdf_token_stream(file_path, max_tokens=60000, chunk_token_limit=256):
    splitter = TokenTextSplitter(chunk_size=chunk_token_limit, chunk_overlap=0)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    buffer = []
    token_count = 0
    footer_keywords = {"page", "hero motocorp", "indian constitution", "super splendor"}

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if not text:
                continue

            lines = [
                line.strip()
                for line in text.splitlines()
                if len(line.strip()) > 20 and not any(k in line.lower() for k in footer_keywords)
            ]

            joined = "\n".join(lines)
            chunks = splitter.split_text(joined)

            for chunk in chunks:
                tokens = tokenizer.encode(chunk)
                token_len = len(tokens)

                if token_len > max_tokens:
                    continue  # skip abnormally long chunks

                if token_count + token_len > max_tokens:
                    yield buffer
                    buffer = []
                    token_count = 0

                buffer.append(chunk)
                token_count += token_len

    if buffer:
        yield buffer