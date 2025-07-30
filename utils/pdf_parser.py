import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter


def chunks_from_pdf(file_path):
    structured_texts = []
    print(f"Extracting text from {file_path}...")
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True)
            if text:
                structured_texts.append(text.strip())

    character_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = character_splitter.split_text("\n\n".join(structured_texts))

    token_splitter = TokenTextSplitter(
        chunk_size=256,
        chunk_overlap=0
    )

    token_chunks = []
    for text in chunks:
        token_chunks.extend(token_splitter.split_text(text))

    print(f"Extracted {len(token_chunks)} chunks from {file_path}.")
    return token_chunks
