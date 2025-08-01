import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from multiprocessing import Pool, cpu_count

footer_keywords = {"super splendor", "page", "hero motocorp"}


def process_page_text(page_num_file):
    page_num, file_path = page_num_file
    with pdfplumber.open(file_path) as pdf:
        page = pdf.pages[page_num]
        text = page.extract_text(layout=True)
        if not text:
            return None

        cleaned_lines = []
        for line in text.splitlines():
            line = line.strip()
            if len(line) < 20:
                continue
            if any(keyword in line.lower() for keyword in footer_keywords):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines) if cleaned_lines else None


def chunks_from_pdf(file_path):
    print(f"📄 Extracting text from {file_path} with multiprocessing...")

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)

    page_args = [(i, file_path) for i in range(total_pages)]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_page_text, page_args)

    meaningful_texts = list(filter(None, results))
    full_text = "\n\n".join(meaningful_texts)

    token_splitter = TokenTextSplitter(
        chunk_size=256,
        chunk_overlap=60
    )
    token_chunks = token_splitter.split_text(full_text)

    print(f"✅ Extracted {len(token_chunks)} meaningful chunks.")
    return token_chunks
