FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System packages for pdfplumber and sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Pre-install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch torchvision \
 && pip install -r requirements.txt

# Copy source code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
