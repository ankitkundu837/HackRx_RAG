# Use lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for pdfplumber + Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Pre-install CPU-only torch
RUN pip install --upgrade pip \
 && pip install torch==2.2.2+cpu torchvision==0.17.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy requirements and install remaining
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Run app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
