FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Preinstall torch CPU-only before other heavy libs
RUN pip install --upgrade pip \
 && pip install torch==2.1.2+cpu torchvision==0.16.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy requirements and install rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
