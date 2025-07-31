# Use slim Python base to reduce image size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for PDF + SSL
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Install CPU-only version of PyTorch to keep it lightweight
RUN pip install --upgrade pip \
 && pip install torch==2.2.2+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
 && pip install -r requirements.txt

# Copy your app files
COPY . .

# Expose port
EXPOSE 8000

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
