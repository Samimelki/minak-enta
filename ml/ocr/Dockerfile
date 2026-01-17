# Minak Enta - OCR Service
# Requires Tesseract OCR and related system dependencies

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OCR
# cmake, make, g++ are needed to compile zxing-cpp
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    cmake \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Remove build tools to reduce image size
RUN apt-get update && apt-get remove -y cmake make g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY main.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 35001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:35001/health')" || exit 1

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "35001"]
