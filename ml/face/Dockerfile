# Minak Enta - Face Recognition Service
# Requires InsightFace and ONNX Runtime

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and InsightFace
# g++ is needed to compile insightface C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Remove build tools to reduce image size
RUN apt-get update && apt-get remove -y g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY main.py .

# Create model cache directory
RUN mkdir -p /app/.insightface/models

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set InsightFace model directory
ENV INSIGHTFACE_HOME=/app/.insightface

# Expose port
EXPOSE 35002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:35002/health')" || exit 1

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "35002"]
