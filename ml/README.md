# Minak Enta ML Services

This directory contains the Python-based machine learning microservices for Minak Enta's identity verification system.

## Services Overview

### 🧠 OCR Service (`ml/ocr/`)
- **Port**: 5001
- **Purpose**: Extracts text from Lebanese ID card images
- **Technology**: Tesseract OCR with OpenCV preprocessing
- **Endpoint**: `POST /process-document`

### 👤 Face Recognition Service (`ml/face/`)
- **Port**: 5002
- **Purpose**: Face detection, embedding extraction, and comparison
- **Technology**: InsightFace (with OpenCV fallback)
- **Endpoints**:
  - `POST /detect-faces` - Detect faces in image
  - `POST /extract-embedding` - Extract face embedding
  - `POST /compare-faces` - Compare two face embeddings

### 👁️ Liveness Detection Service (`ml/liveness/`)
- **Port**: 5003
- **Purpose**: Detects if a face is from a live person (anti-spoofing)
- **Technology**: Computer vision heuristics with OpenCV
- **Endpoint**: `POST /detect-liveness`

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR** (for OCR service)
   ```bash
   brew install tesseract
   ```

### Installation

```bash
# Install dependencies for all services
cd ml/ocr && pip install -r requirements.txt && cd ../..
cd ml/face && pip install -r requirements.txt && cd ../..
cd ml/liveness && pip install -r requirements.txt && cd ../..
```

### Running Individual Services

```bash
# OCR Service
cd ml/ocr && python3 main.py

# Face Recognition Service
cd ml/face && python3 main.py

# Liveness Detection Service
cd ml/liveness && python3 main.py
```

### Running All Services

Use the convenience script from the project root:

```bash
./scripts/run-services.sh
```

This will:
1. Install Python dependencies
2. Start all ML services
3. Wait for services to be ready
4. Start the main Go API server
5. Provide service URLs and health check commands

## API Usage

### OCR Service

```bash
curl -X POST \
  -F "file=@lebanese_id.jpg" \
  -F "document_type=LEBANESE_ID_FRONT" \
  http://localhost:5001/process-document
```

**Response:**
```json
{
  "status": "SUCCESS",
  "confidence_score": 87.5,
  "extracted_data": {
    "name_arabic": "محمد علي حسن",
    "name_latin": "Mohamed Ali Hassan",
    "id_number": "123456789",
    "gender": "male",
    "nationality": "Lebanese"
  },
  "field_confidences": {...},
  "metadata": {...}
}
```

### Face Recognition Service

**Detect Faces:**
```bash
curl -X POST \
  -F "file=@photo.jpg" \
  http://localhost:5002/detect-faces
```

**Extract Embedding:**
```bash
curl -X POST \
  -F "file=@photo.jpg" \
  http://localhost:5002/extract-embedding
```

**Compare Faces:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"embedding1": [...], "embedding2": [...], "metric": "cosine"}' \
  http://localhost:5002/compare-faces
```

### Liveness Detection Service

```bash
curl -X POST \
  -F "file=@selfie.jpg" \
  http://localhost:5003/detect-liveness
```

**Response:**
```json
{
  "status": "SUCCESS",
  "liveness_score": 0.92,
  "confidence": 0.89,
  "analysis": {
    "face_detection": {...},
    "spoofing_detection": {...},
    "quality_assessment": {...}
  }
}
```

## Health Checks

All services provide health check endpoints:

```bash
curl http://localhost:5001/health  # OCR
curl http://localhost:5002/health  # Face
curl http://localhost:5003/health  # Liveness
```

## Development

### Testing Services Individually

Each service can be tested independently using the provided endpoints. Sample test images can be found in the `test/` directory or you can use any JPEG/PNG image.

### Adding New ML Models

1. Add model files to the appropriate service directory
2. Update the service code to load and use the new model
3. Update the requirements.txt if new dependencies are needed
4. Update API endpoints and response formats as needed

### Performance Optimization

- **GPU Support**: Services can be configured to use GPU acceleration (where available)
- **Model Caching**: Face recognition models are loaded once at startup
- **Batch Processing**: Services can be extended to support batch processing for multiple images

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Install with `brew install tesseract`
2. **InsightFace import error**: Install with `pip install insightface`
3. **Port conflicts**: Change ports in the service `main.py` files
4. **Memory issues**: Reduce batch sizes or model sizes for lower-end hardware

### Logs

Services log to stdout. For production deployment, consider:
- Log aggregation (ELK stack, etc.)
- Structured logging with JSON format
- Log levels configuration

## Production Deployment

For production use:

1. **Containerization**: Use Docker for each service
2. **Load Balancing**: Deploy multiple instances behind a load balancer
3. **Monitoring**: Add health checks and metrics endpoints
4. **Security**: Add authentication and rate limiting
5. **Scaling**: Use Kubernetes for auto-scaling based on load

## Dependencies

- **OCR Service**: FastAPI, OpenCV, Tesseract, NumPy
- **Face Service**: FastAPI, InsightFace, OpenCV, NumPy, ONNX Runtime
- **Liveness Service**: FastAPI, OpenCV, NumPy, SciPy

See individual `requirements.txt` files for exact versions.