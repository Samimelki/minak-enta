#!/usr/bin/env python3
"""
Minak Enta Face Recognition Service
Detects faces, extracts embeddings, and compares faces for verification
"""

import logging
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn

# Try to import insightface, fallback to basic OpenCV if not available
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available, using basic OpenCV face detection")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Minak Enta Face Recognition Service", version="1.0.0")

class FaceConfig:
    """Face recognition configuration"""
    MIN_FACE_SIZE = 50
    MAX_FACES = 10
    SIMILARITY_THRESHOLD = 0.6

class FaceDetector:
    """Face detection and recognition handler"""

    def __init__(self):
        self.model = None
        if INSIGHTFACE_AVAILABLE:
            try:
                self.model = FaceAnalysis(name='buffalo_l')  # Larger model, better accuracy
                self.model.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("InsightFace model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load InsightFace model: {e}")
                self.model = None
        else:
            logger.warning("Using fallback OpenCV face detection")

    def detect_faces(self, image: np.ndarray, max_faces: int = FaceConfig.MAX_FACES) -> List[dict]:
        """Detect faces in an image"""
        if self.model and INSIGHTFACE_AVAILABLE:
            return self._detect_insightface(image, max_faces)
        else:
            return self._detect_opencv(image, max_faces)

    def _detect_insightface(self, image: np.ndarray, max_faces: int) -> List[dict]:
        """Detect faces using InsightFace"""
        faces = self.model.get(image)

        results = []
        for i, face in enumerate(faces[:max_faces]):
            bbox = face.bbox.astype(int)
            results.append({
                'face_id': i,
                'detection_confidence': float(face.det_score),
                'bounding_box': {
                    'x': int(bbox[0]),
                    'y': int(bbox[1]),
                    'width': int(bbox[2] - bbox[0]),
                    'height': int(bbox[3] - bbox[1])
                },
                'embedding': face.embedding.tolist() if hasattr(face, 'embedding') else None
            })

        return results

    def _detect_opencv(self, image: np.ndarray, max_faces: int) -> List[dict]:
        """Fallback face detection using OpenCV Haar cascades"""
        # Load Haar cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(FaceConfig.MIN_FACE_SIZE, FaceConfig.MIN_FACE_SIZE)
        )

        results = []
        for i, (x, y, w, h) in enumerate(faces[:max_faces]):
            results.append({
                'face_id': i,
                'detection_confidence': 0.8,  # OpenCV doesn't provide confidence scores
                'bounding_box': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                },
                'embedding': None  # OpenCV doesn't provide embeddings
            })

        return results

    def extract_embedding(self, image: np.ndarray) -> Optional[List[float]]:
        """Extract face embedding from image"""
        if not self.model or not INSIGHTFACE_AVAILABLE:
            return None

        faces = self.model.get(image)
        if len(faces) == 1:
            return faces[0].embedding.tolist()
        elif len(faces) > 1:
            # Use the largest face
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return largest_face.embedding.tolist()

        return None

    def compare_faces(self, embedding1: List[float], embedding2: List[float], metric: str = "cosine") -> float:
        """Compare two face embeddings"""
        if not embedding1 or not embedding2:
            return 0.0

        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        if metric == "cosine":
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        elif metric == "euclidean":
            # Euclidean distance (lower is more similar)
            distance = np.linalg.norm(emb1 - emb2)
            # Convert to similarity score (0-1, higher is better)
            return max(0, 1 - distance / 100)  # Normalize distance
        else:
            return 0.0

# Initialize face detector
face_detector = FaceDetector()

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for face detection"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image data")

    return image

@app.post("/detect-faces")
async def detect_faces(
    file: UploadFile = File(...),
    max_faces: int = FaceConfig.MAX_FACES,
    min_confidence: float = 0.5
):
    """Detect faces in an image"""
    start_time = time.time()

    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        faces = face_detector.detect_faces(image, max_faces)

        # Filter by confidence
        faces = [face for face in faces if face['detection_confidence'] >= min_confidence]

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "status": "SUCCESS" if faces else "NO_FACES_DETECTED",
            "faces": faces,
            "metadata": {
                "processing_time_ms": processing_time,
                "model_version": "InsightFace" if INSIGHTFACE_AVAILABLE else "OpenCV",
                "processed_at": time.time()
            }
        }

    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")

@app.post("/extract-embedding")
async def extract_embedding(
    file: UploadFile = File(...),
    max_faces: int = 1,
    min_confidence: float = 0.5
):
    """Extract face embedding from image"""
    start_time = time.time()

    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        # Detect faces first
        faces = face_detector.detect_faces(image, max_faces)
        faces = [face for face in faces if face['detection_confidence'] >= min_confidence]

        if not faces:
            return {
                "status": "NO_FACES_DETECTED",
                "faces": [],
                "metadata": {
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "model_version": "InsightFace" if INSIGHTFACE_AVAILABLE else "OpenCV",
                    "processed_at": time.time()
                }
            }

        # Extract embedding from the best face
        embedding = face_detector.extract_embedding(image)

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "status": "SUCCESS",
            "faces": [{
                "face_id": 0,
                "detection_confidence": faces[0]['detection_confidence'],
                "bounding_box": faces[0]['bounding_box'],
                "embedding": embedding
            }],
            "metadata": {
                "processing_time_ms": processing_time,
                "model_version": "InsightFace" if INSIGHTFACE_AVAILABLE else "OpenCV",
                "processed_at": time.time()
            }
        }

    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")

@app.post("/compare-faces")
async def compare_faces(
    embedding1: List[float],
    embedding2: List[float],
    metric: str = "cosine"
):
    """Compare two face embeddings"""
    start_time = time.time()

    try:
        similarity = face_detector.compare_faces(embedding1, embedding2, metric)

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "similarity_score": similarity,
            "distance": 1.0 - similarity,  # Approximate distance
            "confidence": 0.9 if similarity > FaceConfig.SIMILARITY_THRESHOLD else 0.5,
            "metadata": {
                "processing_time_ms": processing_time,
                "metric_used": metric,
                "processed_at": time.time()
            }
        }

    except Exception as e:
        logger.error(f"Face comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face comparison failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "face",
        "insightface_available": INSIGHTFACE_AVAILABLE
    }

if __name__ == "__main__":
    if not INSIGHTFACE_AVAILABLE:
        logger.warning("InsightFace not available. Install with: pip install insightface")
        logger.warning("Using basic OpenCV face detection as fallback")

    logger.info("Starting Minak Enta Face Recognition Service on port 5002")
    uvicorn.run(app, host="0.0.0.0", port=5002)