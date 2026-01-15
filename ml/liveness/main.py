#!/usr/bin/env python3
"""
Minak Enta Liveness Detection Service
Detects whether a face in an image/video is from a live person
"""

import logging
import time
from typing import List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Minak Enta Liveness Detection Service", version="1.0.0")

class LivenessConfig:
    """Liveness detection configuration"""
    MIN_FACE_SIZE = 100
    BLUR_THRESHOLD = 100  # Lower is more blurry (spoofing indicator)
    BRIGHTNESS_MIN = 50
    BRIGHTNESS_MAX = 200
    CONTRAST_THRESHOLD = 30

class LivenessDetector:
    """Liveness detection handler"""

    def __init__(self):
        # Load Haar cascade for face detection (fallback)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def analyze_image(self, image: np.ndarray) -> dict:
        """Analyze a single image for liveness indicators"""
        results = {
            'face_detected': False,
            'face_bounds': None,
            'spoofing_probability': 0.0,
            'quality_score': 0.0,
            'brightness_score': 0.0,
            'contrast_score': 0.0,
            'sharpness_score': 0.0,
            'attack_types': [],
            'confidence': 0.0
        }

        # Detect face
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(LivenessConfig.MIN_FACE_SIZE, LivenessConfig.MIN_FACE_SIZE)
        )

        if len(faces) == 0:
            results['spoofing_probability'] = 1.0
            results['confidence'] = 0.9
            results['attack_types'].append('no_face_detected')
            return results

        # Use the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face

        results['face_detected'] = True
        results['face_bounds'] = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}

        # Extract face region
        face_roi = gray[y:y+h, x:x+w]

        # Analyze image quality metrics
        results['brightness_score'] = self._analyze_brightness(face_roi)
        results['contrast_score'] = self._analyze_contrast(face_roi)
        results['sharpness_score'] = self._analyze_sharpness(face_roi)

        # Calculate overall quality score
        results['quality_score'] = (
            results['brightness_score'] * 0.3 +
            results['contrast_score'] * 0.3 +
            results['sharpness_score'] * 0.4
        )

        # Analyze for spoofing indicators
        spoofing_indicators = []

        # Check for unnatural image characteristics
        if results['sharpness_score'] < 0.3:
            spoofing_indicators.append('blurry_image')
        if results['brightness_score'] < 0.4 or results['brightness_score'] > 0.9:
            spoofing_indicators.append('unnatural_lighting')
        if results['contrast_score'] < 0.3:
            spoofing_indicators.append('low_contrast')

        # Simple heuristics for common attacks
        if self._detect_print_attack(face_roi):
            spoofing_indicators.append('printed_photo')
        if self._detect_screen_attack(image):
            spoofing_indicators.append('screen_replay')

        results['attack_types'] = spoofing_indicators

        # Calculate spoofing probability
        base_probability = len(spoofing_indicators) * 0.2
        quality_penalty = max(0, 1.0 - results['quality_score']) * 0.3
        results['spoofing_probability'] = min(1.0, base_probability + quality_penalty)

        # Calculate confidence in the assessment
        results['confidence'] = min(0.95, 0.7 + (results['quality_score'] * 0.3))

        return results

    def _analyze_brightness(self, image: np.ndarray) -> float:
        """Analyze image brightness (0-1, higher is better)"""
        brightness = np.mean(image)
        # Normalize to 0-1 scale (assuming 0-255 range)
        normalized = brightness / 255.0
        # Score: prefer mid-range brightness
        if 0.4 <= normalized <= 0.8:
            return 1.0
        elif 0.2 <= normalized <= 0.9:
            return 0.7
        else:
            return 0.3

    def _analyze_contrast(self, image: np.ndarray) -> float:
        """Analyze image contrast (0-1, higher is better)"""
        contrast = np.std(image)
        # Normalize and score
        normalized = min(1.0, contrast / 50.0)
        return normalized

    def _analyze_sharpness(self, image: np.ndarray) -> float:
        """Analyze image sharpness using Laplacian variance (0-1, higher is sharper)"""
        # Convert to float for better precision
        image_float = image.astype(np.float64)

        # Apply Laplacian filter
        laplacian = cv2.Laplacian(image_float, cv2.CV_64F)

        # Calculate variance
        variance = laplacian.var()

        # Normalize (typical range 0-1000+)
        normalized = min(1.0, variance / 500.0)
        return normalized

    def _detect_print_attack(self, face_roi: np.ndarray) -> bool:
        """Detect if image appears to be a printed photo"""
        # Simple heuristics: printed photos often have visible dot patterns
        # or unnatural color transitions

        # Check for periodic patterns (printing artifacts)
        try:
            # Convert to frequency domain
            f = np.fft.fft2(face_roi)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))

            # Look for high-frequency patterns
            rows, cols = face_roi.shape
            crow, ccol = rows // 2, cols // 2

            # Check center region for printing artifacts
            center_region = magnitude_spectrum[crow-10:crow+10, ccol-10:ccol+10]
            high_freq_energy = np.sum(center_region > np.mean(magnitude_spectrum) + np.std(magnitude_spectrum))

            return high_freq_energy > 50  # Threshold for detection
        except:
            return False

    def _detect_screen_attack(self, image: np.ndarray) -> bool:
        """Detect if image appears to be from a screen/digital display"""
        # Look for RGB color channel correlations typical of digital displays
        try:
            b, g, r = cv2.split(image)

            # Calculate correlations between channels
            r_b_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
            r_g_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
            b_g_corr = np.corrcoef(b.flatten(), g.flatten())[0, 1]

            # Digital displays often have high correlations between channels
            avg_corr = (abs(r_b_corr) + abs(r_g_corr) + abs(b_g_corr)) / 3

            return avg_corr > 0.8  # High correlation indicates screen capture
        except:
            return False

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for liveness analysis"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image data")

    return image

@app.post("/detect-liveness")
async def detect_liveness(
    file: UploadFile = File(...)
):
    """Analyze image for liveness indicators"""
    start_time = time.time()

    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        detector = LivenessDetector()
        analysis = detector.analyze_image(image)

        processing_time = int((time.time() - start_time) * 1000)

        # Calculate liveness score (inverse of spoofing probability)
        liveness_score = 1.0 - analysis['spoofing_probability']

        return {
            "status": "SUCCESS",
            "liveness_score": liveness_score,
            "confidence": analysis['confidence'],
            "analysis": {
                "face_detection": {
                    "face_detected": analysis['face_detected'],
                    "face_bounds": analysis['face_bounds']
                },
                "spoofing_detection": {
                    "spoofing_probability": analysis['spoofing_probability'],
                    "detected_methods": analysis['attack_types']
                },
                "quality_assessment": {
                    "overall_quality": analysis['quality_score'],
                    "brightness_score": analysis['brightness_score'],
                    "contrast_score": analysis['contrast_score'],
                    "sharpness_score": analysis['sharpness_score']
                }
            },
            "metadata": {
                "processing_time_ms": processing_time,
                "detection_method": "computer_vision_heuristics",
                "processed_at": time.time()
            }
        }

    except Exception as e:
        logger.error(f"Liveness detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Liveness detection failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "liveness"}

if __name__ == "__main__":
    logger.info("Starting Minak Enta Liveness Detection Service on port 5003")
    uvicorn.run(app, host="0.0.0.0", port=5003)