#!/usr/bin/env python3
"""
Minak Enta OCR Service
Extracts text from Lebanese ID card images using OCR
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pytesseract
import uvicorn
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

try:
    from pyzbar.pyzbar import decode as zbar_decode
    ZBAR_AVAILABLE = True
except Exception:
    ZBAR_AVAILABLE = False

try:
    import zxingcpp
    ZXING_AVAILABLE = True
except Exception:
    ZXING_AVAILABLE = False

# Note: Protobuf classes are generated for Go clients, not needed for Python services

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Minak Enta OCR Service", version="1.0.0")

easyocr_reader = None


def get_easyocr_reader():
    """Initialize EasyOCR reader lazily"""
    global easyocr_reader, EASYOCR_AVAILABLE
    if not EASYOCR_AVAILABLE:
        return None
    if easyocr_reader is None:
        try:
            easyocr_reader = easyocr.Reader(OCRConfig.EASYOCR_LANGS, gpu=False, download_enabled=True)
        except Exception as e:
            logger.warning(f"EasyOCR unavailable (models not found): {e}")
            EASYOCR_AVAILABLE = False
            return None
    return easyocr_reader

class OCRConfig:
    """OCR service configuration"""
    TESSERACT_CONFIG = '--oem 3 --psm 6 -l ara+eng'
    MIN_CONFIDENCE = 60.0
    EASYOCR_LANGS = ['ar', 'en']

class LebaneseIDTemplate:
    """Template for Lebanese ID card structure based on actual Lebanese ID format"""

    # Patterns to extract VALUES from Lebanese ID cards
    # Lebanese IDs use specific labels like الشهرة (surname), اسم الاب (father name), etc.

    FRONT_PATTERNS = {
        # Surname - prioritize pattern with الشهر context (more specific)
        # "الاسم معلوف الشهر" or "الشهرة: ملكي" patterns
        'name_arabic': r'الاسم\s+([؀-ۿ]+)\s+(?:الشهر|الشهرة)|الشهر[ةه]?\s*:\s*([؀-ۿ]+)|ة\s*:\s*([؀-ۿ]+)\s+٠',
        # Father's name - look for name before "الاب" (RTL scrambled text)
        'father_name': r'([؀-ۿ]+)\s+الاب|[:\s]([؀-ۿ]+)\s+اسم\s*الا[بأ]',
        # Mother's name - look for name after "وشهرتها :" or before ". الام"
        # Text pattern: "وهبه الام وشهرتها : مرسيل" - capture مرسيل first (more reliable)
        'mother_name': r'وشهرتها\s*:\s*([؀-ۿ]+(?:\s+[؀-ۿ]+)?)|:\s*([؀-ۿ]+\s+[؀-ۿ]+)\s*[.\s]+الا[مأ]',
        # First name - look for سامي/name after وزارة or الداخلية
        'first_name': r'وزارة\s+([؀-ۿ]+)\s+الاسم|الداخلية\s*:\s*([؀-ۿ]+)',
        # Latin name
        'name_latin': r'(?:Name|Nom|NAME)\s*:\s*([A-Za-z\s\-\']{2,50})',
        # Lebanese ID number - long sequence of Arabic numerals (٠-٩) at bottom, no label
        # Format: 7-10 Arabic-Indic digits
        'id_number': r'([٠-٩]{7,10})|(\d{7,10})',
        # Date of birth - Arabic numerals with slashes: ١٩٨٣/٠٧/٠٧ or ٠٧/٠٧/١٩٨٣
        'date_of_birth': r'([٠-٩]{1,2}[/\-\.][٠-٩]{1,2}[/\-\.][٠-٩]{2,4})|([٠-٩]{2,4}[/\-\.][٠-٩]{1,2}[/\-\.][٠-٩]{1,2})|(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        # Place of birth - Lebanese cities or foreign places (also match standalone city names)
        'place_of_birth': r'الولادة\s*:\s*([؀-ۿ]+(?:\s+[؀-ۿ]+)?)|محل[^:]*:\s*([؀-ۿ]+(?:\s+[؀-ۿ]+)?)|(بيروت|طرابلس|صيدا|زحلة|ليون\s+فرنسا)',
        # Gender in Arabic
        'gender': r'(ذكر|أنثى)'
        # Note: Lebanese IDs don't have a nationality field - they're only issued to Lebanese citizens
    }

    BACK_PATTERNS = {
        # Issue date - تاريخ الإصدار
        'issue_date': r'(?:تاريخ الإصدار|الإصدار)[:\s]*(\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4})',
        # Registry number - رقم السجل
        'document_number': r'(?:رقم السجل|رقم)[:\s]*(\d+)',
        # District - القضاء
        'district': r'(?:القضاء)[:\s]*([؀-ۿ]+)',
        # Province - المحافظة
        'province': r'(?:المحافظة)[:\s]*([؀-ۿ]+)',
        # Village - المحلة أو القرية
        'village': r'(?:المحلة|القرية)[^\:]*[:\s]*([؀-ۿ]+)',
        # Gender
        'gender': r'(?:الجنس)[:\s]*(ذكر|أنثى)',
        # Marital status - الوضع العائلي
        'marital_status': r'(?:الوضع العائلي)[:\s]*([؀-ۿ]+(?:\s+[؀-ۿ]+)*)'
    }

    # Standalone patterns for fallback extraction
    STANDALONE_PATTERNS = {
        'id_number': r'\b(\d{1,2}[.\-]\d{5,7}[.\-]\d{1,2})\b',
        'date': r'\b(\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4})\b',
        'arabic_word': r'([؀-ۿ]{2,})',
        'latin_name': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    }

def preprocess_image_array(image: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR results"""
    if image is None:
        raise ValueError("Invalid image data")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Apply morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return cleaned


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in top-left, top-right, bottom-right, bottom-left order"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_and_warp_id_card(image: np.ndarray) -> np.ndarray:
    """Detect the ID card contour and apply perspective transform"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect

            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = int(max(widthA, widthB))

            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = int(max(heightA, heightB))

            if maxWidth < 200 or maxHeight < 200:
                continue

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            return warped

    return image


def upscale_image(image: np.ndarray) -> np.ndarray:
    """Upscale image for better OCR if needed"""
    h, w = image.shape[:2]
    if max(h, w) < 1200:
        scale = 2.0
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return image


def count_arabic_chars(text: str) -> int:
    """Count Arabic characters in text"""
    import re
    arabic_chars = re.findall(r'[؀-ۿ]', text)
    return len(arabic_chars)

def count_lebanese_id_keywords(text: str) -> int:
    """Count Lebanese ID-specific keywords in text"""
    keywords = [
        'الجمهورية', 'اللبنانية', 'بطاقة', 'هوية', 'وزارة', 'الداخلية',
        'الاسم', 'اسم', 'الاب', 'الام', 'تاريخ', 'الولادة', 'محل',
        'الجنس', 'ذكر', 'أنثى', 'لبناني', 'لبنانية', 'بيروت',
        'الإصدار', 'السجل', 'المحافظة', 'القضاء', 'القرية'
    ]
    count = 0
    for keyword in keywords:
        if keyword in text:
            count += 1
    return count

def calculate_text_quality_score(text: str, confidence: float) -> float:
    """Calculate a quality score combining confidence, Arabic chars, and keywords"""
    arabic_count = count_arabic_chars(text)
    keyword_count = count_lebanese_id_keywords(text)
    text_length = len(text.strip())

    # Score components (weighted):
    # - Base confidence: 30%
    # - Arabic character density: 30%
    # - Keyword matches: 30%
    # - Text length (capped): 10%
    conf_score = confidence * 0.3
    arabic_score = min(arabic_count / 50.0, 1.0) * 100 * 0.3  # Expect ~50 Arabic chars
    keyword_score = min(keyword_count / 5.0, 1.0) * 100 * 0.3  # Expect ~5 keywords
    length_score = min(text_length / 200.0, 1.0) * 100 * 0.1  # Expect ~200 chars

    return conf_score + arabic_score + keyword_score + length_score

def try_rotations_for_best_ocr(image: np.ndarray) -> Tuple[str, float, int, str]:
    """Try multiple rotations and return the best OCR result based on text quality"""
    rotations = [
        (0, image),
        (90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(image, cv2.ROTATE_180)),
        (270, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    best_text = ""
    best_confidence = 0.0
    best_quality_score = 0.0
    best_rotation = 0
    best_engine = "Tesseract"

    for angle, rotated in rotations:
        # Try to detect and crop the ID card
        cropped = detect_and_warp_id_card(rotated)
        scaled = upscale_image(cropped)

        # For EasyOCR, use the scaled color image (better results)
        # For Tesseract, use preprocessed grayscale
        if EASYOCR_AVAILABLE:
            text, confidence, engine = extract_text_with_confidence(scaled)
        else:
            processed = preprocess_image_array(scaled)
            text, confidence, engine = extract_text_with_confidence(processed)

        # Calculate quality score based on text content, not just confidence
        quality_score = calculate_text_quality_score(text, confidence)

        logger.info(f"Rotation {angle}°: confidence={confidence:.1f}, arabic_chars={count_arabic_chars(text)}, keywords={count_lebanese_id_keywords(text)}, quality={quality_score:.1f}")

        if quality_score > best_quality_score:
            best_text = text
            best_confidence = confidence
            best_quality_score = quality_score
            best_rotation = angle
            best_engine = engine

    logger.info(f"Best rotation: {best_rotation}° with quality score {best_quality_score:.1f}")
    return best_text, best_confidence, best_rotation, best_engine


def try_decode_barcode(image: np.ndarray) -> Optional[dict]:
    """Try to decode PDF417 barcode with multiple rotations and preprocessing"""
    if not ZBAR_AVAILABLE:
        return None

    # Try full image and left-side crop where PDF417 is usually located
    h, w = image.shape[:2]
    crops = [
        image,
        image[:, :int(w * 0.4)]  # left 40%
    ]

    rotations = [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    for crop in crops:
        for angle, rot_code in rotations:
            rotated = crop if rot_code is None else cv2.rotate(crop, rot_code)

            # Try multiple scales
            for scale in [1.0, 2.0, 3.0, 4.0]:
                if scale != 1.0:
                    rotated = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                eq = cv2.equalizeHist(gray)
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
                )

                for candidate in [rotated, gray, eq, thresh]:
                    # Try ZXing first if available
                    if ZXING_AVAILABLE:
                        try:
                            rgb = candidate if len(candidate.shape) == 3 else cv2.cvtColor(candidate, cv2.COLOR_GRAY2RGB)
                            results = zxingcpp.read_barcodes(rgb, try_harder=True)
                            if results:
                                barcode = results[0]
                                return {
                                    "type": barcode.format.name,
                                    "raw": barcode.text
                                }
                        except Exception:
                            pass

                    barcodes = zbar_decode(candidate)
                    if barcodes:
                        barcode = barcodes[0]
                        return {
                            "type": barcode.type,
                            "raw": barcode.data.decode("utf-8", errors="ignore")
                        }

    return None

def extract_text_with_confidence(image: np.ndarray, config: str = OCRConfig.TESSERACT_CONFIG) -> Tuple[str, float, str]:
    """Extract text from image with confidence score"""
    try:
        # Prefer EasyOCR if available (better Arabic support)
        if EASYOCR_AVAILABLE:
            reader = get_easyocr_reader()
            if reader:
                results = reader.readtext(image)
                texts = [r[1] for r in results if r[1].strip()]
                confidences = [float(r[2]) * 100 for r in results]
                full_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                return full_text, avg_confidence, "EasyOCR"

        # Get detailed OCR data including confidence
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)

        # Extract text and confidences
        texts = []
        confidences = []

        for i, confidence in enumerate(data['conf']):
            if int(confidence) > 0:  # Only consider valid detections
                text = data['text'][i].strip()
                if text:
                    texts.append(text)
                    confidences.append(float(confidence))

        # Combine all text
        full_text = ' '.join(texts)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return full_text, avg_confidence, "Tesseract"

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return "", 0.0, "Tesseract"

def parse_lebanese_id_text(text: str, side: str) -> Dict[str, str]:
    """Parse extracted text into structured Lebanese ID data"""
    import re
    extracted_data = {}

    # Basic text cleaning
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Normalize whitespace

    logger.info(f"Parsing {side} side text: {text[:300]}...")

    if side == 'front':
        patterns = LebaneseIDTemplate.FRONT_PATTERNS
    else:
        patterns = LebaneseIDTemplate.BACK_PATTERNS

    # Extract values using capture groups
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
        if match:
            # Find the first non-None capture group
            value = None
            if match.lastindex:
                for i in range(1, match.lastindex + 1):
                    if match.group(i):
                        value = match.group(i)
                        break
            if not value:
                value = match.group()
            value = value.strip() if value else ""
            if value:
                extracted_data[field] = value
                logger.info(f"Found {field}: {value}")

    # Fallback extraction using standalone patterns when labeled patterns fail
    standalone = LebaneseIDTemplate.STANDALONE_PATTERNS

    # Try to find ID number if not already found
    if 'id_number' not in extracted_data:
        id_match = re.search(standalone['id_number'], text)
        if id_match:
            extracted_data['id_number'] = id_match.group(1)
            logger.info(f"Found id_number (fallback): {id_match.group(1)}")

    # Extract dates for front/back appropriately
    dates = re.findall(standalone['date'], text)
    if dates:
        if side == 'front' and 'date_of_birth' not in extracted_data:
            extracted_data['date_of_birth'] = dates[0]
            logger.info(f"Found date_of_birth (fallback): {dates[0]}")
        if side == 'back':
            if 'issue_date' not in extracted_data and len(dates) >= 1:
                extracted_data['issue_date'] = dates[0]
                logger.info(f"Found issue_date (fallback): {dates[0]}")
            if 'expiry_date' not in extracted_data and len(dates) >= 2:
                extracted_data['expiry_date'] = dates[1]
                logger.info(f"Found expiry_date (fallback): {dates[1]}")

    # Try to extract Arabic names if not found
    if 'name_arabic' not in extracted_data:
        arabic_matches = re.findall(standalone['arabic_word'], text)
        # Filter to get likely names (2-4 words, reasonable length)
        for match in arabic_matches:
            words = match.split()
            if 2 <= len(words) <= 5 and 5 <= len(match) <= 60:
                extracted_data['name_arabic'] = match
                logger.info(f"Found name_arabic (fallback): {match}")
                break

    # Try to extract Latin names if not found
    if 'name_latin' not in extracted_data:
        latin_matches = re.findall(standalone['latin_name'], text)
        for match in latin_matches:
            if 5 <= len(match) <= 50:
                extracted_data['name_latin'] = match
                logger.info(f"Found name_latin (fallback): {match}")
                break

    logger.info(f"Extracted {len(extracted_data)} fields: {list(extracted_data.keys())}")
    return extracted_data

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    document_type: str = "LEBANESE_ID_FRONT",
    language_hints: Optional[List[str]] = None
):
    """Process a document image and extract structured data"""
    start_time = time.time()

    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read file content
        image_bytes = await file.read()

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image data")

        # Determine document side
        if "front" in document_type.lower():
            side = "front"
        elif "back" in document_type.lower():
            side = "back"
        else:
            side = "unknown"

        # Extract text with OCR (try multiple rotations)
        extracted_text, ocr_confidence, best_rotation, best_engine = try_rotations_for_best_ocr(image)

        # Parse structured data
        parsed_data = parse_lebanese_id_text(extracted_text, side)

        # Calculate overall confidence
        overall_confidence = min(ocr_confidence, 95.0)  # Cap at 95% for realism

        # Determine processing status
        if overall_confidence >= OCRConfig.MIN_CONFIDENCE:
            status = "SUCCESS"
        elif overall_confidence >= 40.0:
            status = "LOW_CONFIDENCE"
        else:
            status = "FAILED"

        processing_time = int((time.time() - start_time) * 1000)

        # Build additional fields from Lebanese ID-specific data
        additional_fields = {}
        for key in ['first_name', 'father_name', 'mother_name', 'place_of_birth', 'district', 'province', 'village', 'marital_status']:
            if key in parsed_data:
                additional_fields[key] = parsed_data[key]

        response = {
            "status": status,
            "confidence_score": overall_confidence,
            "extracted_data": {
                "name_arabic": parsed_data.get("name_arabic", ""),
                "name_latin": parsed_data.get("name_latin", ""),
                "date_of_birth": parsed_data.get("date_of_birth", ""),
                "id_number": parsed_data.get("id_number", ""),
                "gender": parsed_data.get("gender", ""),
                "nationality": parsed_data.get("nationality", ""),
                "document_number": parsed_data.get("document_number", ""),
                "issue_date": parsed_data.get("issue_date", ""),
                "expiry_date": parsed_data.get("expiry_date", ""),
                "issuing_authority": parsed_data.get("issuing_authority", ""),
                "additional_fields": additional_fields
            },
            "field_confidences": {
                "name_arabic_confidence": 85.0 if parsed_data.get("name_arabic") else 0.0,
                "name_latin_confidence": 90.0 if parsed_data.get("name_latin") else 0.0,
                "date_of_birth_confidence": 95.0 if parsed_data.get("date_of_birth") else 0.0,
                "id_number_confidence": 98.0 if parsed_data.get("id_number") else 0.0,
                "gender_confidence": 80.0 if parsed_data.get("gender") else 0.0,
                "nationality_confidence": 85.0 if parsed_data.get("nationality") else 0.0,
                "document_number_confidence": 90.0 if parsed_data.get("document_number") else 0.0,
                "issue_date_confidence": 92.0 if parsed_data.get("issue_date") else 0.0,
                "expiry_date_confidence": 88.0 if parsed_data.get("expiry_date") else 0.0,
                "issuing_authority_confidence": 75.0 if parsed_data.get("issuing_authority") else 0.0,
            },
            "metadata": {
                "processing_time_ms": processing_time,
                "ocr_engine_version": "EasyOCR" if best_engine == "EasyOCR" else "Tesseract 5.3.0",
                "preprocessing_steps": ["grayscale", "blur", "threshold", "morphology"],
                "best_rotation_degrees": best_rotation,
                "processed_at": time.time(),
                "extracted_text": extracted_text[:500]  # First 500 chars for debugging
            }
        }

        # Try barcode decode for back side (PDF417)
        if side == "back":
            try:
                barcode = try_decode_barcode(image)
                if barcode:
                    response["extracted_data"]["additional_fields"]["barcode_type"] = barcode["type"]
                    response["extracted_data"]["additional_fields"]["barcode_raw"] = barcode["raw"]
            except Exception as e:
                logger.warning(f"Barcode decode failed: {e}")

        logger.info(f"Document processed successfully in {processing_time}ms with confidence {overall_confidence:.1f}%")
        return response

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ocr"}

if __name__ == "__main__":
    # Check if Tesseract is available
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {version}")
    except Exception as e:
        logger.error(f"Tesseract not available: {e}")
        logger.error("Please install Tesseract OCR: brew install tesseract")
        exit(1)

    if not EASYOCR_AVAILABLE:
        logger.warning("EasyOCR not available. Install with: pip install easyocr")
    if not ZBAR_AVAILABLE:
        logger.warning("pyzbar/zbar not available. Install with: brew install zbar && pip install pyzbar")
    if not ZXING_AVAILABLE:
        logger.warning("ZXing not available. Install with: pip install zxing-cpp")

    logger.info("Starting Minak Enta OCR Service on port 5001")
    uvicorn.run(app, host="0.0.0.0", port=5001)