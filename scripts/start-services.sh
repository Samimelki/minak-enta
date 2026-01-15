#!/bin/bash

# Minak Enta - Manual Service Startup
# Run services one by one for easier debugging

echo "🚀 Minak Enta - Manual Service Startup"
echo ""

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "❌ Port $1 is already in use"
        return 1
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=10

    echo "⏳ Waiting for $service_name..."

    for i in $(seq 1 $max_attempts); do
        if curl -s "$url/health" >/dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        echo "   Attempt $i/$max_attempts..."
        sleep 1
    done

    echo "❌ $service_name failed to start"
    return 1
}

echo "📋 Prerequisites Check:"
echo "   ✅ Python 3.8+"
echo "   ✅ Tesseract OCR"
echo "   ✅ PostgreSQL (with minak_enta database and user)"
echo ""

echo "🔧 Step 1: Activate Python Virtual Environment"
echo "   Run: source venv/bin/activate"
echo ""

echo "🤖 Step 2: Start ML Services (in separate terminals)"
echo ""

echo "   Terminal 1 - OCR Service:"
echo "   cd /Users/samimelki/minak-enta"
echo "   source venv/bin/activate"
echo "   python3 ml/ocr/main.py"
echo ""

echo "   Terminal 2 - Face Recognition Service:"
echo "   cd /Users/samimelki/minak-enta"
echo "   source venv/bin/activate"
echo "   python3 ml/face/main.py"
echo ""

echo "   Terminal 3 - Liveness Detection Service:"
echo "   cd /Users/samimelki/minak-enta"
echo "   source venv/bin/activate"
echo "   python3 ml/liveness/main.py"
echo ""

echo "🌐 Step 3: Start API Server"
echo "   Terminal 4 - Go API Server:"
echo "   cd /Users/samimelki/minak-enta"
echo "   go run cmd/server/main.go"
echo ""

echo "🔍 Step 4: Test Services"
echo ""
echo "   Health Checks:"
echo "   curl http://localhost:5001/health  # OCR"
echo "   curl http://localhost:5002/health  # Face"
echo "   curl http://localhost:5003/health  # Liveness"
echo "   curl http://localhost:38081/health # API"
echo ""

echo "   Test OCR (create a proper test image first):"
echo "   curl -X POST -F 'file=@real_image.jpg' -F 'document_type=LEBANESE_ID_FRONT' http://localhost:5001/process-document"
echo ""

echo "   Test Face Detection:"
echo "   curl -X POST -F 'file=@photo.jpg' http://localhost:5002/detect-faces"
echo ""

echo "   Test Full Verification:"
echo "   open http://localhost:38081/"
echo ""

echo "🛑 To stop services:"
echo "   pkill -f 'python3 ml'"
echo "   pkill -f 'go run cmd/server/main.go'"
echo ""

echo "📝 Note: Make sure you have proper image files for testing."
echo "   The test_image.png is just a minimal PNG and won't work for OCR/face detection."