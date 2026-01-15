#!/bin/bash

# Minak Enta Service Runner
# Starts all ML services and the main API server

set -e

echo "🚀 Starting Minak Enta Services..."

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
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $service_name to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url/health" >/dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        echo "   Attempt $attempt/$max_attempts..."
        sleep 2
        ((attempt++))
    done

    echo "❌ $service_name failed to start"
    return 1
}

# Check if required tools are installed
command -v python3 >/dev/null 2>&1 || { echo "❌ Python3 is required but not installed"; exit 1; }
command -v tesseract >/dev/null 2>&1 || { echo "❌ Tesseract is required but not installed. Install with: brew install tesseract"; exit 1; }

# Check if virtual environment exists (either in root or ml/ocr)
if [ -d "venv" ]; then
    VENV_PATH="venv"
elif [ -d "ml/ocr/venv" ]; then
    VENV_PATH="ml/ocr/venv"
else
    echo "❌ Virtual environment not found. Run: python3 -m venv venv && source venv/bin/activate && pip install -r ml/ocr/requirements.txt -r ml/face/requirements.txt -r ml/liveness/requirements.txt"
    exit 1
fi

# Check ports
check_port 5001 || exit 1  # OCR service
check_port 5002 || exit 1  # Face service
check_port 5003 || exit 1  # Liveness service
check_port 38081 || exit 1 # API server

echo "📦 Checking Python dependencies..."

# Activate virtual environment
source $VENV_PATH/bin/activate

# Check if dependencies are installed
python3 -c "import fastapi, uvicorn, cv2, pytesseract" 2>/dev/null || {
    echo "Installing Python dependencies..."
    pip install -r ml/ocr/requirements.txt
    pip install -r ml/face/requirements.txt
    pip install -r ml/liveness/requirements.txt
}

echo "🧠 Starting ML Services..."

# Start OCR service in background
python3 ml/ocr/main.py &
OCR_PID=$!

# Start Face service in background
python3 ml/face/main.py &
FACE_PID=$!

# Start Liveness service in background
python3 ml/liveness/main.py &
LIVENESS_PID=$!

echo "⏳ Waiting for ML services to start..."

# Wait for services to be ready
wait_for_service "http://localhost:5001" "OCR Service"
wait_for_service "http://localhost:5002" "Face Recognition Service"
wait_for_service "http://localhost:5003" "Liveness Detection Service"

echo "🚀 Starting Main API Server..."

# Start the Go API server
go run cmd/server/main.go &
API_PID=$!

echo "✅ All services started!"
echo ""
echo "🌐 Service URLs:"
echo "   OCR Service:      http://localhost:5001"
echo "   Face Service:     http://localhost:5002"
echo "   Liveness Service: http://localhost:5003"
echo "   API Server:       http://localhost:38081"
echo "   Web Interface:    http://localhost:38081/"
echo ""
echo "📊 Health Checks:"
echo "   curl http://localhost:5001/health"
echo "   curl http://localhost:5002/health"
echo "   curl http://localhost:5003/health"
echo "   curl http://localhost:38081/health"
echo ""
echo "🧪 Test Commands:"
echo "   # Test OCR:"
echo "   curl -X POST -F 'file=@test_image.png' -F 'document_type=LEBANESE_ID_FRONT' http://localhost:5001/process-document"
echo ""
echo "   # Test Face Detection:"
echo "   curl -X POST -F 'file=@test_image.png' http://localhost:5002/detect-faces"
echo ""
echo "   # Test Liveness:"
echo "   curl -X POST -F 'file=@test_image.png' http://localhost:5003/detect-liveness"
echo ""
echo "🛑 To stop all services, press Ctrl+C"

# Wait for interrupt
trap "echo '🛑 Stopping services...'; kill $OCR_PID $FACE_PID $LIVENESS_PID $API_PID 2>/dev/null; exit" INT TERM

wait