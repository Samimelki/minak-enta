# Minak Enta (مينك إنت؟)

**"Who are you?"** - A sovereign, open-source identity verification service for Lebanon.

## What is Minak Enta?

Minak Enta is a digital identity verification layer that allows Lebanese citizens to prove who they are online. It provides a foundation for civic participation, government services, and any application that needs to verify a person's identity.

## Why?

Lebanon has biometric ID cards but no digital infrastructure to use them. Citizens cannot:
- Prove their identity online
- Access government services digitally
- Participate in verified digital voting
- Sign documents electronically

Minak Enta bridges this gap.

## Design Principles

1. **Sovereign by design** - No foreign vendor lock-in. All data stays in Lebanon.
2. **Privacy-first** - Minimal data retention. No raw biometrics stored long-term.
3. **Open source** - Full transparency. Auditable by anyone.
4. **Progressive trust** - Identity confidence grows over time with use.
5. **Civic infrastructure** - Not a product. A public good.

## How It Works

```
User                          Minak Enta                    Consumer App
  │                               │                              │
  │  1. Upload ID (front/back)    │                              │
  │  2. Take selfie               │                              │
  │──────────────────────────────>│                              │
  │                               │                              │
  │                          ┌────┴────┐                         │
  │                          │ Verify: │                         │
  │                          │ - OCR   │                         │
  │                          │ - Face  │                         │
  │                          │ - Live  │                         │
  │                          └────┬────┘                         │
  │                               │                              │
  │  3. Verified! Score: 95       │                              │
  │<──────────────────────────────│                              │
  │                               │                              │
  │                               │  4. Is user X verified?      │
  │                               │<─────────────────────────────│
  │                               │                              │
  │                               │  5. Yes, score 95            │
  │                               │─────────────────────────────>│
```

## Verification Process

1. **Document Capture** - User photographs their Lebanese ID card (front and back)
2. **Selfie with Liveness** - User takes a selfie; system verifies it's a live person
3. **Face Matching** - System matches selfie to ID photo
4. **Data Extraction** - OCR extracts name, date of birth, ID number
5. **Confidence Scoring** - System calculates overall verification confidence
6. **Manual Review** - Edge cases flagged for human review

## What Gets Stored

| Data | Stored? | Purpose |
|------|---------|---------|
| Raw ID photos | No (deleted after processing) | - |
| Raw selfie | No (deleted after processing) | - |
| Face embedding hash | Yes | Duplicate detection |
| ID data hash | Yes | Re-verification |
| Extracted text (name, DOB) | Yes (encrypted) | Identity record |
| Confidence score | Yes | Trust level |
| Verification timestamp | Yes | Audit trail |

## Consumer Applications

Minak Enta is designed to be consumed by other applications:

- **Dawlati** - E-government services platform
- **Agora** - Online voting platform
- **Schools/Universities** - Student verification, elections
- **Hospitals** - Patient identity
- **Corporations** - Employee verification
- **Municipalities** - Resident services
- **NGOs** - Beneficiary verification

## API Overview

```
POST   /v1/verify/start      # Begin verification flow
POST   /v1/verify/document   # Upload ID document
POST   /v1/verify/selfie     # Upload selfie with liveness
GET    /v1/verify/{id}       # Check verification status

POST   /v1/authenticate      # Verify returning user
GET    /v1/user/{id}         # Get user verification status

DELETE /v1/user/{id}         # User requests data deletion
```

## Tech Stack

- **Backend**: Go (main API server)
- **AI/ML Services**: Python microservices
  - **OCR Service**: Tesseract + OpenCV for text extraction
  - **Face Recognition**: InsightFace for biometric verification
  - **Liveness Detection**: Computer vision heuristics
- **API**: Protocol Buffers over HTTP
- **Database**: PostgreSQL
- **Web Interface**: Vanilla HTML/CSS/JavaScript
- **Infrastructure**: Self-hosted / Kubernetes

## Quick Start

### Prerequisites
- Go 1.21+
- Python 3.8+
- PostgreSQL
- Tesseract OCR (`brew install tesseract`)

### Manual Startup (Recommended for Testing)

```bash
# 1. Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r ml/ocr/requirements.txt
pip install -r ml/face/requirements.txt
pip install -r ml/liveness/requirements.txt

# 3. Set up database
createdb minak_enta
psql -d minak_enta -c "CREATE USER minak_enta WITH PASSWORD 'password';"
psql -d minak_enta -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO minak_enta;"

# 4. Run database migrations
psql -d minak_enta -f migrations/20260114000000_initial_schema.sql

# 5. Start services in separate terminals:

# Terminal 1 - OCR Service (Port 5001)
python3 ml/ocr/main.py

# Terminal 2 - Face Service (Port 5002)
python3 ml/face/main.py

# Terminal 3 - Liveness Service (Port 5003)
python3 ml/liveness/main.py

# Terminal 4 - API Server (Port 38081)
go run cmd/server/main.go

# 6. Open web interface
open http://localhost:38081/
```

### Automated Startup (Experimental)

```bash
# Use the manual startup script for step-by-step instructions
./scripts/start-services.sh
```

### Manual Startup

```bash
# Terminal 1: OCR Service (Port 5001)
source venv/bin/activate && python3 ml/ocr/main.py

# Terminal 2: Face Service (Port 5002)
source venv/bin/activate && python3 ml/face/main.py

# Terminal 3: Liveness Service (Port 5003)
source venv/bin/activate && python3 ml/liveness/main.py

# Terminal 4: API Server (Port 38081)
go run cmd/server/main.go
```

### Test the System

```bash
# Health checks
curl http://localhost:5001/health  # OCR
curl http://localhost:5002/health  # Face
curl http://localhost:5003/health  # Liveness
curl http://localhost:38081/health # API

# Test OCR with generated test document
curl -X POST -F "file=@test_document.png" -F "document_type=LEBANESE_ID_FRONT" \
  http://localhost:5001/process-document

# Test Face Detection
curl -X POST -F "file=@test_document.png" http://localhost:5002/detect-faces

# Test Liveness Detection
curl -X POST -F "file=@test_document.png" http://localhost:5003/detect-liveness

# Full verification flow via web UI: http://localhost:38081/
```

## Project Status

✅ **Core implementation complete** - Full AI/ML-powered identity verification system
- Go API server with PostgreSQL
- Python ML microservices (OCR, Face, Liveness)
- Web interface for testing
- Complete verification workflow

## Related Projects

- [Dawlati](https://github.com/SebastienMelki/dawlati) - Lebanese e-government platform
- Agora - Online voting platform (coming soon)

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

This is civic infrastructure. Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
