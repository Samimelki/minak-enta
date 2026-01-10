# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Minak Enta (مينك إنت؟ - "Who are you?") is a sovereign, open-source identity verification service for Lebanon. It verifies Lebanese citizens' identities through ID document scanning, facial recognition, and liveness detection.

## Tech Stack

- **Backend**: Go
- **API**: Protocol Buffers over HTTP
- **Database**: PostgreSQL
- **ML Components**: Python microservices (face recognition, OCR, liveness)
- **Infrastructure**: Docker, Kubernetes

## Project Structure

```
minak-enta/
├── cmd/                    # Application entrypoints
│   └── server/             # Main API server
├── internal/               # Private application code
│   ├── api/                # HTTP handlers and routing
│   ├── verification/       # Core verification logic
│   ├── document/           # ID document processing
│   ├── face/               # Face matching service client
│   ├── liveness/           # Liveness detection service client
│   ├── storage/            # Database and file storage
│   └── audit/              # Audit logging
├── pkg/                    # Public libraries (for consumers)
│   └── client/             # Go client SDK
├── proto/                  # Protocol Buffer definitions
├── ml/                     # Python ML microservices
│   ├── face/               # Face recognition service
│   ├── ocr/                # Document OCR service
│   └── liveness/           # Liveness detection service
├── migrations/             # Database migrations
├── deploy/                 # Kubernetes manifests
└── docs/                   # Documentation
```

## Build & Run Commands

```bash
# Run the API server
go run cmd/server/main.go

# Build the application
go build -o minak-enta ./cmd/server

# Run tests
go test ./...

# Run a single test
go test -run TestName ./path/to/package

# Run ML services (Python)
cd ml/face && python -m uvicorn main:app --reload
cd ml/ocr && python -m uvicorn main:app --reload
cd ml/liveness && python -m uvicorn main:app --reload
```

## Development Guidelines

### Security First

This is identity infrastructure. Security is paramount:
- Never log PII (names, ID numbers, biometrics)
- Always encrypt sensitive data at rest
- Use constant-time comparisons for hashes
- Validate all inputs aggressively
- Audit log all verification attempts

### Privacy by Design

- Delete raw images after processing
- Store only hashes and embeddings, not raw biometrics
- Implement data retention policies
- Support user data deletion requests

### API Design

- All endpoints versioned (`/v1/`, `/v2/`)
- Use Protocol Buffers for request/response
- Return confidence scores, not just pass/fail
- Include request IDs for tracing

### Testing

- Unit tests for all business logic
- Integration tests for verification flows
- Mock ML services in tests
- Test with real Lebanese ID samples (redacted)

## Commit Convention

Use conventional commits: `type(scope): description`

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `build`, `ci`, `chore`

Scopes: `api`, `verification`, `document`, `face`, `liveness`, `storage`, `audit`, `ml`, `deploy`

Examples:
- `feat(verification): add confidence score calculation`
- `fix(face): handle low-light selfie images`
- `docs(api): document authentication endpoint`

## Key Concepts

### Verification Flow

1. Client starts verification session
2. User uploads ID document (front/back)
3. System extracts data via OCR
4. User captures selfie with liveness challenge
5. System matches faces
6. System calculates confidence score
7. Result: verified / rejected / pending-review

### Confidence Score

A 0-100 score based on:
- Document quality and authenticity signals
- OCR extraction confidence
- Face match similarity
- Liveness detection confidence
- Any flags or anomalies

### Consumer Integration

External apps integrate via:
1. OAuth2 flow (redirect user to Minak Enta)
2. Receive verification result via callback
3. Query user status via API

## Related Projects

- **Dawlati** (`github.com/SebastienMelki/dawlati`) - E-gov platform, primary consumer
- **Agora** - Voting platform, future consumer
