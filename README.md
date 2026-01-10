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

- **Backend**: Go
- **API**: Protocol Buffers over HTTP
- **Database**: PostgreSQL
- **Face Recognition**: Open-source (InsightFace/DeepFace)
- **Liveness Detection**: Open-source (KBY-AI or similar)
- **OCR**: Tesseract + custom Lebanese ID templates
- **Infrastructure**: Self-hosted / Kubernetes

## Project Status

Early development - architecture and planning phase.

## Related Projects

- [Dawlati](https://github.com/SebastienMelki/dawlati) - Lebanese e-government platform
- Agora - Online voting platform (coming soon)

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

This is civic infrastructure. Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
