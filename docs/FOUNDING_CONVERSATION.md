# Minak Enta - Founding Conversation Summary

**Date**: January 10, 2026
**Participants**: Sami Melki, Claude (Opus 4.5), GPT-5.2
**Context**: Brainstorming session that led to the creation of this project

---

## The Origin: Dawlati

This project emerged from discussions about **Dawlati** (دولتي), a planned e-government platform for Lebanon created by Sebastien Melki (VP Engineering at Anghami). The question arose: where should we start?

The answer: **Authentication and identity verification** - because everything else depends on it.

---

## The Problem

### Lebanon's Digital Identity Gap

1. **Lebanon has biometric ID cards** - with RFID chips storing face, fingerprints, palmprints
2. **But no infrastructure to use them** - can't authenticate online, no digital signatures in practice
3. **Government initiatives exist on paper** (LENDID, Super App, 2020-2030 Digital Transformation Strategy) but nothing functional on the ground
4. **Reality check from Sami** (who lives in Lebanon): "the new ID is just an id with biometric and can't be used for any payment or signature as of yet"

### The Vendor Problem

Services like Onfido and Jumio charge $3-7 per verification. This is:
- **Fine for banks** - one verified user generates thousands in revenue
- **Poisonous for civic projects** - at 1M users, that's $5M+ upfront

More critically, foreign vendors create:
- Vendor lock-in
- Data sovereignty issues
- Political trust problems (especially for voting)
- Dependency without control

---

## The Solution: Minak Enta

**"Minak Enta"** (مينك إنت؟) means "Who are you?" in Lebanese dialect.

A **sovereign, open-source identity verification service** that:
- Verifies Lebanese citizens' identities
- Stores no raw biometrics
- Is fully auditable
- Can be consumed by any application

---

## Architecture Decision: Three Separate Projects

```
┌─────────────────────────────────────────────────────────────┐
│                      Minak Enta                              │
│            (Identity Layer - Sovereign, Open)                │
│                                                              │
│  "Who is this person? Are they Lebanese? Are they unique?"   │
└─────────────────┬───────────────────────┬───────────────────┘
                  │                       │
                  ▼                       ▼
    ┌─────────────────────┐   ┌─────────────────────┐
    │      Dawlati        │   │       Agora         │
    │ github.com/         │   │                     │
    │ SebastienMelki/     │   │  Sami's voting      │
    │ dawlati             │   │  platform           │
    │                     │   │                     │
    │  E-gov services     │   │  Online voting      │
    │  (like UAE's TAMM)  │   │  (needs verified    │
    │                     │   │   unique persons)   │
    └─────────────────────┘   └─────────────────────┘
```

**Minak Enta** = Identity backbone (this repo)
**Dawlati** = E-government platform (Sebastien's repo)
**Agora** = Voting platform (Sami's future project)

---

## Strategic Insights from GPT-5.2

### Position 1: In Favor of Building Hybrid System

> "Lebanon already issued identity hardware without identity capability. That is a historical gift. Rare, accidental, and usually wasted. You can build around it."

Key points:
- **Avoid vendor capture** - once identity flows through a foreign black box, you never get it back
- **Tech is no longer the hard part** - face matching is commoditized, OCR is good enough, passive liveness exists
- **Stage legitimacy** - not building "national digital ID", but "a progressively trusted identity layer for civic participation"

### What You're Actually Building

Not face recognition. You're building:
- An **identity confidence score**
- A **verifiable audit trail**
- A **trust ladder**

That ladder later supports: petitions, consultative votes, municipal pilots, diaspora participation, eventually binding elections.

### Position 2: The Counter-Argument

> "Identity is the most dangerous thing to get 90% right."

Risks:
- False positives undermine elections
- False negatives disenfranchise citizens
- One scandal kills trust for a decade
- In Lebanon, identity is sectarian, legal, historical, and explosive

**Alternative for voting**: Anonymous civic tokens (blind signatures, ZK proofs, one-person-one-token) instead of persistent identity.

### The Verdict

> "Your instinct to go hybrid is correct only if you treat identity as: A necessary evil to be constrained, not a feature to be celebrated."

---

## Technical Approach: Hybrid Open-Source

### Why Not Pure Third-Party?

| Factor | Third-Party (Onfido/Jumio) | Hybrid Open-Source |
|--------|---------------------------|-------------------|
| Cost | $5/check forever | One-time build cost |
| Data sovereignty | Data leaves Lebanon | Data stays local |
| Audit | Black box | Fully auditable |
| Vendor lock-in | High | None |
| Legitimacy for voting | Questionable | Stronger |

### Open-Source Components Available

**Document OCR:**
- PassportEye (Python) - MRZ extraction, ~80% accuracy
- FastMRZ (Python) - MRZ extraction
- Tesseract + OpenCV - DIY approach

**Face Recognition:**
- InsightFace - very popular, high accuracy
- DeepFace (Python) - wraps multiple models
- CVARTEL - open-source, KYC-focused

**Liveness Detection:**
- KBY-AI - 3D passive liveness, detects printed photos, masks, video replay
- MiniAiLive SDK - iBeta 2 certified, fully on-premise

---

## MVP Verification Flow

```
INPUT:
├── Lebanese ID card (front photo)
├── Lebanese ID card (back photo)
└── Selfie (with liveness check)

PROCESS:
├── Extract data from ID (name, DOB, ID number)
├── Extract face from ID photo
├── Verify selfie is live (not a photo of a photo)
├── Match selfie face to ID face
└── Calculate confidence score

OUTPUT:
├── Verified: yes/no/pending-review
├── Confidence score (0-100)
├── Unique person ID (hash-based)
└── Verification timestamp

STORAGE (privacy-first):
├── NO raw biometrics
├── Face embedding hash (for duplicate detection)
├── ID data hash (for re-verification)
└── Audit log
```

---

## Pilot Strategy

**Target organizations** (hundreds to thousands of users):
- Schools / Universities (student elections)
- Large hospitals (patient/staff verification)
- Corporations (employee verification)
- Small municipalities (willing to experiment)
- NGOs

**Why this approach:**
- Contained user bases
- Real problems to solve
- Can say "yes" without government approval
- Builds proof points for later

---

## Phased Roadmap (from GPT-5.2)

### Phase 0: Legal + Ethical Guardrails (before code)
- Explicit scope: not a government ID
- Explicit non-scope: not binding elections
- Data minimization doctrine
- Public threat model document

### Phase 1: Identity Proof MVP
Goal: "This is likely a real Lebanese adult, once."
- ID upload, selfie, liveness, face match
- Manual review fallback
- Store hashes + confidence score

### Phase 2: Trust Accumulation
Goal: Move from "verified once" to "known over time"
- Device reputation
- Behavioral consistency
- Re-verification triggers
- Diaspora verification flow

### Phase 3: Civic Pilots
Goal: Legitimacy through use, not decree
- University elections, professional orders, syndicates, municipal consultations, NGOs

### Phase 4: Government Interface
- Either the state ignores you, or it knocks
- Either way, you negotiate from strength

---

## Project Structure

```
minak-enta/
├── cmd/server/             # API server entrypoint
├── internal/
│   ├── api/                # HTTP handlers
│   ├── verification/       # Core verification logic
│   ├── document/           # ID document processing
│   ├── face/               # Face matching client
│   ├── liveness/           # Liveness detection client
│   ├── storage/            # Database layer
│   └── audit/              # Audit logging
├── pkg/client/             # Go SDK for consumers
├── proto/                  # Protocol Buffer definitions
├── ml/
│   ├── face/               # Face recognition (Python)
│   ├── ocr/                # Document OCR (Python)
│   └── liveness/           # Liveness detection (Python)
├── migrations/             # Database migrations
├── deploy/                 # Kubernetes manifests
└── docs/                   # Documentation
```

---

## Key Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Separate repo from Dawlati | Yes | Clean boundaries, reusable by multiple consumers |
| Name | "Minak Enta" | Lebanese dialect, memorable, literally describes function |
| Repo owner | Samimelki | Sami's project, Sebastien consumes via Dawlati |
| Tech stack | Go + Python ML services | Go for API performance, Python for ML ecosystem |
| API format | Protocol Buffers over HTTP | Efficient, typed, multilingual support |
| Database | PostgreSQL | Reliable, good for complex queries |
| Build vs Buy | Hybrid open-source | Sovereignty, cost, auditability |

---

## Open Questions for Future Sessions

1. **Lebanese ID card format** - Need sample images to tune OCR. What fields are on front vs back?
2. **Diaspora flow** - How to verify Lebanese abroad? Different ID types?
3. **Re-verification triggers** - When should users re-verify? Suspicious activity? Time-based?
4. **Consumer auth flow** - OAuth2? OIDC? Custom?
5. **Anonymous tokens for Agora** - ZK proofs? Blind signatures? How to derive anonymous voting tokens from verified identity?

---

## People

- **Sami Melki** - Creator of Minak Enta and Agora
- **Sebastien Melki** - Creator of Dawlati, VP Engineering at Anghami, Sami's brother

---

## Next Steps

1. Define Protocol Buffer API schema
2. Build OCR service for Lebanese ID cards
3. Build face matching service
4. Build liveness detection service
5. Wire together into verification flow
6. Build simple web UI for testing
7. Find first pilot organization

---

*This document summarizes a conversation that led to the founding of Minak Enta. It serves as context for future development sessions.*
