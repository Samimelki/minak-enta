# 2. Hybrid Open-Source Identity Verification

Date: 2026-01-10

## Status

Accepted

## Context

We need to verify Lebanese citizens' identities for civic applications. Options considered:

1. **Third-party vendors** (Onfido, Jumio): $3-7 per verification, foreign data storage, vendor lock-in
2. **Pure government solution**: Requires government cooperation, slow, political
3. **Hybrid open-source**: Build with open-source components, self-hosted, sovereign

## Decision

We will build a hybrid open-source identity verification system using:

- **External API**: OpenAPI/REST for consumer applications
- **Internal services**: gRPC/Protocol Buffers for ML microservices
- **ML Components**: Open-source (InsightFace, Tesseract, passive liveness)
- **Infrastructure**: Self-hosted, data stays in Lebanon

## Consequences

### Positive
- No per-verification costs at scale
- Full data sovereignty
- Auditable by anyone
- No vendor lock-in
- Stronger legitimacy for civic use (voting, petitions)

### Negative
- Higher upfront development cost
- Must maintain ML models ourselves
- Need to tune for Lebanese ID card format
- Responsible for security ourselves

### Risks
- False positives/negatives in verification
- Need pilot organizations to validate before scale
