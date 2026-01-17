# Minak Enta - Go API Server
# Multi-stage build for minimal image size

# Build stage
FROM golang:1.24-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git ca-certificates

# Copy go mod files first for layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o minak-enta ./cmd/server

# Runtime stage
FROM alpine:3.19

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata

# Create non-root user for security
RUN adduser -D -g '' appuser

# Copy binary from builder
COPY --from=builder /app/minak-enta .

# Copy migrations for database initialization
COPY --from=builder /app/migrations ./migrations

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 38080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:38080/health || exit 1

# Run the application
CMD ["./minak-enta"]
