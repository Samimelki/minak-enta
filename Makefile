# Minak Enta - Makefile
# Commands for building, running, and managing the application

.PHONY: help build up down logs clean dev test lint

# Default target
help:
	@echo "Minak Enta - Available Commands:"
	@echo ""
	@echo "  Docker Commands:"
	@echo "    make build      - Build all Docker images"
	@echo "    make up         - Start all services (docker-compose up -d)"
	@echo "    make down       - Stop all services"
	@echo "    make logs       - View logs from all services"
	@echo "    make logs-api   - View API server logs"
	@echo "    make logs-ocr   - View OCR service logs"
	@echo "    make logs-face  - View Face service logs"
	@echo "    make logs-live  - View Liveness service logs"
	@echo "    make restart    - Restart all services"
	@echo "    make clean      - Stop services and remove volumes"
	@echo ""
	@echo "  Development Commands:"
	@echo "    make dev        - Run services locally (without Docker)"
	@echo "    make test       - Run all tests"
	@echo "    make lint       - Run linters (Go + Python)"
	@echo ""
	@echo "  Database Commands:"
	@echo "    make db-shell   - Open PostgreSQL shell"
	@echo "    make db-reset   - Reset database (destructive!)"

# ============================================================
# Docker Commands
# ============================================================

# Build all Docker images
build:
	docker-compose build

# Build without cache
build-nocache:
	docker-compose build --no-cache

# Start all services in detached mode
up:
	docker-compose up -d

# Start all services with build
up-build:
	docker-compose up -d --build

# Stop all services
down:
	docker-compose down

# View logs from all services
logs:
	docker-compose logs -f

# View specific service logs
logs-api:
	docker-compose logs -f api

logs-ocr:
	docker-compose logs -f ocr

logs-face:
	docker-compose logs -f face

logs-live:
	docker-compose logs -f liveness

logs-db:
	docker-compose logs -f postgres

# Restart all services
restart:
	docker-compose restart

# Stop services and remove volumes (destructive)
clean:
	docker-compose down -v --remove-orphans

# Show status of all services
status:
	docker-compose ps

# ============================================================
# Development Commands (without Docker)
# ============================================================

# Run services locally using existing scripts
dev:
	./scripts/run-services.sh

# Run Go server only (assumes ML services running)
run-api:
	go run cmd/server/main.go

# ============================================================
# Testing
# ============================================================

# Run all Go tests
test:
	go test -v ./...

# Run tests with coverage
test-coverage:
	go test -v -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# ============================================================
# Linting
# ============================================================

# Run Go linter
lint-go:
	golangci-lint run ./...

# Run Python linter
lint-python:
	cd ml/ocr && ruff check .
	cd ml/face && ruff check .
	cd ml/liveness && ruff check .

# Run all linters
lint: lint-go lint-python

# Format Go code
fmt:
	gofmt -w .

# ============================================================
# Database Commands
# ============================================================

# Open PostgreSQL shell
db-shell:
	docker-compose exec postgres psql -U minak -d minak_enta

# Reset database (destructive!)
db-reset:
	docker-compose down -v postgres
	docker-compose up -d postgres
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 5
	@echo "Database reset complete"

# ============================================================
# Utility
# ============================================================

# Check service health
health:
	@echo "Checking service health..."
	@curl -s http://localhost:38080/health && echo " - API: OK" || echo " - API: FAILED"
	@curl -s http://localhost:35001/health && echo " - OCR: OK" || echo " - OCR: FAILED"
	@curl -s http://localhost:35002/health && echo " - Face: OK" || echo " - Face: FAILED"
	@curl -s http://localhost:35003/health && echo " - Liveness: OK" || echo " - Liveness: FAILED"

# Pull base images
pull:
	docker pull golang:1.24-alpine
	docker pull python:3.11-slim
	docker pull postgres:16-alpine
