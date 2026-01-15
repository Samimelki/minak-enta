-- Initial database schema for Minak Enta
-- Migration: 20260114000000_initial_schema

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table - stores verified user information
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    is_verified BOOLEAN NOT NULL DEFAULT false,
    verification_level TEXT NOT NULL DEFAULT 'none' CHECK (verification_level IN ('none', 'basic', 'standard', 'high')),
    confidence_score INTEGER NOT NULL DEFAULT 0 CHECK (confidence_score >= 0 AND confidence_score <= 100),
    verified_at TIMESTAMP WITH TIME ZONE,
    last_verification TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Privacy-preserving hashes for duplicate detection
    id_data_hash TEXT UNIQUE, -- Hash of ID data for re-verification
    face_embedding BYTEA,     -- Serialized face embedding vector

    -- Audit fields
    verification_count INTEGER NOT NULL DEFAULT 0,
    last_activity TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Verification sessions table - tracks verification attempts
CREATE TABLE verification_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status TEXT NOT NULL DEFAULT 'pending_document_front' CHECK (status IN ('pending_document_front', 'pending_document_back', 'pending_selfie', 'processing', 'completed', 'failed', 'expired')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    callback_url TEXT,
    metadata JSONB,
    user_id UUID REFERENCES users(user_id)
);

-- Document results table - OCR and processing results for ID documents
CREATE TABLE document_results (
    session_id UUID NOT NULL REFERENCES verification_sessions(session_id) ON DELETE CASCADE,
    side TEXT NOT NULL CHECK (side IN ('front', 'back')),
    status TEXT NOT NULL CHECK (status IN ('accepted', 'rejected', 'needs_retry')),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
    rejection_reason TEXT,
    processed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    image_hash TEXT NOT NULL, -- SHA-256 hash of the image for duplicate detection

    PRIMARY KEY (session_id, side)
);

-- Extracted data table - OCR results from ID documents
CREATE TABLE extracted_data (
    session_id UUID NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('front', 'back')),
    name_arabic TEXT,
    name_latin TEXT,
    date_of_birth TEXT,
    id_number TEXT,
    gender TEXT CHECK (gender IN ('male', 'female')),
    nationality TEXT,

    PRIMARY KEY (session_id, side),
    FOREIGN KEY (session_id, side) REFERENCES document_results(session_id, side) ON DELETE CASCADE
);

-- Selfie results table - liveness and face matching results
CREATE TABLE selfie_results (
    session_id UUID PRIMARY KEY REFERENCES verification_sessions(session_id) ON DELETE CASCADE,
    image_hash TEXT NOT NULL, -- SHA-256 hash of the selfie image
    liveness_score REAL NOT NULL CHECK (liveness_score >= 0 AND liveness_score <= 100),
    face_match_score REAL CHECK (face_match_score >= 0 AND face_match_score <= 100),
    processed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Audit log table - comprehensive audit trail
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id),
    session_id UUID REFERENCES verification_sessions(session_id),
    action TEXT NOT NULL, -- e.g., 'verification_started', 'document_uploaded', 'verification_completed'
    resource TEXT NOT NULL, -- e.g., 'verification_session', 'user', 'document'
    ip_address INET NOT NULL,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    request_id TEXT NOT NULL,
    details JSONB
);

-- Create indexes for performance
CREATE INDEX CONCURRENTLY idx_verification_sessions_status ON verification_sessions(status);
CREATE INDEX CONCURRENTLY idx_verification_sessions_expires_at ON verification_sessions(expires_at);
CREATE INDEX CONCURRENTLY idx_verification_sessions_user_id ON verification_sessions(user_id);

CREATE INDEX CONCURRENTLY idx_document_results_session_id ON document_results(session_id);
CREATE INDEX CONCURRENTLY idx_document_results_image_hash ON document_results(image_hash);

CREATE INDEX CONCURRENTLY idx_extracted_data_id_number ON extracted_data(id_number) WHERE id_number IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_selfie_results_session_id ON selfie_results(session_id);

CREATE INDEX CONCURRENTLY idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX CONCURRENTLY idx_audit_logs_session_id ON audit_logs(session_id);
CREATE INDEX CONCURRENTLY idx_audit_logs_action ON audit_logs(action);
CREATE INDEX CONCURRENTLY idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX CONCURRENTLY idx_audit_logs_request_id ON audit_logs(request_id);

-- Additional performance indexes
CREATE INDEX CONCURRENTLY idx_users_id_data_hash ON users(id_data_hash) WHERE id_data_hash IS NOT NULL;
-- Note: GIN index on face_embedding removed as BYTEA doesn't support GIN by default
-- CREATE INDEX CONCURRENTLY idx_users_face_embedding ON users USING gin (face_embedding) WHERE face_embedding IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_verification_sessions_status_expires_at ON verification_sessions(status, expires_at);
CREATE INDEX CONCURRENTLY idx_audit_logs_composite ON audit_logs(user_id, session_id, action, timestamp);

-- Add comments for documentation
COMMENT ON TABLE users IS 'Verified users with privacy-preserving identity data';
COMMENT ON TABLE verification_sessions IS 'Verification session tracking and state management';
COMMENT ON TABLE document_results IS 'ID document processing results and OCR outcomes';
COMMENT ON TABLE extracted_data IS 'Structured data extracted from ID documents via OCR';
COMMENT ON TABLE selfie_results IS 'Selfie processing results including liveness and face matching';
COMMENT ON TABLE audit_logs IS 'Comprehensive audit trail for all system actions';