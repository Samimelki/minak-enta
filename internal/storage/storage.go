package storage

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"

	"github.com/Samimelki/minak-enta/internal/models"
)

// Storage handles all database operations
type Storage struct {
	db *sql.DB
}

// New creates a new storage instance
func New(db *sql.DB) *Storage {
	return &Storage{db: db}
}

// VerificationSession operations

// CreateVerificationSession creates a new verification session
func (s *Storage) CreateVerificationSession(session *models.VerificationSession) error {
	// Marshal metadata to JSON if present
	var metadataJSON []byte
	var marshalErr error
	if session.Metadata != nil && len(session.Metadata) > 0 {
		metadataJSON, marshalErr = json.Marshal(session.Metadata)
		if marshalErr != nil {
			return fmt.Errorf("failed to marshal metadata: %w", marshalErr)
		}
	} else {
		// For empty metadata, marshal an empty object
		metadataJSON, marshalErr = json.Marshal(map[string]interface{}{})
		if marshalErr != nil {
			return fmt.Errorf("failed to marshal empty metadata: %w", marshalErr)
		}
	}

	query := `
		INSERT INTO verification_sessions (
			session_id, status, expires_at, callback_url, metadata, user_id
		) VALUES ($1, $2, $3, $4, $5, $6)`

	_, err := s.db.Exec(query,
		session.SessionID,
		session.Status,
		session.ExpiresAt,
		session.CallbackURL,
		metadataJSON,
		session.UserID,
	)

	if err != nil {
		log.Printf("Database error creating verification session: %v", err)
		return fmt.Errorf("failed to create verification session: %w", err)
	}

	return nil
}

// GetVerificationSession retrieves a verification session by ID
func (s *Storage) GetVerificationSession(sessionID string) (*models.VerificationSession, error) {
	query := `
		SELECT session_id, status, created_at, expires_at, callback_url, metadata, user_id
		FROM verification_sessions
		WHERE session_id = $1`

	session := &models.VerificationSession{}
	var metadataJSON []byte
	err := s.db.QueryRow(query, sessionID).Scan(
		&session.SessionID,
		&session.Status,
		&session.CreatedAt,
		&session.ExpiresAt,
		&session.CallbackURL,
		&metadataJSON,
		&session.UserID,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("verification session not found: %s", sessionID)
		}
		return nil, fmt.Errorf("failed to get verification session: %w", err)
	}

	// Unmarshal metadata from JSON if present
	if metadataJSON != nil && len(metadataJSON) > 0 {
		err = json.Unmarshal(metadataJSON, &session.Metadata)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
		}
	}

	return session, nil
}

// UpdateVerificationSessionStatus updates the status of a verification session
func (s *Storage) UpdateVerificationSessionStatus(sessionID string, status models.SessionStatus) error {
	query := `UPDATE verification_sessions SET status = $1 WHERE session_id = $2`

	result, err := s.db.Exec(query, status, sessionID)
	if err != nil {
		return fmt.Errorf("failed to update verification session status: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("verification session not found: %s", sessionID)
	}

	return nil
}

// UpdateSessionMetadata updates the metadata of a verification session
func (s *Storage) UpdateSessionMetadata(sessionID string, metadata map[string]any) error {
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `UPDATE verification_sessions SET metadata = $1 WHERE session_id = $2`

	result, err := s.db.Exec(query, metadataJSON, sessionID)
	if err != nil {
		return fmt.Errorf("failed to update session metadata: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("verification session not found: %s", sessionID)
	}

	return nil
}

// User operations

// CreateUser creates a new verified user
func (s *Storage) CreateUser(user *models.User) error {
	query := `
		INSERT INTO users (
			user_id, is_verified, verification_level, confidence_score,
			verified_at, last_verification, id_data_hash, face_embedding
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`

	_, err := s.db.Exec(query,
		user.UserID,
		user.IsVerified,
		user.VerificationLevel,
		user.ConfidenceScore,
		user.VerifiedAt,
		user.LastVerification,
		user.IDDataHash,
		user.FaceEmbedding,
	)

	if err != nil {
		return fmt.Errorf("failed to create user: %w", err)
	}

	return nil
}

// GetUserByID retrieves a user by their ID
func (s *Storage) GetUserByID(userID string) (*models.User, error) {
	query := `
		SELECT user_id, is_verified, verification_level, confidence_score,
		       verified_at, last_verification, created_at, id_data_hash, face_embedding,
		       verification_count, last_activity
		FROM users
		WHERE user_id = $1`

	user := &models.User{}
	err := s.db.QueryRow(query, userID).Scan(
		&user.UserID,
		&user.IsVerified,
		&user.VerificationLevel,
		&user.ConfidenceScore,
		&user.VerifiedAt,
		&user.LastVerification,
		&user.CreatedAt,
		&user.IDDataHash,
		&user.FaceEmbedding,
		&user.VerificationCount,
		&user.LastActivity,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("user not found: %s", userID)
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	return user, nil
}

// FindUserByIDDataHash finds a user by their ID data hash for duplicate detection
func (s *Storage) FindUserByIDDataHash(hash string) (*models.User, error) {
	query := `
		SELECT user_id, is_verified, verification_level, confidence_score,
		       verified_at, last_verification, created_at, id_data_hash, face_embedding,
		       verification_count, last_activity
		FROM users
		WHERE id_data_hash = $1`

	user := &models.User{}
	err := s.db.QueryRow(query, hash).Scan(
		&user.UserID,
		&user.IsVerified,
		&user.VerificationLevel,
		&user.ConfidenceScore,
		&user.VerifiedAt,
		&user.LastVerification,
		&user.CreatedAt,
		&user.IDDataHash,
		&user.FaceEmbedding,
		&user.VerificationCount,
		&user.LastActivity,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil // No duplicate found
		}
		return nil, fmt.Errorf("failed to find user by ID data hash: %w", err)
	}

	return user, nil
}

// Document operations

// SaveDocumentResult saves the result of document processing
func (s *Storage) SaveDocumentResult(result *models.DocumentResult) error {
	query := `
		INSERT INTO document_results (
			session_id, side, status, confidence, rejection_reason,
			processed_at, image_hash
		) VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (session_id, side)
		DO UPDATE SET
			status = EXCLUDED.status,
			confidence = EXCLUDED.confidence,
			rejection_reason = EXCLUDED.rejection_reason,
			processed_at = EXCLUDED.processed_at,
			image_hash = EXCLUDED.image_hash`

	_, err := s.db.Exec(query,
		result.SessionID,
		result.Side,
		result.Status,
		result.Confidence,
		result.RejectionReason,
		result.ProcessedAt,
		result.ImageHash,
	)

	if err != nil {
		return fmt.Errorf("failed to save document result: %w", err)
	}

	return nil
}

// SaveExtractedData saves the OCR-extracted data from a document
func (s *Storage) SaveExtractedData(data *models.ExtractedData) error {
	query := `
		INSERT INTO extracted_data (
			session_id, side, name_arabic, name_latin, date_of_birth,
			id_number, gender, nationality
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (session_id, side)
		DO UPDATE SET
			name_arabic = EXCLUDED.name_arabic,
			name_latin = EXCLUDED.name_latin,
			date_of_birth = EXCLUDED.date_of_birth,
			id_number = EXCLUDED.id_number,
			gender = EXCLUDED.gender,
			nationality = EXCLUDED.nationality`

	// Convert empty Gender to nil for database constraint
	var gender *string
	if data.Gender != nil && (*data.Gender == "male" || *data.Gender == "female") {
		g := string(*data.Gender)
		gender = &g
	}

	_, err := s.db.Exec(query,
		data.SessionID,
		data.Side,
		data.NameArabic,
		data.NameLatin,
		data.DateOfBirth,
		data.IDNumber,
		gender,
		data.Nationality,
	)

	if err != nil {
		log.Printf("Database error saving extracted data: %v", err)
		return fmt.Errorf("failed to save extracted data: %w", err)
	}

	return nil
}

// Selfie operations

// SaveSelfieResult saves the result of selfie processing
func (s *Storage) SaveSelfieResult(result *models.SelfieResult) error {
	query := `
		INSERT INTO selfie_results (
			session_id, image_hash, liveness_score, face_match_score, processed_at
		) VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (session_id)
		DO UPDATE SET
			image_hash = EXCLUDED.image_hash,
			liveness_score = EXCLUDED.liveness_score,
			face_match_score = EXCLUDED.face_match_score,
			processed_at = EXCLUDED.processed_at`

	_, err := s.db.Exec(query,
		result.SessionID,
		result.ImageHash,
		result.LivenessScore,
		result.FaceMatchScore,
		result.ProcessedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to save selfie result: %w", err)
	}

	return nil
}

// Audit operations

// LogAuditEvent logs an audit event
func (s *Storage) LogAuditEvent(event *models.AuditLog) error {
	query := `
		INSERT INTO audit_logs (
			id, user_id, session_id, action, resource, ip_address,
			user_agent, request_id, details
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`

	_, err := s.db.Exec(query,
		event.ID,
		event.UserID,
		event.SessionID,
		event.Action,
		event.Resource,
		event.IPAddress,
		event.UserAgent,
		event.RequestID,
		event.Details,
	)

	if err != nil {
		return fmt.Errorf("failed to log audit event: %w", err)
	}

	return nil
}