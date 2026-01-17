package ml

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"os"
	"time"

	"github.com/Samimelki/minak-enta/internal/models"
)

// getEnvOrDefault returns the environment variable value or a default if not set
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

type MLClient struct {
	httpClient *http.Client
	baseURLs   map[string]string
}

type OCRResponse struct {
	Status          string             `json:"status"`
	ConfidenceScore float64            `json:"confidence_score"`
	ExtractedData   models.ExtractedData `json:"extracted_data"`
	FieldConfidences map[string]float64 `json:"field_confidences"`
	Metadata        map[string]interface{} `json:"metadata"`
}

type FaceDetectResponse struct {
	Status   string                   `json:"status"`
	Faces    []map[string]interface{} `json:"faces"`
	Metadata map[string]interface{}   `json:"metadata"`
}

type FaceExtractResponse struct {
	Status   string                   `json:"status"`
	Faces    []map[string]interface{} `json:"faces"`
	Metadata map[string]interface{}   `json:"metadata"`
}

type FaceCompareResponse struct {
	SimilarityScore float64            `json:"similarity_score"`
	Distance        float64            `json:"distance"`
	Confidence      float64            `json:"confidence"`
	Metadata        map[string]interface{} `json:"metadata"`
}

type LivenessResponse struct {
	Status        string  `json:"status"`
	LivenessScore float64 `json:"liveness_score"`
	Confidence    float64 `json:"confidence"`
	Analysis      map[string]interface{} `json:"analysis"`
	Metadata      map[string]interface{} `json:"metadata"`
}

func NewMLClient() *MLClient {
	return &MLClient{
		httpClient: &http.Client{
			Timeout: 120 * time.Second, // OCR can take 50+ seconds on first run
		},
		baseURLs: map[string]string{
			"ocr":      getEnvOrDefault("OCR_SERVICE_URL", "http://localhost:35001"),
			"face":     getEnvOrDefault("FACE_SERVICE_URL", "http://localhost:35002"),
			"liveness": getEnvOrDefault("LIVENESS_SERVICE_URL", "http://localhost:35003"),
		},
	}
}

func (c *MLClient) ProcessDocument(imageData []byte, documentType string) (*OCRResponse, error) {
	url := fmt.Sprintf("%s/process-document", c.baseURLs["ocr"])

	// Create multipart form
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	// Add image file with proper content type
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", `form-data; name="file"; filename="document.jpg"`)
	h.Set("Content-Type", "image/jpeg")
	fw, err := w.CreatePart(h)
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	fw.Write(imageData)

	// Add document type
	w.WriteField("document_type", documentType)

	w.Close()

	// Make request
	req, err := http.NewRequest("POST", url, &b)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call OCR service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OCR service returned status %d: %s", resp.StatusCode, string(body))
	}

	var result OCRResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode OCR response: %w", err)
	}

	return &result, nil
}

func (c *MLClient) DetectFaces(imageData []byte) (*FaceDetectResponse, error) {
	url := fmt.Sprintf("%s/detect-faces", c.baseURLs["face"])

	// Create multipart form
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", `form-data; name="file"; filename="face.jpg"`)
	h.Set("Content-Type", "image/jpeg")
	fw, err := w.CreatePart(h)
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	fw.Write(imageData)
	w.Close()

	req, err := http.NewRequest("POST", url, &b)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call face detection service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("face detection service returned status %d: %s", resp.StatusCode, string(body))
	}

	var result FaceDetectResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode face detection response: %w", err)
	}

	return &result, nil
}

func (c *MLClient) ExtractFaceEmbedding(imageData []byte) (*FaceExtractResponse, error) {
	url := fmt.Sprintf("%s/extract-embedding", c.baseURLs["face"])

	// Create multipart form
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", `form-data; name="file"; filename="face.jpg"`)
	h.Set("Content-Type", "image/jpeg")
	fw, err := w.CreatePart(h)
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	fw.Write(imageData)
	w.Close()

	req, err := http.NewRequest("POST", url, &b)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call face extraction service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("face extraction service returned status %d: %s", resp.StatusCode, string(body))
	}

	var result FaceExtractResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode face extraction response: %w", err)
	}

	return &result, nil
}

func (c *MLClient) CompareFaces(embedding1, embedding2 []float64) (*FaceCompareResponse, error) {
	url := fmt.Sprintf("%s/compare-faces", c.baseURLs["face"])

	payload := map[string]interface{}{
		"embedding1": embedding1,
		"embedding2": embedding2,
		"metric":     "cosine",
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal comparison payload: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call face comparison service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("face comparison service returned status %d: %s", resp.StatusCode, string(body))
	}

	var result FaceCompareResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode face comparison response: %w", err)
	}

	return &result, nil
}

func (c *MLClient) DetectLiveness(imageData []byte) (*LivenessResponse, error) {
	url := fmt.Sprintf("%s/detect-liveness", c.baseURLs["liveness"])

	// Create multipart form
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", `form-data; name="file"; filename="face.jpg"`)
	h.Set("Content-Type", "image/jpeg")
	fw, err := w.CreatePart(h)
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	fw.Write(imageData)
	w.Close()

	req, err := http.NewRequest("POST", url, &b)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call liveness detection service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("liveness detection service returned status %d: %s", resp.StatusCode, string(body))
	}

	var result LivenessResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode liveness response: %w", err)
	}

	return &result, nil
}

// Helper methods for converting between types
func ConvertToExtractedData(ocrData map[string]interface{}) *models.ExtractedData {
	extracted := &models.ExtractedData{}

	if nameArabic, ok := ocrData["name_arabic"].(string); ok && nameArabic != "" {
		extracted.NameArabic = &nameArabic
	}
	if nameLatin, ok := ocrData["name_latin"].(string); ok && nameLatin != "" {
		extracted.NameLatin = &nameLatin
	}
	if dateOfBirth, ok := ocrData["date_of_birth"].(string); ok && dateOfBirth != "" {
		extracted.DateOfBirth = &dateOfBirth
	}
	if idNumber, ok := ocrData["id_number"].(string); ok && idNumber != "" {
		extracted.IDNumber = &idNumber
	}
	if gender, ok := ocrData["gender"].(string); ok && gender != "" {
		g := models.Gender(gender)
		extracted.Gender = &g
	}
	if nationality, ok := ocrData["nationality"].(string); ok && nationality != "" {
		extracted.Nationality = &nationality
	}

	return extracted
}