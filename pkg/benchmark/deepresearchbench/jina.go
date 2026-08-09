package deepresearchbench

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	jinaReaderURLPrefix = "https://r.jina.ai/"
	jinaMaxRetries      = 3
	jinaRetryDelay      = time.Second
	jinaTimeout         = 60 * time.Second
	maxJinaBytes        = 8 << 20
)

// jinaScraper fetches URL content for FACT's citation-verification stage.
// It is an interface so tests can substitute an in-memory fake instead of
// making real network calls.
type jinaScraper interface {
	Fetch(ctx context.Context, url string) (content string, err error)
}

// jinaClient calls the Jina AI Reader API.
type jinaClient struct {
	apiKey     []byte
	httpClient *http.Client
}

var _ jinaScraper = (*jinaClient)(nil)

func newJinaClient(apiKey []byte) *jinaClient {
	return &jinaClient{apiKey: apiKey, httpClient: &http.Client{Timeout: jinaTimeout}}
}

type jinaResponse struct {
	Data struct {
		URL           string `json:"url"`
		Title         string `json:"title"`
		Description   string `json:"description"`
		Content       string `json:"content"`
		PublishedTime string `json:"publishedTime"`
	} `json:"data"`
}

// Fetch calls the Jina AI Reader API for url, retrying up to jinaMaxRetries
// times with jinaRetryDelay between attempts (matching upstream). On
// success it concatenates title/description/content, matching upstream's
// scrape.py formatting.
func (j *jinaClient) Fetch(ctx context.Context, targetURL string) (string, error) {
	var lastErr error
	for attempt := 0; attempt < jinaMaxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(jinaRetryDelay):
			}
		}
		content, err := j.fetchOnce(ctx, targetURL)
		if err == nil {
			return content, nil
		}
		lastErr = err
	}
	return "", lastErr
}

func (j *jinaClient) fetchOnce(ctx context.Context, targetURL string) (string, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, jinaReaderURLPrefix+targetURL, nil)
	if err != nil {
		return "", fmt.Errorf("build jina reader request: %w", err)
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Authorization", "Bearer "+string(j.apiKey))
	request.Header.Set("X-Timeout", "60000")
	request.Header.Set("X-With-Generated-Alt", "true")

	response, err := j.httpClient.Do(request)
	if err != nil {
		return "", fmt.Errorf("jina reader request failed: %w", err)
	}
	defer response.Body.Close()

	body, err := io.ReadAll(io.LimitReader(response.Body, maxJinaBytes+1))
	if err != nil {
		return "", fmt.Errorf("read jina reader response: %w", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", fmt.Errorf("jina reader request for %q returned HTTP %d", targetURL, response.StatusCode)
	}
	if len(body) > maxJinaBytes {
		return "", fmt.Errorf("jina reader response for %q is too large", targetURL)
	}

	var decoded jinaResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return "", fmt.Errorf("parse jina reader response for %q: %w", targetURL, err)
	}
	return fmt.Sprintf("%s\n\n%s\n\n%s", decoded.Data.Title, decoded.Data.Description, decoded.Data.Content), nil
}
