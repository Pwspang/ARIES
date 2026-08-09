package deepresearchbench

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hyscale-lab/aries/pkg/core"
)

// chatCompletionServer is a shared httptest fake OpenAI-compatible
// chat-completions endpoint, reused by judge_test.go, race_test.go, and
// fact_test.go: all three build on judgeClient.chat, so they all exercise
// the same request/response shape.
func chatCompletionServer(t *testing.T, content string, statusCode int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer fake-key" {
			t.Fatalf("unexpected Authorization header %q", r.Header.Get("Authorization"))
		}
		w.WriteHeader(statusCode)
		response := map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"content": content}},
			},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
}

func newTestJudgeClient(t *testing.T, baseURL string) *judgeClient {
	t.Helper()
	client, err := newJudgeClient(core.ModelConfig{
		Provider: "openai", BaseURL: baseURL, Model: "gpt-4.1", APIKeyEnv: "OPENAI_API_KEY",
	}, fakeAPIKeyLookup)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func TestJudgeClientChatReturnsContent(t *testing.T) {
	server := chatCompletionServer(t, "hello", http.StatusOK)
	defer server.Close()
	client := newTestJudgeClient(t, server.URL+"/v1")
	content, err := client.chat(context.Background(), "system", "user")
	if err != nil {
		t.Fatal(err)
	}
	if content != "hello" {
		t.Fatalf("chat() = %q, want %q", content, "hello")
	}
}

func TestJudgeClientChatRejectsHTTPError(t *testing.T) {
	server := chatCompletionServer(t, "{}", http.StatusInternalServerError)
	defer server.Close()
	client := newTestJudgeClient(t, server.URL+"/v1")
	if _, err := client.chat(context.Background(), "system", "user"); err == nil {
		t.Fatal("chat() accepted an HTTP error response")
	}
}

func TestJudgeClientChatRejectsEmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{}})
	}))
	defer server.Close()
	client := newTestJudgeClient(t, server.URL+"/v1")
	if _, err := client.chat(context.Background(), "system", "user"); err == nil {
		t.Fatal("chat() accepted a response with no choices")
	}
}

func TestNewJudgeClientRejectsMissingAPIKey(t *testing.T) {
	_, err := newJudgeClient(core.ModelConfig{
		Provider: "openai", BaseURL: "https://api.openai.com/v1", Model: "gpt-4.1", APIKeyEnv: "OPENAI_API_KEY",
	}, func(string) ([]byte, bool) { return nil, false })
	if err == nil {
		t.Fatal("newJudgeClient accepted a missing API key")
	}
}

func TestNewJudgeClientRejectsInvalidBaseURL(t *testing.T) {
	for _, baseURL := range []string{"", "not-a-url", "ftp://example.com", "https://user:pass@example.com/v1"} {
		if _, err := newJudgeClient(core.ModelConfig{
			Provider: "openai", BaseURL: baseURL, Model: "gpt-4.1", APIKeyEnv: "OPENAI_API_KEY",
		}, fakeAPIKeyLookup); err == nil {
			t.Fatalf("newJudgeClient accepted invalid base URL %q", baseURL)
		}
	}
}
