package deepresearchbench

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hyscale-lab/aries/pkg/core"
)

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func exampleCriteria() taskCriteria {
	return taskCriteria{
		Weights: dimensionWeights{Comprehensiveness: 0.4, Insight: 0.3, InstructionFollowing: 0.2, Readability: 0.1},
		Criterions: dimensionCriteria{
			Comprehensiveness: []criterion{
				{Criterion: "coverage", Explanation: "breadth", Weight: 0.6},
				{Criterion: "depth", Explanation: "detail", Weight: 0.4},
			},
			Insight:              []criterion{{Criterion: "analysis", Explanation: "insight", Weight: 1.0}},
			InstructionFollowing: []criterion{{Criterion: "adherence", Explanation: "follows task", Weight: 1.0}},
			Readability:          []criterion{{Criterion: "clarity", Explanation: "clear", Weight: 1.0}},
		},
	}
}

func exampleJudgeOutput() raceJudgeOutput {
	return raceJudgeOutput{
		Comprehensiveness: []raceCriterionScore{
			{Criterion: "coverage", Article1Score: 8, Article2Score: 6},
			{Criterion: "depth", Article1Score: 4, Article2Score: 5},
		},
		Insight:              []raceCriterionScore{{Criterion: "analysis", Article1Score: 7, Article2Score: 7}},
		InstructionFollowing: []raceCriterionScore{{Criterion: "adherence", Article1Score: 9, Article2Score: 3}},
		Readability:          []raceCriterionScore{{Criterion: "clarity", Article1Score: 5, Article2Score: 5}},
	}
}

func TestCalculateWeightedScoresMatchesHandComputedExample(t *testing.T) {
	result := calculateWeightedScores(exampleJudgeOutput(), exampleCriteria())

	// comprehensiveness: target=(8*0.6+4*0.4)/1.0=6.4, reference=(6*0.6+5*0.4)/1.0=5.6
	// insight: target=7, reference=7
	// instruction_following: target=9, reference=3
	// readability: target=5, reference=5
	// total_target = 6.4*0.4 + 7*0.3 + 9*0.2 + 5*0.1 = 6.96
	// total_reference = 5.6*0.4 + 7*0.3 + 3*0.2 + 5*0.1 = 5.44
	wantOverall := 6.96 / (6.96 + 5.44)
	wantComprehensiveness := 6.4 / (6.4 + 5.6)
	wantInsight := 0.5
	wantInstructionFollowing := 9.0 / 12.0
	wantReadability := 0.5

	if !almostEqual(result.Overall, wantOverall) {
		t.Fatalf("Overall = %v, want %v", result.Overall, wantOverall)
	}
	if !almostEqual(result.Comprehensiveness, wantComprehensiveness) {
		t.Fatalf("Comprehensiveness = %v, want %v", result.Comprehensiveness, wantComprehensiveness)
	}
	if !almostEqual(result.Insight, wantInsight) {
		t.Fatalf("Insight = %v, want %v", result.Insight, wantInsight)
	}
	if !almostEqual(result.InstructionFollowing, wantInstructionFollowing) {
		t.Fatalf("InstructionFollowing = %v, want %v", result.InstructionFollowing, wantInstructionFollowing)
	}
	if !almostEqual(result.Readability, wantReadability) {
		t.Fatalf("Readability = %v, want %v", result.Readability, wantReadability)
	}
	if len(result.Warnings) != 0 {
		t.Fatalf("Warnings = %v, want none for exact criterion matches", result.Warnings)
	}
}

func TestMatchCriterionWeightExactMatch(t *testing.T) {
	criteria := []criterion{{Criterion: "coverage", Weight: 0.6}, {Criterion: "depth", Weight: 0.4}}
	weight, warning := matchCriterionWeight("coverage", criteria)
	if weight != 0.6 || warning != "" {
		t.Fatalf("matchCriterionWeight() = (%v, %q), want (0.6, \"\")", weight, warning)
	}
}

func TestMatchCriterionWeightCaseInsensitiveMatch(t *testing.T) {
	criteria := []criterion{{Criterion: "Coverage", Weight: 0.6}}
	weight, warning := matchCriterionWeight("coverage", criteria)
	if weight != 0.6 || warning != "" {
		t.Fatalf("matchCriterionWeight() = (%v, %q), want (0.6, \"\")", weight, warning)
	}
}

func TestMatchCriterionWeightSubstringMatch(t *testing.T) {
	criteria := []criterion{{Criterion: "Information Coverage Breadth", Weight: 0.6}}
	weight, warning := matchCriterionWeight("Coverage Breadth", criteria)
	if weight != 0.6 || warning != "" {
		t.Fatalf("matchCriterionWeight() = (%v, %q), want (0.6, \"\")", weight, warning)
	}
}

func TestMatchCriterionWeightFallsBackToAverage(t *testing.T) {
	criteria := []criterion{{Criterion: "coverage", Weight: 0.6}, {Criterion: "depth", Weight: 0.2}}
	weight, warning := matchCriterionWeight("completely unrelated", criteria)
	if !almostEqual(weight, 0.4) {
		t.Fatalf("matchCriterionWeight() weight = %v, want average 0.4", weight)
	}
	if warning == "" {
		t.Fatal("matchCriterionWeight() warning is empty, want a fallback warning")
	}
}

func TestRaceClientScoreParsesValidStructuredOutput(t *testing.T) {
	content := `{"comprehensiveness":[{"criterion":"coverage","analysis":"a","article_1_score":8,"article_2_score":6},` +
		`{"criterion":"depth","analysis":"a","article_1_score":4,"article_2_score":5}],` +
		`"insight":[{"criterion":"analysis","analysis":"a","article_1_score":7,"article_2_score":7}],` +
		`"instruction_following":[{"criterion":"adherence","analysis":"a","article_1_score":9,"article_2_score":3}],` +
		`"readability":[{"criterion":"clarity","analysis":"a","article_1_score":5,"article_2_score":5}]}`
	server := chatCompletionServer(t, content, http.StatusOK)
	defer server.Close()
	chat, err := newJudgeClient(core.ModelConfig{Provider: "openai", BaseURL: server.URL + "/v1", Model: "gpt-4.1", APIKeyEnv: "OPENAI_API_KEY"}, fakeAPIKeyLookup)
	if err != nil {
		t.Fatal(err)
	}
	client := &raceClient{chat: chat}
	result, err := client.Score(context.Background(), "prompt", "target article", "reference article", exampleCriteria())
	if err != nil {
		t.Fatal(err)
	}
	wantOverall := 6.96 / (6.96 + 5.44)
	if !almostEqual(result.Overall, wantOverall) {
		t.Fatalf("Overall = %v, want %v", result.Overall, wantOverall)
	}
}

func TestRaceClientScoreRetriesOnMalformedJSONThenFails(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		response := map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": "not json"}}}}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	chat, err := newJudgeClient(core.ModelConfig{Provider: "openai", BaseURL: server.URL + "/v1", Model: "gpt-4.1", APIKeyEnv: "OPENAI_API_KEY"}, fakeAPIKeyLookup)
	if err != nil {
		t.Fatal(err)
	}
	client := &raceClient{chat: chat}
	if _, err := client.Score(context.Background(), "prompt", "target", "reference", exampleCriteria()); err == nil {
		t.Fatal("Score accepted malformed JSON forever")
	}
	if calls != maxRaceRetries {
		t.Fatalf("calls = %d, want %d retries", calls, maxRaceRetries)
	}
}

func TestRaceClientScoreRetriesOnMissingDimensionThenSucceeds(t *testing.T) {
	valid := `{"comprehensiveness":[{"criterion":"coverage","article_1_score":8,"article_2_score":6},` +
		`{"criterion":"depth","article_1_score":4,"article_2_score":5}],` +
		`"insight":[{"criterion":"analysis","article_1_score":7,"article_2_score":7}],` +
		`"instruction_following":[{"criterion":"adherence","article_1_score":9,"article_2_score":3}],` +
		`"readability":[{"criterion":"clarity","article_1_score":5,"article_2_score":5}]}`
	missingDimension := `{"comprehensiveness":[{"criterion":"coverage","article_1_score":8,"article_2_score":6}]}`

	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		content := missingDimension
		if calls > 1 {
			content = valid
		}
		response := map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": content}}}}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()
	chat, err := newJudgeClient(core.ModelConfig{Provider: "openai", BaseURL: server.URL + "/v1", Model: "gpt-4.1", APIKeyEnv: "OPENAI_API_KEY"}, fakeAPIKeyLookup)
	if err != nil {
		t.Fatal(err)
	}
	client := &raceClient{chat: chat}
	if _, err := client.Score(context.Background(), "prompt", "target", "reference", exampleCriteria()); err != nil {
		t.Fatalf("Score() error = %v, want success on second attempt", err)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want 2 (one failure, one success)", calls)
	}
}

func TestBuildMergedScorePromptEmbedsArticlesAndCriteria(t *testing.T) {
	_, user := buildMergedScorePrompt("the prompt", "target text", "reference text", exampleCriteria().Criterions)
	for _, want := range []string{"the prompt", "target text", "reference text", "coverage", "adherence", "clarity"} {
		if !strings.Contains(user, want) {
			t.Fatalf("prompt missing %q:\n%s", want, user)
		}
	}
	if strings.Contains(user, `"weight"`) {
		t.Fatal("prompt leaks criterion weights, which upstream strips before showing the judge")
	}
}

func TestParseRaceJudgeOutputRejectsMissingDimension(t *testing.T) {
	if _, err := parseRaceJudgeOutput(`{"comprehensiveness":[{"criterion":"c","article_1_score":1,"article_2_score":1}]}`); err == nil {
		t.Fatal("parseRaceJudgeOutput accepted a response missing three dimensions")
	}
}

func TestParseRaceJudgeOutputRejectsInvalidJSON(t *testing.T) {
	if _, err := parseRaceJudgeOutput("not json"); err == nil {
		t.Fatal("parseRaceJudgeOutput accepted invalid JSON")
	}
}

func TestNormalizeRatioHandlesZeroDenominator(t *testing.T) {
	if got := normalizeRatio(0, 0); got != 0 {
		t.Fatalf("normalizeRatio(0, 0) = %v, want 0", got)
	}
	if got := normalizeRatio(1, 1); got != 0.5 {
		t.Fatalf("normalizeRatio(1, 1) = %v, want 0.5", got)
	}
}
