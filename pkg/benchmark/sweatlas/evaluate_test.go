package sweatlas

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/hyscale-lab/aries/pkg/core"
)

func benchmarkWithFixture(t *testing.T, outputDir string) (*Benchmark, core.Task, taskDetails) {
	t.Helper()
	root := writeFixture(t)
	task, details, err := loadTask(root, qaTaskID)
	if err != nil {
		t.Fatal(err)
	}
	benchmark, err := New(testOptions(root, []string{qaTaskID}, outputDir))
	if err != nil {
		t.Fatal(err)
	}
	benchmark.details[qaTaskID] = details
	return benchmark, task, details
}

func TestEvaluateSynthesizesJudgeEnvironmentRatherThanTaskFileTemplateValues(t *testing.T) {
	benchmark, task, details := benchmarkWithFixture(t, filepath.Join(t.TempDir(), "runs"))
	sandbox := &fakeSandbox{verifierOutputs: map[string][]byte{
		filepath.Join(verifierLogPath, "reward.txt"):              []byte("1\n"),
		filepath.Join(verifierLogPath, "evaluation_results.json"): []byte(`{"agg_score":1.0}`),
	}}

	evaluation, err := benchmark.Evaluate(context.Background(), task, sandbox)
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if evaluation.Status != core.StatusSucceeded || evaluation.Reward != 1 || evaluation.Score != 1 {
		t.Fatalf("Evaluation = %#v", evaluation)
	}
	wantEnv := map[string]string{
		"EVAL_API_KEY":  "fake-key",
		"EVAL_BASE_URL": "https://api.deepseek.com",
		"EVAL_MODEL":    "deepseek-v4-flash",
	}
	if !reflect.DeepEqual(sandbox.verifierCommand.Env, wantEnv) {
		t.Fatalf("verifier environment = %#v, want %#v", sandbox.verifierCommand.Env, wantEnv)
	}
	for _, literal := range []string{"${OPENAI_API_KEY}", "${OPENAI_API_BASE}", "${EVAL_MODEL"} {
		for _, value := range sandbox.verifierCommand.Env {
			if strings.Contains(value, literal) {
				t.Fatalf("verifier environment leaked task.toml template literal %q: %#v", literal, sandbox.verifierCommand.Env)
			}
		}
	}
	if sandbox.verifierCommand.Dir != details.workdir {
		t.Fatalf("verifier dir = %q, want %q", sandbox.verifierCommand.Dir, details.workdir)
	}
}

func TestEvaluateFailsClosedWhenJudgeAPIKeyLookupMisses(t *testing.T) {
	root := writeFixture(t)
	task, details, err := loadTask(root, qaTaskID)
	if err != nil {
		t.Fatal(err)
	}
	options := testOptions(root, []string{qaTaskID}, filepath.Join(t.TempDir(), "runs"))
	options.APIKeyLookup = func(string) ([]byte, bool) { return nil, false }
	benchmark, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	benchmark.details[qaTaskID] = details
	sandbox := &fakeSandbox{}

	_, err = benchmark.Evaluate(context.Background(), task, sandbox)
	if err == nil || !strings.Contains(err.Error(), "judge API key") {
		t.Fatalf("Evaluate() error = %v, want judge API key resolution failure", err)
	}
	if len(sandbox.events) != 0 {
		t.Fatalf("sandbox touched before judge key resolved: %v", sandbox.events)
	}
}

func TestEvaluateRewardStates(t *testing.T) {
	tests := []struct {
		name       string
		reward     []byte
		downloadOK bool
		wantStatus string
		wantReward float64
		wantErr    string
	}{
		{"one", []byte("1\n"), true, core.StatusSucceeded, 1, ""},
		{"surrounding whitespace", []byte(" 1\n\n"), true, core.StatusSucceeded, 1, ""},
		{"zero", []byte("0\n"), true, core.StatusFailed, 0, ""},
		{"missing", nil, false, core.StatusFailed, 0, "download verifier reward"},
		{"malformed", []byte("2\n"), true, core.StatusFailed, 0, "malformed verifier reward"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			benchmark, task, _ := benchmarkWithFixture(t, filepath.Join(t.TempDir(), "runs"))
			downloads := map[string][]byte{filepath.Join(verifierLogPath, "evaluation_results.json"): []byte(`{"agg_score":0.5}`)}
			if test.downloadOK {
				downloads[filepath.Join(verifierLogPath, "reward.txt")] = test.reward
			}
			sandbox := &fakeSandbox{verifierOutputs: downloads}

			evaluation, err := benchmark.Evaluate(context.Background(), task, sandbox)
			if evaluation.Status != test.wantStatus || evaluation.Reward != test.wantReward {
				t.Fatalf("Evaluation = %#v, want status %s reward %v", evaluation, test.wantStatus, test.wantReward)
			}
			if test.wantErr == "" && err != nil {
				t.Fatalf("Evaluate() error = %v", err)
			}
			if test.wantErr != "" && (err == nil || !strings.Contains(err.Error(), test.wantErr)) {
				t.Fatalf("Evaluate() error = %v, want %q", err, test.wantErr)
			}
		})
	}
}

func TestEvaluateRejectsOutOfRangeAggScoreRatherThanClamping(t *testing.T) {
	tests := []struct {
		name    string
		results string
	}{
		{"negative", `{"agg_score":-0.1}`},
		{"above one", `{"agg_score":1.1}`},
		{"nan", `{"agg_score":NaN}`},
		{"missing field", `{}`},
		{"malformed json", `not json`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			benchmark, task, _ := benchmarkWithFixture(t, filepath.Join(t.TempDir(), "runs"))
			sandbox := &fakeSandbox{verifierOutputs: map[string][]byte{
				filepath.Join(verifierLogPath, "reward.txt"):              []byte("1\n"),
				filepath.Join(verifierLogPath, "evaluation_results.json"): []byte(test.results),
			}}

			evaluation, err := benchmark.Evaluate(context.Background(), task, sandbox)
			if test.name == "missing field" {
				// agg_score absent decodes as the zero value 0.0, which is a
				// legitimate in-range score, not a fault.
				if err != nil || evaluation.Score != 0 {
					t.Fatalf("Evaluate() = %#v, %v, want accepted zero score", evaluation, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("Evaluate() accepted malformed agg_score: %#v", evaluation)
			}
			if evaluation.Status != core.StatusFailed {
				t.Fatalf("Evaluation = %#v, want failed status on malformed agg_score", evaluation)
			}
		})
	}
}

func TestEvaluateInjectsTestsOnlyWhenCalled(t *testing.T) {
	benchmark, task, details := benchmarkWithFixture(t, filepath.Join(t.TempDir(), "runs"))
	sandbox := &fakeSandbox{preseededPaths: true, preseededSymlinks: true, verifierOutputs: map[string][]byte{
		filepath.Join(verifierLogPath, "reward.txt"):              []byte("1\n"),
		filepath.Join(verifierLogPath, "evaluation_results.json"): []byte(`{"agg_score":1.0}`),
	}}
	if len(sandbox.events) != 0 {
		t.Fatal("verifier material appeared before Evaluate")
	}

	evaluation, err := benchmark.Evaluate(context.Background(), task, sandbox)
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if evaluation.Status != core.StatusSucceeded {
		t.Fatalf("Evaluation = %#v", evaluation)
	}
	wantEvents := []string{
		"exec:/bin/rm",
		"exec:/bin/mkdir",
		"upload:/tests/evaluate_answer.py",
		"upload:/tests/rubrics.json",
		"upload:/tests/test.sh",
		"exec:/bin/bash",
		"download:" + filepath.Join(verifierLogPath, "reward.txt"),
		"download:" + filepath.Join(verifierLogPath, "evaluation_results.json"),
	}
	got := append([]string(nil), sandbox.events...)
	if !reflect.DeepEqual(sortedCopy(got[:2]), sortedCopy(wantEvents[:2])) {
		t.Fatalf("reset events = %v", got[:2])
	}
	uploadCount := len(details.verifierFiles)
	if len(got) != len(wantEvents) {
		t.Fatalf("events = %v, want %d events", got, len(wantEvents))
	}
	if uploadCount != 3 {
		t.Fatalf("verifier files = %d, want 3", uploadCount)
	}
	if sandbox.preseededPaths || sandbox.preseededSymlinks {
		t.Fatal("preseeded verifier paths or symlinks survived cleanup")
	}
	for _, source := range sandbox.uploadSources {
		if strings.Contains(source, "solution") {
			t.Fatalf("solution path uploaded as verifier: %q", source)
		}
	}
}

func TestEvaluateRejectsNonzeroVerifierCommand(t *testing.T) {
	benchmark, task, _ := benchmarkWithFixture(t, filepath.Join(t.TempDir(), "runs"))
	sandbox := &fakeSandbox{
		commandResult: core.CommandResult{ExitCode: 7},
		verifierOutputs: map[string][]byte{
			filepath.Join(verifierLogPath, "reward.txt"):              []byte("1\n"),
			filepath.Join(verifierLogPath, "evaluation_results.json"): []byte(`{"agg_score":1.0}`),
		},
	}

	evaluation, err := benchmark.Evaluate(context.Background(), task, sandbox)
	if err == nil || !strings.Contains(err.Error(), "exit code 7") {
		t.Fatalf("Evaluate() error = %v, want verifier exit code", err)
	}
	if evaluation.Status != core.StatusFailed || evaluation.Reward != 0 {
		t.Fatalf("Evaluation accepted nonzero verifier command: %#v", evaluation)
	}
}

func sortedCopy(values []string) []string {
	copied := append([]string(nil), values...)
	for i := range copied {
		for j := i + 1; j < len(copied); j++ {
			if copied[j] < copied[i] {
				copied[i], copied[j] = copied[j], copied[i]
			}
		}
	}
	return copied
}

type fakeSandbox struct {
	events             []string
	commands           []core.Command
	downloads          map[string][]byte
	verifierOutputs    map[string][]byte
	commandResult      core.CommandResult
	commandErr         error
	uploadSources      []string
	uploadDestinations []string
	verifierCommand    core.Command
	preseededPaths     bool
	preseededSymlinks  bool
	cleanupComplete    bool
}

func (s *fakeSandbox) Exec(_ context.Context, command core.Command) (core.CommandResult, error) {
	s.events = append(s.events, "exec:"+command.Path)
	s.commands = append(s.commands, command)
	if command.Path == "/bin/rm" {
		s.preseededPaths = false
		s.preseededSymlinks = false
		s.cleanupComplete = true
		s.downloads = nil
	}
	if command.Path == "/bin/bash" {
		if !s.cleanupComplete {
			return core.CommandResult{}, errors.New("verifier ran before cleanup")
		}
		s.verifierCommand = command
		s.downloads = cloneBytesMap(s.verifierOutputs)
		return s.commandResult, s.commandErr
	}
	return core.CommandResult{}, nil
}

func (s *fakeSandbox) Upload(_ context.Context, source, destination string) error {
	s.events = append(s.events, "upload:"+destination)
	if !s.cleanupComplete {
		return errors.New("upload before cleanup")
	}
	s.uploadSources = append(s.uploadSources, source)
	s.uploadDestinations = append(s.uploadDestinations, destination)
	return nil
}

func (s *fakeSandbox) Download(_ context.Context, source, destination string) error {
	s.events = append(s.events, "download:"+source)
	content, ok := s.downloads[source]
	if !ok {
		return errors.New("not found")
	}
	return os.WriteFile(destination, content, 0o600)
}

func cloneBytesMap(source map[string][]byte) map[string][]byte {
	clone := make(map[string][]byte, len(source))
	for key, value := range source {
		clone[key] = append([]byte(nil), value...)
	}
	return clone
}
