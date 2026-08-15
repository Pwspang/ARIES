package deepresearchbench

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hyscale-lab/aries/pkg/core"
)

func newPlanOnlyTestBenchmark(t *testing.T) (*Benchmark, string) {
	t.Helper()
	root := writeFixture(t, defaultFixtureRows(t))
	plansetDir := filepath.Join(t.TempDir(), "planset")
	options := baseOptions(root)
	options.Judge = core.ModelConfig{}
	options.JudgeDisabled = true
	options.PlanOnly = true
	options.PlansetDir = plansetDir
	options.OutputDir = t.TempDir()
	benchmark, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := benchmark.Tasks(context.Background()); err != nil {
		t.Fatal(err)
	}
	return benchmark, plansetDir
}

func TestNewRequiresJudgeDisabledForPlanOnly(t *testing.T) {
	root := writeFixture(t, defaultFixtureRows(t))
	options := baseOptions(root)
	options.PlanOnly = true
	options.PlansetDir = t.TempDir()
	if _, err := New(options); err == nil {
		t.Fatal("accepted plan-only mode without JudgeDisabled")
	}
}

func TestNewRequiresPlansetDirForPlanOnly(t *testing.T) {
	root := writeFixture(t, defaultFixtureRows(t))
	options := baseOptions(root)
	options.Judge = core.ModelConfig{}
	options.JudgeDisabled = true
	options.PlanOnly = true
	if _, err := New(options); err == nil {
		t.Fatal("accepted plan-only mode without a planset directory")
	}
}

func TestNewRejectsPlanOnlyWithStructuredSubtasks(t *testing.T) {
	root := writeFixture(t, defaultFixtureRows(t))
	options := baseOptions(root)
	options.Judge = core.ModelConfig{}
	options.JudgeDisabled = true
	options.PlanOnly = true
	options.PlansetDir = t.TempDir()
	options.StructuredSubtasks = &StructuredSubtasksOptions{PlansetDir: t.TempDir(), Order: "sequential"}
	if _, err := New(options); err == nil {
		t.Fatal("accepted plan-only mode combined with structured subtasks")
	}
}

func TestTasksPlanOnlyOmitsReportContract(t *testing.T) {
	benchmark, _ := newPlanOnlyTestBenchmark(t)
	tasks, err := benchmark.Tasks(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	instruction := tasks[0].Instruction
	if !strings.Contains(instruction, planPath) {
		t.Fatalf("plan-only instruction = %q, want it to mention %q", instruction, planPath)
	}
	if strings.Contains(instruction, reportPath) {
		t.Fatalf("plan-only instruction = %q, want it to never mention reportPath %q", instruction, reportPath)
	}
	if strings.Contains(instruction, "inline citations") {
		t.Fatalf("plan-only instruction = %q, want no citation-format rules", instruction)
	}
}

func TestEvaluatePlanCapturesValidPlanToPlansetDir(t *testing.T) {
	benchmark, plansetDir := newPlanOnlyTestBenchmark(t)
	sandbox := &evaluateFake{downloadContent: `["first sub-question", "  second sub-question  "]`}
	evaluation, err := benchmark.Evaluate(context.Background(), core.Task{ID: "1"}, sandbox)
	if err != nil {
		t.Fatal(err)
	}
	if evaluation.Status != core.StatusSucceeded {
		t.Fatalf("evaluation = %+v, want Status succeeded", evaluation)
	}
	if evaluation.Score != 0 || evaluation.Reward != 0 {
		t.Fatalf("evaluation = %+v, want Score=0 Reward=0 (administrative pass, not a judged score)", evaluation)
	}
	plansetPath := filepath.Join(plansetDir, "1.json")
	raw, err := os.ReadFile(plansetPath)
	if err != nil {
		t.Fatalf("planset file not written: %v", err)
	}
	var decoded []string
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	want := []string{"first sub-question", "second sub-question"}
	if len(decoded) != 2 || decoded[0] != want[0] || decoded[1] != want[1] {
		t.Fatalf("decoded planset = %v, want %v (trimmed)", decoded, want)
	}
	if sandbox.downloadSource != planPath {
		t.Fatalf("downloadSource = %q, want %q", sandbox.downloadSource, planPath)
	}
}

func TestEvaluatePlanFailsLoudlyWhenMissing(t *testing.T) {
	benchmark, plansetDir := newPlanOnlyTestBenchmark(t)
	sandbox := &evaluateFake{downloadErr: os.ErrNotExist}
	evaluation, err := benchmark.Evaluate(context.Background(), core.Task{ID: "1"}, sandbox)
	if err != nil {
		t.Fatalf("Evaluate returned a Go error for a missing plan: %v", err)
	}
	if evaluation.Status != core.StatusFailed {
		t.Fatalf("evaluation.Status = %q, want failed", evaluation.Status)
	}
	if evaluation.Error == "" {
		t.Fatal("evaluation.Error is empty for a missing plan")
	}
	if _, err := os.Stat(filepath.Join(plansetDir, "1.json")); !os.IsNotExist(err) {
		t.Fatalf("planset file exists despite a missing plan: %v", err)
	}
}

func TestEvaluatePlanFailsLoudlyWhenUnparseable(t *testing.T) {
	benchmark, plansetDir := newPlanOnlyTestBenchmark(t)
	sandbox := &evaluateFake{downloadContent: "not json"}
	evaluation, err := benchmark.Evaluate(context.Background(), core.Task{ID: "1"}, sandbox)
	if err != nil {
		t.Fatal(err)
	}
	if evaluation.Status != core.StatusFailed || evaluation.Error == "" {
		t.Fatalf("evaluation = %+v, want failed with a clear error", evaluation)
	}
	if _, err := os.Stat(filepath.Join(plansetDir, "1.json")); !os.IsNotExist(err) {
		t.Fatalf("planset file exists despite an unparseable plan: %v", err)
	}
}

func TestEvaluatePlanFailsLoudlyWhenEmptyArray(t *testing.T) {
	benchmark, _ := newPlanOnlyTestBenchmark(t)
	sandbox := &evaluateFake{downloadContent: `[]`}
	evaluation, err := benchmark.Evaluate(context.Background(), core.Task{ID: "1"}, sandbox)
	if err != nil {
		t.Fatal(err)
	}
	if evaluation.Status != core.StatusFailed || evaluation.Error == "" {
		t.Fatalf("evaluation = %+v, want failed with a clear error", evaluation)
	}
}

func TestEvaluatePlanFailsLoudlyWhenContainsEmptyString(t *testing.T) {
	benchmark, _ := newPlanOnlyTestBenchmark(t)
	sandbox := &evaluateFake{downloadContent: `["a real sub-question", "   "]`}
	evaluation, err := benchmark.Evaluate(context.Background(), core.Task{ID: "1"}, sandbox)
	if err != nil {
		t.Fatal(err)
	}
	if evaluation.Status != core.StatusFailed || evaluation.Error == "" {
		t.Fatalf("evaluation = %+v, want failed with a clear error", evaluation)
	}
}
