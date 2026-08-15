package deepresearchbench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hyscale-lab/aries/pkg/core"
)

// structuredSubtasksFakeSandbox is a runner.Sandbox fake driven purely by
// which findings paths are "present and non-empty", so tests can simulate a
// subtask turn having (or not having) written its findings file without a
// real container. Exec dispatches on the command path the same way
// verifyAndLockFindings actually issues them: a `/bin/sh -c 'test -s ...'`
// non-empty check, and a `/bin/chmod 0444` lock.
type structuredSubtasksFakeSandbox struct {
	findingsPresent map[string]bool
	chmodCalls      []string
}

func (s *structuredSubtasksFakeSandbox) Exec(_ context.Context, cmd core.Command) (core.CommandResult, error) {
	if len(cmd.Args) == 0 {
		return core.CommandResult{}, nil
	}
	path := cmd.Args[len(cmd.Args)-1]
	switch cmd.Path {
	case "/bin/sh":
		if s.findingsPresent[path] {
			return core.CommandResult{ExitCode: 0}, nil
		}
		return core.CommandResult{ExitCode: 1}, nil
	case "/bin/chmod":
		s.chmodCalls = append(s.chmodCalls, path)
		return core.CommandResult{ExitCode: 0}, nil
	default:
		return core.CommandResult{}, nil
	}
}

func (s *structuredSubtasksFakeSandbox) Upload(context.Context, string, string) error { return nil }
func (s *structuredSubtasksFakeSandbox) Download(context.Context, string, string) error {
	return nil
}

func writePlansetFile(t *testing.T, plansetDir string, numericID int, subtasks []string) {
	t.Helper()
	if err := os.MkdirAll(plansetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(subtasks)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(plansetDir, fmt.Sprintf("%d.json", numericID))
	if err := os.WriteFile(path, encoded, 0o600); err != nil {
		t.Fatal(err)
	}
}

func newStructuredSubtasksTestBenchmark(t *testing.T, structured StructuredSubtasksOptions) *Benchmark {
	t.Helper()
	root := writeFixture(t, defaultFixtureRows(t))
	options := baseOptions(root)
	options.OutputDir = t.TempDir()
	options.StructuredSubtasks = &structured
	benchmark, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := benchmark.Tasks(context.Background()); err != nil {
		t.Fatal(err)
	}
	return benchmark
}

func TestNewRequiresStructuredSubtasksPlansetDir(t *testing.T) {
	root := writeFixture(t, defaultFixtureRows(t))
	options := baseOptions(root)
	options.StructuredSubtasks = &StructuredSubtasksOptions{Order: "sequential"}
	if _, err := New(options); err == nil {
		t.Fatal("accepted structured subtasks without a planset directory")
	}
}

func TestNewValidatesStructuredSubtasksOrder(t *testing.T) {
	root := writeFixture(t, defaultFixtureRows(t))
	options := baseOptions(root)
	options.StructuredSubtasks = &StructuredSubtasksOptions{PlansetDir: t.TempDir(), Order: "random"}
	if _, err := New(options); err == nil {
		t.Fatal("accepted an invalid structured subtasks order")
	}
}

func TestNewRequiresSeedWhenShuffled(t *testing.T) {
	root := writeFixture(t, defaultFixtureRows(t))
	options := baseOptions(root)
	options.StructuredSubtasks = &StructuredSubtasksOptions{PlansetDir: t.TempDir(), Order: "shuffled"}
	if _, err := New(options); err == nil {
		t.Fatal("accepted order:shuffled without a seed")
	}
}

func TestNewAcceptsAdversarialOrderWithoutSeed(t *testing.T) {
	root := writeFixture(t, defaultFixtureRows(t))
	options := baseOptions(root)
	options.StructuredSubtasks = &StructuredSubtasksOptions{PlansetDir: t.TempDir(), Order: "adversarial"}
	if _, err := New(options); err != nil {
		t.Fatalf("rejected order:adversarial without a seed: %v", err)
	}
}

func TestNextTurnSingleTurnWhenStructuredSubtasksNil(t *testing.T) {
	root := writeFixture(t, defaultFixtureRows(t))
	options := baseOptions(root)
	options.OutputDir = t.TempDir()
	benchmark, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	tasks, err := benchmark.Tasks(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	task := tasks[0]
	instruction, ok, err := benchmark.NextTurn(context.Background(), task, 0, &structuredSubtasksFakeSandbox{})
	if err != nil || !ok || instruction != task.Instruction {
		t.Fatalf("NextTurn(turn 0) = (%q, %v, %v), want (task.Instruction, true, nil)", instruction, ok, err)
	}
	if _, ok, err := benchmark.NextTurn(context.Background(), task, 1, &structuredSubtasksFakeSandbox{}); err != nil || ok {
		t.Fatalf("NextTurn(turn 1) ok = %v, err = %v, want ok=false, err=nil", ok, err)
	}
}

func TestNextTurnSequentialOrderMatchesPlanFileAndYieldsNPlusOneTurns(t *testing.T) {
	plansetDir := t.TempDir()
	subtasks := []string{"sub A", "sub B", "sub C"}
	writePlansetFile(t, plansetDir, 1, subtasks)
	benchmark := newStructuredSubtasksTestBenchmark(t, StructuredSubtasksOptions{PlansetDir: plansetDir, Order: "sequential"})

	sandbox := &structuredSubtasksFakeSandbox{findingsPresent: map[string]bool{
		"/tmp/aries-findings/1_subtask.md": true,
		"/tmp/aries-findings/2_subtask.md": true,
		"/tmp/aries-findings/3_subtask.md": true,
	}}
	task := core.Task{ID: "1"}

	for index, want := range subtasks {
		instruction, ok, err := benchmark.NextTurn(context.Background(), task, index, sandbox)
		if err != nil || !ok {
			t.Fatalf("NextTurn(turn %d) = (_, %v, %v), want ok", index, ok, err)
		}
		if !strings.Contains(instruction, want) {
			t.Fatalf("NextTurn(turn %d) instruction = %q, want it to contain %q", index, instruction, want)
		}
		wantFindingsFile := fmt.Sprintf("/tmp/aries-findings/%d_subtask.md", index+1)
		if !strings.Contains(instruction, wantFindingsFile) {
			t.Fatalf("NextTurn(turn %d) instruction = %q, want it to mention %q", index, instruction, wantFindingsFile)
		}
	}

	// Turn N (== len(subtasks)) is the final synthesis turn.
	synthesis, ok, err := benchmark.NextTurn(context.Background(), task, len(subtasks), sandbox)
	if err != nil || !ok {
		t.Fatalf("NextTurn(turn N) = (_, %v, %v), want ok", ok, err)
	}
	if !strings.Contains(synthesis, reportInstruction) {
		t.Fatal("synthesis instruction does not contain reportInstruction verbatim")
	}
	if !strings.Contains(synthesis, "inline citations") || !strings.Contains(synthesis, "Citation format") {
		t.Fatal("synthesis instruction does not reuse the citation-format rules")
	}

	// Turn N+1 ends the loop.
	if _, ok, err := benchmark.NextTurn(context.Background(), task, len(subtasks)+1, sandbox); err != nil || ok {
		t.Fatalf("NextTurn(turn N+1) ok = %v, err = %v, want ok=false, err=nil", ok, err)
	}

	if len(sandbox.chmodCalls) != len(subtasks) {
		t.Fatalf("chmodCalls = %v, want one lock per subtask findings file", sandbox.chmodCalls)
	}
}

func TestNextTurnAdversarialOrderReversesPlanFileOrder(t *testing.T) {
	plansetDir := t.TempDir()
	subtasks := []string{"sub A", "sub B", "sub C"}
	writePlansetFile(t, plansetDir, 1, subtasks)
	benchmark := newStructuredSubtasksTestBenchmark(t, StructuredSubtasksOptions{PlansetDir: plansetDir, Order: "adversarial"})

	sandbox := &structuredSubtasksFakeSandbox{findingsPresent: map[string]bool{
		"/tmp/aries-findings/1_subtask.md": true,
		"/tmp/aries-findings/2_subtask.md": true,
		"/tmp/aries-findings/3_subtask.md": true,
	}}
	task := core.Task{ID: "1"}

	reversed := []string{"sub C", "sub B", "sub A"}
	for index, want := range reversed {
		instruction, ok, err := benchmark.NextTurn(context.Background(), task, index, sandbox)
		if err != nil || !ok {
			t.Fatalf("NextTurn(turn %d) = (_, %v, %v), want ok", index, ok, err)
		}
		if !strings.Contains(instruction, want) {
			t.Fatalf("NextTurn(turn %d) instruction = %q, want it to contain %q (adversarial order should be the exact reverse of the plan file)", index, instruction, want)
		}
	}
}

func TestNextTurnShuffledOrderIsDeterministicGivenSeedAndTaskID(t *testing.T) {
	subtasks := []string{"alpha", "bravo", "charlie", "delta", "echo"}
	sandbox := &structuredSubtasksFakeSandbox{findingsPresent: map[string]bool{
		"/tmp/aries-findings/1_subtask.md": true, "/tmp/aries-findings/2_subtask.md": true, "/tmp/aries-findings/3_subtask.md": true,
		"/tmp/aries-findings/4_subtask.md": true, "/tmp/aries-findings/5_subtask.md": true,
	}}
	task := core.Task{ID: "1"}

	orderOf := func(seed int64) []string {
		plansetDir := t.TempDir()
		writePlansetFile(t, plansetDir, 1, append([]string(nil), subtasks...))
		benchmark := newStructuredSubtasksTestBenchmark(t, StructuredSubtasksOptions{PlansetDir: plansetDir, Order: "shuffled", Seed: seed})
		var order []string
		for index := 0; index < len(subtasks); index++ {
			instruction, ok, err := benchmark.NextTurn(context.Background(), task, index, sandbox)
			if err != nil || !ok {
				t.Fatalf("NextTurn(turn %d) = (_, %v, %v), want ok", index, ok, err)
			}
			for _, subtask := range subtasks {
				if strings.Contains(instruction, subtask) {
					order = append(order, subtask)
					break
				}
			}
		}
		return order
	}

	firstRun := orderOf(42)
	secondRun := orderOf(42)
	if len(firstRun) != len(subtasks) || len(secondRun) != len(subtasks) {
		t.Fatalf("orders = %v, %v, want %d entries each", firstRun, secondRun, len(subtasks))
	}
	for index := range firstRun {
		if firstRun[index] != secondRun[index] {
			t.Fatalf("shuffled order not deterministic for the same seed/task ID: %v vs %v", firstRun, secondRun)
		}
	}

	differentSeed := orderOf(43)
	identical := true
	for index := range firstRun {
		if firstRun[index] != differentSeed[index] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatalf("orders for seed 42 and 43 are identical (%v); want a different permutation", firstRun)
	}
}

func TestNextTurnMissingFindingsFileFailsTurn(t *testing.T) {
	plansetDir := t.TempDir()
	subtasks := []string{"sub A", "sub B"}
	writePlansetFile(t, plansetDir, 1, subtasks)
	benchmark := newStructuredSubtasksTestBenchmark(t, StructuredSubtasksOptions{PlansetDir: plansetDir, Order: "sequential"})

	sandbox := &structuredSubtasksFakeSandbox{findingsPresent: map[string]bool{
		// /tmp/aries-findings/1_subtask.md is deliberately absent.
	}}
	task := core.Task{ID: "1"}

	if _, ok, err := benchmark.NextTurn(context.Background(), task, 0, sandbox); err != nil || !ok {
		t.Fatalf("NextTurn(turn 0) = (_, %v, %v), want ok", ok, err)
	}
	// Turn 1 must first verify /tmp/aries-findings/1_subtask.md, which is missing.
	if _, ok, err := benchmark.NextTurn(context.Background(), task, 1, sandbox); ok || err == nil {
		t.Fatalf("NextTurn(turn 1) = (_, %v, %v), want ok=false with a non-nil error for a missing findings file", ok, err)
	}
}

func TestNextTurnEmptyFindingsFileFailsTurn(t *testing.T) {
	plansetDir := t.TempDir()
	subtasks := []string{"sub A", "sub B"}
	writePlansetFile(t, plansetDir, 1, subtasks)
	benchmark := newStructuredSubtasksTestBenchmark(t, StructuredSubtasksOptions{PlansetDir: plansetDir, Order: "sequential"})

	// findingsPresent absent for /tmp/aries-findings/1_subtask.md models "exists but
	// empty" identically to "missing": both fail the `test -s` non-empty
	// check verifyAndLockFindings issues.
	sandbox := &structuredSubtasksFakeSandbox{findingsPresent: map[string]bool{}}
	task := core.Task{ID: "1"}
	if _, _, err := benchmark.NextTurn(context.Background(), task, 0, sandbox); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := benchmark.NextTurn(context.Background(), task, 1, sandbox); ok || err == nil {
		t.Fatalf("NextTurn(turn 1) ok=%v err=%v, want ok=false with a non-nil error for an empty findings file", ok, err)
	}
}
