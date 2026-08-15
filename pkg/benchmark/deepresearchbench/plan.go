package deepresearchbench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/hyscale-lab/aries/pkg/core"
	"github.com/hyscale-lab/aries/pkg/runner"
)

// evaluatePlan is Evaluate's plan-generation-pass counterpart (see
// Options.PlanOnly): it never runs RACE/FACT. Instead it downloads planPath,
// validates it's a JSON array of at least one non-empty-after-trim string,
// and — only on success — persists the canonical (trimmed) plan host-side at
// "<PlansetDir>/<numeric_task_id>.json" for the later structured-subtasks
// pass to read. A missing, unparseable, or empty/invalid plan fails this
// task's "evaluation" loudly (Status=Failed, a clear Error message) rather
// than degrading silently, since a broken plan invisibly propagating into
// both structured-execution arms would invalidate the whole comparison.
//
// This is "evaluation" in name only: there is no RACE/FACT judgement here, so
// Score/Reward stay at their zero value regardless of outcome — Status is
// the only signal that matters, and it reflects whether plan capture itself
// succeeded.
func (b *Benchmark) evaluatePlan(ctx context.Context, task core.Task, sandbox runner.Sandbox) (core.Evaluation, error) {
	started := time.Now()
	evaluation := core.Evaluation{Status: core.StatusFailed, VerifierStatus: core.StatusFailed}
	finish := func(err error) (core.Evaluation, error) {
		evaluation.Duration = time.Since(started)
		if err != nil {
			evaluation.Error = err.Error()
		}
		return evaluation, err
	}

	if sandbox == nil {
		return finish(errors.New("deepresearchbench plan evaluator requires a live sandbox"))
	}
	b.mu.RLock()
	numericID, ok := b.numericIDs[task.ID]
	b.mu.RUnlock()
	if !ok {
		return finish(fmt.Errorf("deepresearchbench task %q was not loaded by Tasks", task.ID))
	}

	artifactDir := filepath.Join(b.outputDir, task.ID, "evaluation")
	if err := os.MkdirAll(artifactDir, 0o700); err != nil {
		return finish(fmt.Errorf("create evaluator artifact directory: %w", err))
	}
	planArtifactPath := filepath.Join(artifactDir, "plan.json")
	if err := os.Remove(planArtifactPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return finish(fmt.Errorf("remove stale plan artifact %q: %w", planArtifactPath, err))
	}
	evaluation.LogPaths = []string{planArtifactPath}

	// A missing plan is a legitimate (bad) task outcome, not a plumbing
	// failure — an agent that misbehaves or times out never writes the file.
	// Score it as a failed "evaluation" without surfacing a Go error, the
	// same treatment Evaluate gives a missing report.
	if err := sandbox.Download(ctx, planPath, planArtifactPath); err != nil {
		evaluation.Error = fmt.Sprintf("agent did not produce a plan at %s: %v", planPath, err)
		evaluation.Duration = time.Since(started)
		return evaluation, nil
	}

	raw, err := os.ReadFile(planArtifactPath)
	if err != nil {
		return finish(fmt.Errorf("read downloaded plan: %w", err))
	}

	subtasks, err := validatePlan(raw)
	if err != nil {
		evaluation.Error = fmt.Sprintf("invalid plan at %s: %v", planPath, err)
		evaluation.Duration = time.Since(started)
		return evaluation, nil
	}

	if err := os.MkdirAll(b.plansetDir, 0o755); err != nil {
		return finish(fmt.Errorf("create planset directory %q: %w", b.plansetDir, err))
	}
	canonical, err := json.MarshalIndent(subtasks, "", "  ")
	if err != nil {
		return finish(fmt.Errorf("encode canonical plan: %w", err))
	}
	plansetPath := filepath.Join(b.plansetDir, fmt.Sprintf("%d.json", numericID))
	if err := os.WriteFile(plansetPath, canonical, 0o600); err != nil {
		return finish(fmt.Errorf("write planset artifact %q: %w", plansetPath, err))
	}
	evaluation.LogPaths = append(evaluation.LogPaths, plansetPath)

	evaluation.Status = core.StatusSucceeded
	evaluation.VerifierStatus = core.StatusSucceeded
	return finish(nil)
}

// validatePlan decodes raw as a JSON array of strings and requires at least
// one element, all non-empty after trimming surrounding whitespace. It
// returns the trimmed strings (the canonical form persisted to the planset
// directory and later read back by the structured-subtasks pass), never the
// untrimmed originals.
func validatePlan(raw []byte) ([]string, error) {
	var subtasks []string
	if err := json.Unmarshal(raw, &subtasks); err != nil {
		return nil, fmt.Errorf("parse plan as a JSON array of strings: %w", err)
	}
	if len(subtasks) == 0 {
		return nil, errors.New("plan must contain at least one sub-question")
	}
	trimmed := make([]string, len(subtasks))
	for index, subtask := range subtasks {
		clean := strings.TrimSpace(subtask)
		if clean == "" {
			return nil, fmt.Errorf("plan sub-question %d is empty or whitespace-only", index)
		}
		trimmed[index] = clean
	}
	return trimmed, nil
}
