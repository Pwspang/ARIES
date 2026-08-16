package sweatlas

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/hyscale-lab/aries/pkg/core"
	"github.com/hyscale-lab/aries/pkg/runner"
)

// evaluationResults is the informational rubric breakdown SWE-Atlas QA's
// verifier writes to /logs/verifier/evaluation_results.json alongside
// reward.txt. Only AggScore is consumed for core.Evaluation.Score; reward.txt
// alone drives core.Evaluation.Reward and Status, exactly as
// terminalbench's ctrf.json is a secondary cross-check relative to its own
// reward.txt.
type evaluationResults struct {
	AggScore float64 `json:"agg_score"`
}

// Evaluate injects private verifier material (the tests/ tree) and the
// resolved judge model environment into the still-live sandbox, and runs the
// LLM-judge-backed verifier independently of the harness.
func (b *Benchmark) Evaluate(ctx context.Context, task core.Task, sandbox runner.Sandbox) (core.Evaluation, error) {
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
		return finish(errors.New("sweatlas evaluator requires a live sandbox"))
	}
	b.mu.RLock()
	details, ok := b.details[task.ID]
	b.mu.RUnlock()
	if !ok {
		return finish(fmt.Errorf("sweatlas task %q was not loaded by Tasks", task.ID))
	}
	if err := VerifyRevision(ctx, b.root, b.revision); err != nil {
		return finish(fmt.Errorf("reverify sweatlas checkout before evaluation: %w", err))
	}

	apiKey, ok := b.apiKeyLookup(b.judge.APIKeyEnv)
	if !ok {
		return finish(fmt.Errorf("resolve sweatlas judge API key %q", b.judge.APIKeyEnv))
	}
	verifierEnv := map[string]string{
		"EVAL_API_KEY":  string(apiKey),
		"EVAL_BASE_URL": b.judge.BaseURL,
		"EVAL_MODEL":    b.judge.Model,
	}

	artifactDir := filepath.Join(b.outputDir, task.ID, "evaluation")
	if err := os.MkdirAll(artifactDir, 0o700); err != nil {
		return finish(fmt.Errorf("create evaluator artifact directory: %w", err))
	}
	stdoutPath := filepath.Join(artifactDir, "stdout.log")
	stderrPath := filepath.Join(artifactDir, "stderr.log")
	rewardPath := filepath.Join(artifactDir, "reward.txt")
	resultsPath := filepath.Join(artifactDir, "evaluation_results.json")
	evaluation.LogPaths = []string{stdoutPath, stderrPath, rewardPath, resultsPath}
	for _, path := range evaluation.LogPaths {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return finish(fmt.Errorf("remove stale evaluator artifact %q: %w", path, err))
		}
	}

	resetResult, err := sandbox.Exec(ctx, core.Command{
		Path: "/bin/rm",
		Args: []string{"-rf", "--", testsPath, verifierLogPath},
	})
	if err != nil {
		return finish(fmt.Errorf("remove stale verifier paths: %w", err))
	}
	if resetResult.ExitCode != 0 {
		return finish(fmt.Errorf("remove stale verifier paths: exit code %d", resetResult.ExitCode))
	}
	directories := []string{testsPath, verifierLogPath}
	seenDirectories := map[string]struct{}{testsPath: {}, verifierLogPath: {}}
	for _, file := range details.verifierFiles {
		for directory := path.Dir(file.destination); directory != testsPath; directory = path.Dir(directory) {
			if _, seen := seenDirectories[directory]; !seen {
				seenDirectories[directory] = struct{}{}
				directories = append(directories, directory)
			}
		}
	}
	slices.Sort(directories[2:])
	createResult, err := sandbox.Exec(ctx, core.Command{
		Path: "/bin/mkdir",
		Args: append([]string{"-p", "--"}, directories...),
	})
	if err != nil {
		return finish(fmt.Errorf("create clean verifier directories: %w", err))
	}
	if createResult.ExitCode != 0 {
		return finish(fmt.Errorf("create clean verifier directories: exit code %d", createResult.ExitCode))
	}
	for _, file := range details.verifierFiles {
		if err := sandbox.Upload(ctx, file.source, file.destination); err != nil {
			return finish(fmt.Errorf("inject private verifier file %q: %w", file.name, err))
		}
	}

	commandResult, commandErr := sandbox.Exec(ctx, core.Command{
		Path:    "/bin/bash",
		Args:    []string{filepath.Join(testsPath, "test.sh")},
		Dir:     details.workdir,
		Env:     verifierEnv,
		Timeout: details.timeout,
	})
	var artifactErrors []error
	if err := os.WriteFile(stdoutPath, []byte(commandResult.Stdout), 0o600); err != nil {
		artifactErrors = append(artifactErrors, fmt.Errorf("write verifier stdout: %w", err))
	}
	if err := os.WriteFile(stderrPath, []byte(commandResult.Stderr), 0o600); err != nil {
		artifactErrors = append(artifactErrors, fmt.Errorf("write verifier stderr: %w", err))
	}
	if err := sandbox.Download(ctx, filepath.Join(verifierLogPath, "reward.txt"), rewardPath); err != nil {
		artifactErrors = append(artifactErrors, fmt.Errorf("download verifier reward: %w", err))
	}
	if err := sandbox.Download(ctx, filepath.Join(verifierLogPath, "evaluation_results.json"), resultsPath); err != nil {
		artifactErrors = append(artifactErrors, fmt.Errorf("download verifier evaluation results: %w", err))
	}

	if commandErr != nil {
		artifactErrors = append(artifactErrors, fmt.Errorf("run verifier: %w", commandErr))
	}
	if commandResult.ExitCode != 0 {
		artifactErrors = append(artifactErrors, fmt.Errorf("run verifier: exit code %d", commandResult.ExitCode))
	}
	if len(artifactErrors) != 0 {
		return finish(errors.Join(artifactErrors...))
	}

	reward, err := parseRewardFile(rewardPath)
	if err != nil {
		return finish(err)
	}
	evaluation.Reward = reward

	aggScore, err := parseAggScore(resultsPath)
	if err != nil {
		return finish(err)
	}
	evaluation.Score = aggScore

	if reward == 1 {
		evaluation.Status = core.StatusSucceeded
		evaluation.VerifierStatus = core.StatusSucceeded
		return finish(nil)
	}
	return finish(nil)
}

func parseRewardFile(path string) (float64, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return 0, fmt.Errorf("read verifier reward: %w", err)
	}
	switch strings.TrimSpace(string(content)) {
	case "1":
		return 1, nil
	case "0":
		return 0, nil
	default:
		return 0, fmt.Errorf("malformed verifier reward in %q: expected 1 or 0", path)
	}
}

// parseAggScore decodes the verifier's evaluation_results.json and validates
// its agg_score is a finite value in [0, 1] (the mathematically guaranteed
// range for an average of per-rubric 0/1 scores, per evaluate_answer.py). A
// badly-behaved judge script producing an out-of-range score is a real fault
// that must surface as an evaluation error rather than be silently clamped.
func parseAggScore(path string) (float64, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return 0, fmt.Errorf("read verifier evaluation results: %w", err)
	}
	var results evaluationResults
	if err := json.Unmarshal(content, &results); err != nil {
		return 0, fmt.Errorf("parse verifier evaluation results %q: %w", path, err)
	}
	if math.IsNaN(results.AggScore) || math.IsInf(results.AggScore, 0) || results.AggScore < 0 || results.AggScore > 1 {
		return 0, fmt.Errorf("malformed verifier agg_score in %q: %g is not finite and in [0, 1]", path, results.AggScore)
	}
	return results.AggScore, nil
}
