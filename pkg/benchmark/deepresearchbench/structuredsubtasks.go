package deepresearchbench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"

	"github.com/hyscale-lab/aries/pkg/core"
	"github.com/hyscale-lab/aries/pkg/runner"
)

// findingsDir is the fixed, ARIES-owned directory subtask findings files are
// written to and verified/locked from. Like reportPath, it deliberately lives
// outside any benchmark-declared workdir (e.g. "/workspace") so a check run
// via sandbox.Exec with no Dir set resolves the same absolute path the agent
// itself was instructed to use, regardless of what working directory the
// agent's own shell session defaults to.
const findingsDir = "/tmp/aries-findings"

// findingsPath returns the fixed absolute path for one subtask's findings
// file (1-indexed, matching subtaskInstruction's numbering).
func findingsPath(index int) string {
	return fmt.Sprintf("%s/%d_subtask.md", findingsDir, index)
}

// outputFormatMarker is the heading synthesisInstruction slices
// taskPromptTemplate at, to reuse its "Output format"/citation/URL/"Why this
// matters"/"Length"/"Rules" sections verbatim in the structured-subtasks
// pass's final synthesis turn. A build-time panic (not a returned error) is
// deliberate here: this can only fail if taskPromptTemplate's own text
// changes out from under this slice, a programmer error caught immediately
// by any test that exercises synthesisInstruction, not a runtime condition
// callers need to handle.
const outputFormatMarker = "## Output format"

// subtaskInstruction builds the minimal, isolated prompt for one structured-
// subtasks subtask turn (1-indexed, matching the findings/{index}_subtask.md
// filename convention). It deliberately omits the original research
// question, the full subtask list, and taskPromptTemplate's citation-format
// rules — those apply only to the final synthesis report, not to a single
// subtask's findings notes.
func subtaskInstruction(subtask string, index int) string {
	var priorFindingsNote string
	if index > 1 {
		priorFindingsNote = fmt.Sprintf(
			"%s through %s already exist in\n"+
				"this sandbox (from earlier subtasks) and may be read for context, e.g.\n"+
				"with `cat`.\n\n", findingsPath(1), findingsPath(index-1))
	}
	return "" +
		"You are an autonomous research analyst carrying out one step of a larger\n" +
		"DeepResearch-Bench research plan in a Linux container.\n" +
		"\n" +
		"## Sub-question\n" +
		"\n" +
		subtask + "\n" +
		"\n" +
		priorFindingsNote +
		"## Task\n" +
		"\n" +
		"1. Investigate this sub-question by searching the web with the search tool\n" +
		"   and reading sources with your web fetch/extract tool (e.g.\n" +
		"   `web_fetch`/`web_extract`).\n" +
		fmt.Sprintf("2. Write your findings and analysis for this sub-question to\n   `%s`.\n", findingsPath(index)) +
		"3. When finished, reply with a single line: `DONE`.\n" +
		"\n" +
		"## Rules\n" +
		"\n" +
		"- Answer only this sub-question. Do not restate or answer the overall\n" +
		"  research question, and do not write the final report.\n" +
		fmt.Sprintf("- You must actually invoke a tool (e.g. a shell/exec command that creates\n  the file) to write %s in this same turn — a message\n  that only states an intention to write it does not satisfy this\n  requirement.\n", findingsPath(index)) +
		"- Reply `DONE` as a standalone line, with nothing else, once the findings\n" +
		"  file is written."
}

// synthesisInstruction is the structured-subtasks pass's final turn: it
// reuses taskPromptTemplate's synthesis/citation/report-contract text
// (everything from "## Output format" onward, including the citation
// format, URL rules, "Why this matters", "Length", and "Rules" sections)
// verbatim, so the strict citation contract FACT depends on is unchanged.
// Only the preceding "go do the research" framing is replaced: this turn
// reads already-gathered findings/*.md files instead of searching the web.
// reportInstruction (verbatim) is still appended, matching taskPromptTemplate.
func synthesisInstruction() string {
	index := strings.Index(taskPromptTemplate, outputFormatMarker)
	if index < 0 {
		panic("deepresearchbench: taskPromptTemplate no longer contains " + outputFormatMarker)
	}
	citationAndRules := taskPromptTemplate[index:]
	header := "" +
		"You are an autonomous research analyst finishing a DeepResearch-Bench task in\n" +
		"a Linux container. Earlier turns already researched each sub-question of this\n" +
		"task's research plan and left their findings in " + findingsDir + "/*.md files\n" +
		"already present in this sandbox.\n" +
		"\n" +
		"## Task\n" +
		"\n" +
		"1. Read every " + findingsDir + "/*.md file already present in the sandbox — do\n" +
		"   not search the web or fetch new pages; the research is already done.\n" +
		"2. Synthesize the evidence into a coherent long-form markdown report at\n" +
		"   `" + reportPath + "`, with **inline citations** to the source URLs cited\n" +
		"   in the findings.\n" +
		"3. When the report is complete, reply with a single line: `DONE`.\n" +
		"\n"
	return header + citationAndRules + reportInstruction
}

// structuredSubtasksOrderArtifact is the host-side record of the effective
// (possibly shuffled) per-task subtask order, written once per task the
// first time subtasksFor computes it, so a shuffled arm's actual order is
// reproducible/inspectable after the run without a dedicated logging
// framework.
type structuredSubtasksOrderArtifact struct {
	NumericTaskID int      `json:"numeric_task_id"`
	Order         string   `json:"order"`
	Seed          int64    `json:"seed,omitempty"`
	Subtasks      []string `json:"subtasks"`
}

// subtasksFor loads, orders (per Options.StructuredSubtasks.Order), and
// caches the ordered subtask list for task.ID, so repeated NextTurn calls
// for the same task see a stable list without re-reading or re-shuffling.
// Only called when b.structuredSubtasks != nil.
func (b *Benchmark) subtasksFor(task core.Task) ([]string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if cached, ok := b.subtaskPlans[task.ID]; ok {
		return cached, nil
	}
	numericID, ok := b.numericIDs[task.ID]
	if !ok {
		return nil, fmt.Errorf("deepresearchbench task %q was not loaded by Tasks", task.ID)
	}

	plansetPath := filepath.Join(b.structuredSubtasks.PlansetDir, fmt.Sprintf("%d.json", numericID))
	raw, err := os.ReadFile(plansetPath)
	if err != nil {
		return nil, fmt.Errorf("read structured-subtasks plan %q: %w", plansetPath, err)
	}
	subtasks, err := validatePlan(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid structured-subtasks plan %q: %w", plansetPath, err)
	}

	switch b.structuredSubtasks.Order {
	case "shuffled":
		// Seeded from the configured seed combined with this task's own
		// numeric ID, so every task in the run gets its own deterministic
		// (but distinct) permutation, reproducible across runs given the
		// same seed and plan file.
		source := rand.NewPCG(uint64(b.structuredSubtasks.Seed), uint64(numericID))
		rng := rand.New(source)
		rng.Shuffle(len(subtasks), func(i, j int) { subtasks[i], subtasks[j] = subtasks[j], subtasks[i] })
	case "adversarial":
		// Deterministic full reversal, not a permutation seeded like
		// "shuffled": for a planset written as a cumulative build-up chain
		// (each subtask's prompt tells it earlier findings/*.md files "may
		// be read for context", see subtaskInstruction), reversing the plan
		// file order guarantees every dependent subtask now runs before the
		// findings it would have relied on, rather than relying on a random
		// seed to accidentally invert a dependency.
		for i, j := 0, len(subtasks)-1; i < j; i, j = i+1, j-1 {
			subtasks[i], subtasks[j] = subtasks[j], subtasks[i]
		}
	}

	if err := b.writeSubtaskOrderArtifact(task, numericID, subtasks); err != nil {
		return nil, err
	}

	if b.subtaskPlans == nil {
		b.subtaskPlans = make(map[string][]string)
	}
	b.subtaskPlans[task.ID] = subtasks
	return subtasks, nil
}

func (b *Benchmark) writeSubtaskOrderArtifact(task core.Task, numericID int, subtasks []string) error {
	artifactDir := filepath.Join(b.outputDir, task.ID)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		return fmt.Errorf("create structured-subtasks artifact directory: %w", err)
	}
	artifact := structuredSubtasksOrderArtifact{
		NumericTaskID: numericID,
		Order:         b.structuredSubtasks.Order,
		Subtasks:      subtasks,
	}
	if b.structuredSubtasks.Order == "shuffled" {
		artifact.Seed = b.structuredSubtasks.Seed
	}
	encoded, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return fmt.Errorf("encode structured-subtasks order artifact: %w", err)
	}
	path := filepath.Join(artifactDir, "structured_subtasks_order.json")
	if err := os.WriteFile(path, encoded, 0o600); err != nil {
		return fmt.Errorf("write structured-subtasks order artifact %q: %w", path, err)
	}
	return nil
}

// verifyAndLockFindings confirms path (an absolute findingsPath(...) result)
// exists and is non-empty, then locks it read-only (chmod 0444) via a direct
// sandbox.Exec call — not the agent — so that even if a later subtask's agent
// session tries to edit an earlier findings file, the file is read-only
// inside the same container. path must be absolute: sandbox.Exec runs with no
// Dir set, so a relative path would resolve against whatever default working
// directory the exec transport uses, which is not guaranteed to match the
// workdir the agent's own shell session defaults to (see findingsDir). A
// missing/empty findings file is returned as an error so NextTurn's caller
// (the Runner) treats it as a turn failure and skips straight to cleanup,
// per the existing "fail loudly rather than silently continue" convention.
func verifyAndLockFindings(ctx context.Context, sandbox runner.Sandbox, path string) error {
	// path is always one of our own absolute findingsPath(...) values, never
	// agent-controlled, so it never begins with "-" — "--" before it is
	// deliberately omitted: this task image's /bin/sh (dash/busybox ash)
	// rejects "--" as an operand to the "test" builtin ("unexpected
	// operator"), unlike GNU coreutils' standalone test/[ binary.
	const nonEmptyPredicate = `test -s "$1"`
	checked, err := sandbox.Exec(ctx, core.Command{
		Path: "/bin/sh", Args: []string{"-c", nonEmptyPredicate, "aries-findings-check", path},
	})
	if err != nil {
		return fmt.Errorf("check findings file %q: %w", path, err)
	}
	if checked.ExitCode != 0 {
		return fmt.Errorf("findings file %q is missing or empty after its subtask turn", path)
	}
	locked, err := sandbox.Exec(ctx, core.Command{Path: "/bin/chmod", Args: []string{"0444", path}})
	if err != nil {
		return fmt.Errorf("lock findings file %q read-only: %w", path, err)
	}
	if locked.ExitCode != 0 {
		return fmt.Errorf("lock findings file %q read-only: exit code %d", path, locked.ExitCode)
	}
	return nil
}

// NextTurn implements runner.MultiTurnBenchmark. When
// Options.StructuredSubtasks is nil, *Benchmark still structurally satisfies
// the interface (it must, statically — Go has no way to implement it only
// conditionally), but behaves as a strict one-turn generalization of the
// original single-cycle Runner path: turn 0 runs task.Instruction exactly
// once, then the loop ends and Evaluate proceeds exactly as it always has.
// This keeps ordinary single-turn runs (and the plan-generation pass, which
// is also single-turn) byte-for-byte behaviorally identical to before this
// method existed.
//
// When Options.StructuredSubtasks is set, turns 0..N-1 (N = number of
// subtasks) are isolated per-subtask sessions (see subtaskInstruction),
// turn N is the final synthesis session (see synthesisInstruction), and
// turn N+1 ends the loop. Before returning the instruction for turn k in
// [1, N], the previous subtask's findings file is verified non-empty and
// locked read-only (see verifyAndLockFindings) — a missing/empty findings
// file fails the turn instead of silently continuing.
func (b *Benchmark) NextTurn(ctx context.Context, task core.Task, turnIndex int, sandbox runner.Sandbox) (string, bool, error) {
	if b.structuredSubtasks == nil {
		if turnIndex == 0 {
			return task.Instruction, true, nil
		}
		return "", false, nil
	}
	if sandbox == nil {
		return "", false, errors.New("deepresearchbench structured subtasks requires a live sandbox")
	}

	subtasks, err := b.subtasksFor(task)
	if err != nil {
		return "", false, err
	}
	subtaskCount := len(subtasks)

	if turnIndex >= 1 && turnIndex <= subtaskCount {
		if err := verifyAndLockFindings(ctx, sandbox, findingsPath(turnIndex)); err != nil {
			return "", false, err
		}
	}

	switch {
	case turnIndex < subtaskCount:
		return subtaskInstruction(subtasks[turnIndex], turnIndex+1), true, nil
	case turnIndex == subtaskCount:
		return synthesisInstruction(), true, nil
	default:
		return "", false, nil
	}
}
