package sweatlas

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
)

func TestPreparedCatalogForksConcurrentOccurrencesWithoutReload(t *testing.T) {
	root := writeFixture(t)
	catalog, err := New(testOptions(root, []string{qaTaskID}, t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}
	tasks, err := catalog.Tasks(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	logicalID := tasks[0].ID
	if _, err := catalog.ForTask(logicalID, "../invalid"); err == nil {
		t.Fatal("unsafe occurrence accepted")
	}
	if _, err := catalog.ForTask("missing", "missing-001"); err == nil {
		t.Fatal("unloaded task accepted")
	}
	// A second source read or revision check during occurrence creation would fail.
	// Evaluation still performs its own pin check before touching private files.
	if err := os.RemoveAll(root); err != nil {
		t.Fatal(err)
	}
	var group sync.WaitGroup
	for i := range 8 {
		group.Add(1)
		go func() {
			defer group.Done()
			id := fmt.Sprintf("%s-%03d", logicalID, i+1)
			occurrence, err := catalog.ForTask(logicalID, id)
			if err != nil {
				t.Error(err)
				return
			}
			owner := occurrence.(*Benchmark)
			selected, err := owner.Tasks(context.Background())
			if err != nil || len(selected) != 1 || selected[0].ID != id {
				t.Errorf("occurrence tasks = %v, %v", selected, err)
				return
			}
			if owner.details[id].testsDir == "" {
				t.Error("private evaluation inputs lost")
			}
			encoded, err := json.Marshal(selected[0])
			if err != nil {
				t.Error(err)
				return
			}
			if strings.Contains(string(encoded), "rubrics.json") {
				t.Error("public task leaked verifier material")
			}
		}()
	}
	group.Wait()
}

// InstructionAppend reaches the harness's instruction for every occurrence and
// never reaches the judge, whose question comes from the private tests/ tree.
func TestInstructionAppendReachesHarnessNotJudge(t *testing.T) {
	const appended = "Use coderag__search_code to locate code."
	root := writeFixture(t)
	options := testOptions(root, []string{qaTaskID}, t.TempDir())
	options.InstructionAppend = "  " + appended + "\n"
	catalog, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	tasks, err := catalog.Tasks(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := strings.TrimSpace(qaInstruction) + "\n\n" + appended
	if tasks[0].Instruction != want {
		t.Fatalf("Instruction = %q, want %q", tasks[0].Instruction, want)
	}
	occurrence, err := catalog.ForTask(tasks[0].ID, tasks[0].ID+"-001")
	if err != nil {
		t.Fatal(err)
	}
	owner := occurrence.(*Benchmark)
	selected, err := owner.Tasks(context.Background())
	if err != nil || selected[0].Instruction != want {
		t.Fatalf("occurrence instruction = %v, %v", selected, err)
	}

	chat := &stubChat{responses: []string{ratingResponse("YES"), ratingResponse("YES")}}
	owner.judge = chat
	if _, err := owner.Evaluate(context.Background(), selected[0], &evaluateFake{downloadContent: finalAnswerTag + "\nthe answer\n"}, nil); err != nil {
		t.Fatal(err)
	}
	if chat.calls == 0 {
		t.Fatal("judge was not called")
	}
	for _, prompt := range chat.prompts {
		if strings.Contains(prompt, appended) {
			t.Fatalf("judge prompt contains the harness-only text: %q", prompt)
		}
	}
}

func TestTasksKeepInstructionVerbatimByDefault(t *testing.T) {
	root := writeFixture(t)
	catalog, err := New(testOptions(root, []string{qaTaskID}, t.TempDir()))
	if err != nil {
		t.Fatal(err)
	}
	tasks, err := catalog.Tasks(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if tasks[0].Instruction != strings.TrimSpace(qaInstruction) {
		t.Fatalf("Instruction = %q, want instruction.md verbatim", tasks[0].Instruction)
	}
}
