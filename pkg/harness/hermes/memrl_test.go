package hermes

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/hyscale-lab/aries/pkg/core"
)

func sqliteStore(body string) []byte {
	return append(append([]byte(nil), sqliteHeader...), body...)
}

func newMemRLManager(t *testing.T, fake *fakeDocker) *Manager {
	t.Helper()
	manager := newTestManager(t, fake, []byte("model-secret"))
	manager.memrlEnabled = true
	manager.memrlStorePath = filepath.Join(manager.outputDir, "memrl", memrlStoreName)
	return manager
}

func stagedFileContent(t *testing.T, archive []byte, name string) ([]byte, bool) {
	t.Helper()
	reader := tar.NewReader(bytes.NewReader(archive))
	for {
		header, err := reader.Next()
		if err == io.EOF {
			return nil, false
		}
		if err != nil {
			t.Fatal(err)
		}
		if header.Name == name {
			content, err := io.ReadAll(reader)
			if err != nil {
				t.Fatal(err)
			}
			return content, true
		}
	}
}

func TestMemRLDisabledLeavesConfigAndArchiveUnchanged(t *testing.T) {
	fake := newFakeDocker()
	fake.memrlExport = sqliteStore("ignored")
	manager := newTestManager(t, fake, []byte("model-secret"))
	if err := manager.Start(context.Background(), testRequest(t)); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	if _, err := manager.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	config, _ := stagedFileContent(t, fake.archive, strings.TrimPrefix(configContainerPath, "/"))
	if strings.Contains(string(config), "memory:") {
		t.Fatalf("config enables memory:\n%s", config)
	}
	if _, ok := archiveEntries(t, fake.archive)[strings.TrimPrefix(memrlStoreContainerDir, "/")]; ok {
		t.Fatal("archive stages a MemRL directory")
	}
	if len(fake.copyFrom) != 0 {
		t.Fatalf("copied from container: %v", fake.copyFrom)
	}
}

func TestMemRLFirstTaskStartsEmptyAndPersistsExport(t *testing.T) {
	fake := newFakeDocker()
	fake.memrlExport = sqliteStore("after task one")
	manager := newMemRLManager(t, fake)
	request := testRequest(t)
	if err := manager.Start(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	config, _ := stagedFileContent(t, fake.archive, strings.TrimPrefix(configContainerPath, "/"))
	if !strings.Contains(string(config), memrlConfigBlock) {
		t.Fatalf("config lacks memory provider:\n%s", config)
	}
	entries := archiveEntries(t, fake.archive)
	directory, ok := entries[strings.TrimPrefix(memrlStoreContainerDir, "/")]
	if !ok || directory.Uid != runtimeUID || directory.Typeflag != tar.TypeDir {
		t.Fatalf("MemRL directory entry = %#v", directory)
	}
	if _, ok := entries[strings.TrimPrefix(memrlStoreContainerPath, "/")]; ok {
		t.Fatal("first task staged a store that does not exist yet")
	}
	result, err := manager.Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	stored, err := os.ReadFile(manager.memrlStorePath)
	if err != nil || !bytes.Equal(stored, fake.memrlExport) {
		t.Fatalf("run store = %q, %v", stored, err)
	}
	info, err := os.Stat(manager.memrlStorePath)
	if err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("run store mode = %v, %v", info, err)
	}
	artifact := filepath.Join(manager.outputDir, request.TaskID, "harness", "memrl", memrlStoreName)
	if retained, err := os.ReadFile(artifact); err != nil || !bytes.Equal(retained, fake.memrlExport) {
		t.Fatalf("artifact = %q, %v", retained, err)
	}
	found := false
	for _, path := range result.LogPaths {
		found = found || path == artifact
	}
	if !found {
		t.Fatalf("log paths %v lack %s", result.LogPaths, artifact)
	}
}

func TestMemRLNextTaskReceivesRunStore(t *testing.T) {
	fake := newFakeDocker()
	manager := newMemRLManager(t, fake)
	previous := sqliteStore("from the previous task")
	if err := replacePrivateFile(manager.memrlStorePath, previous); err != nil {
		t.Fatal(err)
	}
	if err := manager.Start(context.Background(), testRequest(t)); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	name := strings.TrimPrefix(memrlStoreContainerPath, "/")
	staged, ok := stagedFileContent(t, fake.archive, name)
	if !ok || !bytes.Equal(staged, previous) {
		t.Fatalf("staged store = %q", staged)
	}
	if header := archiveEntries(t, fake.archive)[name]; header.Uid != runtimeUID || header.Mode != 0o600 {
		t.Fatalf("staged store header = %#v", header)
	}
	// The provider never committed (for example, a trivial prompt): the run
	// store must be left as it was.
	if _, err := manager.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if stored, _ := os.ReadFile(manager.memrlStorePath); !bytes.Equal(stored, previous) {
		t.Fatalf("run store changed to %q", stored)
	}
}

func TestMemRLRejectsCorruptStores(t *testing.T) {
	fake := newFakeDocker()
	manager := newMemRLManager(t, fake)
	if err := replacePrivateFile(manager.memrlStorePath, []byte("not a database")); err != nil {
		t.Fatal(err)
	}
	if err := manager.Start(context.Background(), testRequest(t)); err == nil || !strings.Contains(err.Error(), "not an SQLite database") {
		t.Fatalf("start with corrupt store: %v", err)
	}

	fake = newFakeDocker()
	fake.memrlExport = []byte("garbage")
	manager = newMemRLManager(t, fake)
	if err := manager.Start(context.Background(), testRequest(t)); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	if _, err := manager.Run(context.Background(), "task"); err == nil || !strings.Contains(err.Error(), "not an SQLite database") {
		t.Fatalf("run with corrupt export: %v", err)
	}
	if _, err := os.Stat(manager.memrlStorePath); !os.IsNotExist(err) {
		t.Fatalf("corrupt export reached the run store: %v", err)
	}
}

// A canceled one-shot may still be writing, so its store is not trusted.
func TestMemRLSkipsExportWhenRunDidNotExit(t *testing.T) {
	fake := newFakeDocker()
	fake.memrlExport = sqliteStore("possibly torn")
	manager := newMemRLManager(t, fake)
	if err := manager.Start(context.Background(), testRequest(t)); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := manager.Run(ctx, "task"); err == nil {
		t.Fatal("canceled run reported success")
	}
	if len(fake.copyFrom) != 0 {
		t.Fatalf("copied from container after cancellation: %v", fake.copyFrom)
	}
	if _, err := os.Stat(manager.memrlStorePath); !os.IsNotExist(err) {
		t.Fatalf("run store written after cancellation: %v", err)
	}
}

func TestSingleRegularFileRejectsUnexpectedEntries(t *testing.T) {
	build := func(headers ...tar.Header) []byte {
		var archive bytes.Buffer
		writer := tar.NewWriter(&archive)
		for _, header := range headers {
			header.Size = int64(len("x"))
			if header.Typeflag != tar.TypeReg {
				header.Size = 0
			}
			_ = writer.WriteHeader(&header)
			if header.Size > 0 {
				_, _ = writer.Write([]byte("x"))
			}
		}
		_ = writer.Close()
		return archive.Bytes()
	}
	for name, archive := range map[string][]byte{
		"empty":      build(),
		"wrong name": build(tar.Header{Name: "other.db", Typeflag: tar.TypeReg}),
		"symlink":    build(tar.Header{Name: memrlStoreName, Typeflag: tar.TypeSymlink, Linkname: "/etc/passwd"}),
		"two files":  build(tar.Header{Name: memrlStoreName, Typeflag: tar.TypeReg}, tar.Header{Name: memrlStoreName, Typeflag: tar.TypeReg}),
	} {
		if _, err := singleRegularFile(bytes.NewReader(archive), memrlStoreName); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestMemRLStartPassesTaskIDAndStagesRewards(t *testing.T) {
	fake := newFakeDocker()
	manager := newMemRLManager(t, fake)
	rewards := []byte(`{"fix-git-000": 1}` + "\n")
	if err := replacePrivateFile(filepath.Join(filepath.Dir(manager.memrlStorePath), memrlRewardsName), rewards); err != nil {
		t.Fatal(err)
	}
	request := testRequest(t)
	if err := manager.Start(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	env := fake.created.Config.Env
	for _, want := range []string{"MEMRL_REWARD_SOURCE=external", "MEMRL_TASK_ID=" + request.TaskID} {
		if !slices.Contains(env, want) {
			t.Fatalf("env %v lacks %s", env, want)
		}
	}
	staged, ok := stagedFileContent(t, fake.archive, strings.TrimPrefix(memrlStoreContainerDir+"/"+memrlRewardsName, "/"))
	if !ok || !bytes.Equal(staged, rewards) {
		t.Fatalf("staged rewards = %q", staged)
	}
}

func TestMemRLEnvironmentByMode(t *testing.T) {
	base := []string{"MEMRL_REWARD_SOURCE=external", "MEMRL_TASK_ID=t-001"}
	for name, test := range map[string]struct {
		frozen, batch bool
		want          []string
	}{
		"sequential": {want: base},
		"frozen":     {frozen: true, want: append(slices.Clone(base), "MEMRL_FROZEN=1", "MEMRL_EPSILON=0")},
		"batched":    {batch: true, want: append(slices.Clone(base), "MEMRL_BATCH=1")},
	} {
		if got := memrlEnvironment("t-001", test.frozen, test.batch); !slices.Equal(got, test.want) {
			t.Fatalf("%s environment = %v, want %v", name, got, test.want)
		}
	}
}

func TestMemRLStateByMode(t *testing.T) {
	fake := newFakeDocker()
	manager := newMemRLManager(t, fake)
	run, rewards := sqliteStore("run"), []byte(`{"t-001": 1}`+"\n")
	if err := replacePrivateFile(manager.memrlStorePath, run); err != nil {
		t.Fatal(err)
	}
	if err := replacePrivateFile(filepath.Join(filepath.Dir(manager.memrlStorePath), memrlRewardsName), rewards); err != nil {
		t.Fatal(err)
	}
	store, staged, err := manager.readMemRLState()
	if err != nil || !bytes.Equal(store, run) || !bytes.Equal(staged, rewards) {
		t.Fatalf("sequential state = %q, %q, %v", store, staged, err)
	}
	manager.memrlBatch = true
	store, staged, err = manager.readMemRLState()
	if err != nil || !bytes.Equal(store, run) || staged != nil {
		t.Fatalf("batched state = %q, %q, %v: a batch must start from the run store and apply no rewards", store, staged, err)
	}
	manager.memrlBatch = false
	manager.memrlFrozenStore = filepath.Join(t.TempDir(), "trained")
	trained := sqliteStore("trained")
	if err := replacePrivateFile(filepath.Join(manager.memrlFrozenStore, memrlStoreName), trained); err != nil {
		t.Fatal(err)
	}
	store, staged, err = manager.readMemRLState()
	if err != nil || !bytes.Equal(store, trained) || staged != nil {
		t.Fatalf("frozen state = %q, %q, %v: every frozen task must read the trained store", store, staged, err)
	}
	if err := manager.Start(context.Background(), testRequest(t)); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	if !slices.Contains(fake.created.Config.Env, "MEMRL_FROZEN=1") {
		t.Fatalf("env %v lacks MEMRL_FROZEN=1", fake.created.Config.Env)
	}
}

func TestMemRLBatchedTaskQueuesItsSessionAndKeepsTheRunStore(t *testing.T) {
	fake := newFakeDocker()
	manager := newMemRLManager(t, fake)
	manager.memrlBatch = true
	run := sqliteStore("snapshot for this batch")
	if err := replacePrivateFile(manager.memrlStorePath, run); err != nil {
		t.Fatal(err)
	}
	request := testRequest(t)
	parked := []byte(`{"task_id": "` + request.TaskID + `", "session_id": "s"}`)
	fake.copyFiles = map[string][]byte{memrlStoreName: sqliteStore("task's own copy"), memrlPendingName: parked}
	if err := manager.Start(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(fake.created.Config.Env, "MEMRL_BATCH=1") {
		t.Fatalf("env %v lacks MEMRL_BATCH=1", fake.created.Config.Env)
	}
	if _, err := manager.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if err := manager.Stop(context.Background()); err != nil {
		t.Fatal(err)
	}
	if stored, _ := os.ReadFile(manager.memrlStorePath); !bytes.Equal(stored, run) {
		t.Fatalf("a batched task replaced the run store with %q", stored)
	}
	queued, err := os.ReadFile(filepath.Join(filepath.Dir(manager.memrlStorePath), memrlPendingDir, request.TaskID+".json"))
	if err != nil || !bytes.Equal(queued, parked) {
		t.Fatalf("queued session = %q, %v", queued, err)
	}
}

func TestMemRLBatchRejectsAnotherTasksSession(t *testing.T) {
	fake := newFakeDocker()
	manager := newMemRLManager(t, fake)
	manager.memrlBatch = true
	fake.copyFiles = map[string][]byte{memrlStoreName: sqliteStore("copy"), memrlPendingName: []byte(`{"task_id": "someone-else"}`)}
	if err := manager.Start(context.Background(), testRequest(t)); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	if _, err := manager.Run(context.Background(), "task"); err == nil || !strings.Contains(err.Error(), "not a JSON object for this task") {
		t.Fatalf("Run with a foreign parked session: %v", err)
	}
	if entries, _ := os.ReadDir(filepath.Join(filepath.Dir(manager.memrlStorePath), memrlPendingDir)); len(entries) != 0 {
		t.Fatalf("a foreign session was queued: %v", entries)
	}
}

func TestFinalizeMemRLBatchAppliesParkedSessions(t *testing.T) {
	fake := newFakeDocker()
	manager := newMemRLManager(t, fake)
	manager.memrlBatch = true
	runDir := filepath.Dir(manager.memrlStorePath)
	if err := manager.FinalizeMemRLBatch(context.Background(), testRequest(t).Model); err != nil || fake.createCalls != 0 {
		t.Fatalf("an empty batch must do nothing: %v, %d containers", err, fake.createCalls)
	}
	for _, task := range []string{"t-001", "t-002"} {
		if err := replacePrivateFile(filepath.Join(runDir, memrlPendingDir, task+".json"), []byte(`{"task_id": "`+task+`"}`)); err != nil {
			t.Fatal(err)
		}
	}
	if err := replacePrivateFile(filepath.Join(runDir, memrlRewardsName), []byte(`{"t-001": 1, "t-002": -1}`+"\n")); err != nil {
		t.Fatal(err)
	}
	updated := sqliteStore("after the batch")
	fake.copyFiles = map[string][]byte{memrlStoreName: updated}
	if err := manager.FinalizeMemRLBatch(context.Background(), testRequest(t).Model); err != nil {
		t.Fatal(err)
	}
	if fake.finalizeCalls != 1 || !fake.removed {
		t.Fatalf("finalize calls = %d, container removed = %v", fake.finalizeCalls, fake.removed)
	}
	if fake.created.Config.Labels["aries.kind"] != "hermes-memrl-batch" || string(fake.created.HostConfig.NetworkMode) != "bridge" {
		t.Fatalf("batch update container = %#v", fake.created)
	}
	for _, name := range []string{"t-001.json", "t-002.json", "rewards.json"} {
		if _, ok := stagedFileContent(t, fake.archive, strings.TrimPrefix(memrlStoreContainerDir, "/")+"/"+map[bool]string{true: memrlPendingDir + "/", false: ""}[name != "rewards.json"]+name); !ok {
			t.Fatalf("batch update did not stage %s", name)
		}
	}
	if key, ok := stagedFileContent(t, fake.archive, strings.TrimPrefix(modelKeyPath, "/")); !ok || string(key) != "model-secret" {
		t.Fatal("batch update did not stage the model key file")
	}
	if slices.ContainsFunc(fake.created.Config.Env, func(value string) bool { return strings.Contains(value, "model-secret") }) {
		t.Fatal("the model key reached the container environment")
	}
	if stored, _ := os.ReadFile(manager.memrlStorePath); !bytes.Equal(stored, updated) {
		t.Fatalf("run store = %q", stored)
	}
	if entries, _ := os.ReadDir(filepath.Join(runDir, memrlPendingDir)); len(entries) != 0 {
		t.Fatalf("applied sessions still queued: %v", entries)
	}
	for _, name := range []string{"t-001.json", "t-002.json", "finalize.log"} {
		if _, err := os.Stat(filepath.Join(runDir, memrlBatchesDir, "001", name)); err != nil {
			t.Fatalf("batch 001 lacks %s: %v", name, err)
		}
	}
}

func TestFinalizeMemRLBatchFailureKeepsTheQueue(t *testing.T) {
	fake := newFakeDocker()
	manager := newMemRLManager(t, fake)
	runDir := filepath.Dir(manager.memrlStorePath)
	if err := replacePrivateFile(filepath.Join(runDir, memrlPendingDir, "t-001.json"), []byte(`{"task_id": "t-001"}`)); err != nil {
		t.Fatal(err)
	}
	if err := manager.FinalizeMemRLBatch(context.Background(), testRequest(t).Model); err == nil || !strings.Contains(err.Error(), "no recorded rewards") {
		t.Fatalf("batch without rewards: %v", err)
	}
	if err := replacePrivateFile(filepath.Join(runDir, memrlRewardsName), []byte(`{"t-001": 1}`+"\n")); err != nil {
		t.Fatal(err)
	}
	previous := sqliteStore("before")
	if err := replacePrivateFile(manager.memrlStorePath, previous); err != nil {
		t.Fatal(err)
	}
	fake.finalizeExit = 3
	if err := manager.FinalizeMemRLBatch(context.Background(), testRequest(t).Model); err == nil || !strings.Contains(err.Error(), "exited with status 3") {
		t.Fatalf("failed batch update: %v", err)
	}
	if !fake.removed {
		t.Fatal("failed batch update container was not removed")
	}
	if stored, _ := os.ReadFile(manager.memrlStorePath); !bytes.Equal(stored, previous) {
		t.Fatalf("a failed update changed the run store to %q", stored)
	}
	if _, err := os.Stat(filepath.Join(runDir, memrlPendingDir, "t-001.json")); err != nil {
		t.Fatalf("a failed update dropped the queued session: %v", err)
	}
}

func TestMemRLMissingFrozenStoreFailsTheTask(t *testing.T) {
	fake := newFakeDocker()
	manager := newMemRLManager(t, fake)
	manager.memrlFrozenStore = filepath.Join(t.TempDir(), "never-trained")
	if err := manager.Start(context.Background(), testRequest(t)); err == nil || !strings.Contains(err.Error(), "frozen store") {
		t.Fatalf("Start with a missing frozen store: %v", err)
	}
}

func TestMemRLDisabledSetsNoMemRLEnvironment(t *testing.T) {
	fake := newFakeDocker()
	manager := newTestManager(t, fake, []byte("model-secret"))
	if err := manager.Start(context.Background(), testRequest(t)); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	for _, value := range fake.created.Config.Env {
		if strings.HasPrefix(value, "MEMRL_") {
			t.Fatalf("env carries %s", value)
		}
	}
}

func TestRecordMemRLOutcomeMapsVerdicts(t *testing.T) {
	dir := t.TempDir()
	outcomes := map[string]core.Evaluation{
		"pass-001":    {Status: core.StatusSucceeded, Reward: 1},
		"fail-002":    {Status: core.StatusFailed, Reward: 0, Score: 0.9},
		"error-003":   {Status: core.StatusFailed, Error: "judge unavailable"},
		"blocked-004": {Status: core.StatusBlockedIsolation},
		"notrun-005":  {Status: core.StatusNotRun},
	}
	for id, evaluation := range outcomes {
		if err := RecordMemRLOutcome(dir, core.TaskResult{TaskID: id, Evaluation: evaluation}); err != nil {
			t.Fatal(err)
		}
	}
	content, err := readMemRLRewards(filepath.Join(dir, "memrl", memrlRewardsName))
	if err != nil {
		t.Fatal(err)
	}
	var rewards map[string]*float64
	if err := json.Unmarshal(content, &rewards); err != nil {
		t.Fatal(err)
	}
	if len(rewards) != 5 || *rewards["pass-001"] != 1 || *rewards["fail-002"] != -1 {
		t.Fatalf("rewards = %s", content)
	}
	for _, id := range []string{"error-003", "blocked-004", "notrun-005"} {
		if value, ok := rewards[id]; !ok || value != nil {
			t.Fatalf("%s: want an explicit null, got %s", id, content)
		}
	}
	info, err := os.Stat(filepath.Join(dir, "memrl", memrlRewardsName))
	if err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("rewards mode = %v, %v", info, err)
	}
}

func TestRecordMemRLOutcomeRejectsBadInput(t *testing.T) {
	dir := t.TempDir()
	if err := RecordMemRLOutcome(dir, core.TaskResult{TaskID: "../escape"}); err == nil {
		t.Fatal("unsafe task ID accepted")
	}
	if err := replacePrivateFile(filepath.Join(dir, "memrl", memrlRewardsName), []byte("[not an object]")); err != nil {
		t.Fatal(err)
	}
	if err := RecordMemRLOutcome(dir, core.TaskResult{TaskID: "task-001"}); err == nil {
		t.Fatal("corrupt rewards file accepted")
	}
}
