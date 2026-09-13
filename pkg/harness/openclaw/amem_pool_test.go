package openclaw

import (
	"context"
	"testing"

	"github.com/hyscale-lab/aries/pkg/core"
)

func newRepoScopedTestManager(t *testing.T, fake *fakeDocker) *Manager {
	t.Helper()
	manager, err := New(Options{
		Image: testOpenClawImage, OutputDir: t.TempDir(), AMEMEnabled: true, AMEMQdrantImage: "qdrant/qdrant:v1.19.0",
		AMEMScope: "repo",
	})
	if err != nil {
		t.Fatal(err)
	}
	manager.client = fake
	return manager
}

func TestAMEMRepoScopeSharesQdrantAcrossManagers(t *testing.T) {
	t.Cleanup(resetAMEMRepoRegistry)
	fake := newFakeDocker()
	first := newRepoScopedTestManager(t, fake)
	second := newRepoScopedTestManager(t, fake)

	request := core.HarnessRequest{RunID: "run-1", TaskID: "task-1", Repository: "owner/repo", BaseCommit: "deadbeef"}
	if err := first.ensureAMEMQdrant(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if !first.amemQdrantShared {
		t.Fatal("expected the first manager's amemQdrant to be marked shared under repo scope")
	}

	// A second task occurrence against the same repository+commit (a
	// different task ID, a different Manager — mirroring internal/app/run.go
	// building a fresh Manager per task occurrence) must reuse the first
	// occurrence's Qdrant store rather than creating a second one.
	secondRequest := core.HarnessRequest{RunID: "run-1", TaskID: "task-2", Repository: "owner/repo", BaseCommit: "deadbeef"}
	if err := second.ensureAMEMQdrant(context.Background(), secondRequest); err != nil {
		t.Fatal(err)
	}
	if fake.createCalls != 1 || len(fake.volumesCreated) != 1 || len(fake.networksCreated) != 1 {
		t.Fatalf("repo-scoped ensureAMEMQdrant created its own store instead of sharing: creates=%d volumes=%d networks=%d",
			fake.createCalls, len(fake.volumesCreated), len(fake.networksCreated))
	}
	if second.amemQdrant != first.amemQdrant {
		t.Fatal("second manager did not reuse the first manager's shared amemQdrantState")
	}

	// A different repository (or a different commit of the same repository)
	// must get its own store.
	otherRequest := core.HarnessRequest{RunID: "run-1", TaskID: "task-3", Repository: "owner/other-repo", BaseCommit: "deadbeef"}
	third := newRepoScopedTestManager(t, fake)
	if err := third.ensureAMEMQdrant(context.Background(), otherRequest); err != nil {
		t.Fatal(err)
	}
	if fake.createCalls != 2 {
		t.Fatalf("a distinct repository must get its own Qdrant store: creates = %d, want 2", fake.createCalls)
	}
}

func TestAMEMRepoScopeCloseDoesNotTearDownSharedStore(t *testing.T) {
	t.Cleanup(resetAMEMRepoRegistry)
	fake := newFakeDocker()
	manager := newRepoScopedTestManager(t, fake)
	request := core.HarnessRequest{RunID: "run-1", TaskID: "task-1", Repository: "owner/repo", BaseCommit: "deadbeef"}
	if err := manager.ensureAMEMQdrant(context.Background(), request); err != nil {
		t.Fatal(err)
	}

	if err := manager.exportAMEMMemory(context.Background()); err != nil {
		t.Fatalf("exportAMEMMemory on a shared store must be a no-op, got %v", err)
	}
	if err := manager.teardownAMEMQdrant(context.Background()); err != nil {
		t.Fatal(err)
	}
	if fake.stopCalls != 0 || fake.removeCalls != 0 || len(fake.networksRemoved) != 0 || len(fake.volumesRemoved) != 0 {
		t.Fatalf("a task occurrence's Close must not tear down a repo-scoped shared store: stop=%d remove=%d networksRemoved=%v volumesRemoved=%v",
			fake.stopCalls, fake.removeCalls, fake.networksRemoved, fake.volumesRemoved)
	}
	if manager.amemQdrant != nil {
		t.Fatal("teardownAMEMQdrant should still clear the manager's own reference to the shared state")
	}
	if !amemRepoRegistryHasEntries() {
		t.Fatal("the shared store must remain registered for CleanupSharedAMEMRepoScope after one task occurrence's Close")
	}
}

func TestCleanupSharedAMEMRepoScopeTearsDownEveryRegisteredRepo(t *testing.T) {
	t.Cleanup(resetAMEMRepoRegistry)
	fake := newFakeDocker()
	repoManager := newRepoScopedTestManager(t, fake)
	repoRequest := core.HarnessRequest{RunID: "run-1", TaskID: "task-1", Repository: "owner/repo", BaseCommit: "deadbeef"}
	if err := repoManager.ensureAMEMQdrant(context.Background(), repoRequest); err != nil {
		t.Fatal(err)
	}
	otherManager := newRepoScopedTestManager(t, fake)
	otherRequest := core.HarnessRequest{RunID: "run-1", TaskID: "task-2", Repository: "owner/other-repo", BaseCommit: "deadbeef"}
	if err := otherManager.ensureAMEMQdrant(context.Background(), otherRequest); err != nil {
		t.Fatal(err)
	}

	outputRoot := t.TempDir()
	// The fake's ContainerInspect never populates a NetworkSettings.Networks
	// entry keyed by either store's network name (same gap
	// TestExportAMEMMemoryFailsWithoutNetworkAddress documents for the
	// task-scoped path), so exporting is expected to fail here — this test
	// is only checking that cleanup still tears down both stores and clears
	// the registry despite that.
	if err := cleanupSharedAMEMRepoScope(context.Background(), outputRoot, fake); err == nil {
		t.Fatal("expected an export error given the fake's incomplete network settings")
	}
	if fake.stopCalls != 2 || fake.removeCalls != 2 || len(fake.networksRemoved) != 2 || len(fake.volumesRemoved) != 2 {
		t.Fatalf("cleanup must tear down every registered repo-scoped store: stop=%d remove=%d networksRemoved=%v volumesRemoved=%v",
			fake.stopCalls, fake.removeCalls, fake.networksRemoved, fake.volumesRemoved)
	}
	if amemRepoRegistryHasEntries() {
		t.Fatal("cleanup must clear the repo registry so a later run starts empty")
	}

	// Idempotent: nothing left to clean up the second time.
	if err := cleanupSharedAMEMRepoScope(context.Background(), outputRoot, fake); err != nil {
		t.Fatalf("cleanup with an empty registry must be a no-op, got %v", err)
	}
	if fake.stopCalls != 2 {
		t.Fatalf("cleanup re-ran against an already torn-down registry: stopCalls = %d", fake.stopCalls)
	}
}

func TestAMEMRepoScopeFallsBackToTaskScopeWithoutRepositoryMetadata(t *testing.T) {
	t.Cleanup(resetAMEMRepoRegistry)
	fake := newFakeDocker()
	manager := newRepoScopedTestManager(t, fake)
	// No Repository/BaseCommit set — a task.toml without that metadata (or a
	// benchmark that never sets it) must not be treated as repo-scoped.
	request := core.HarnessRequest{RunID: "run-1", TaskID: "task-1"}
	if err := manager.ensureAMEMQdrant(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if manager.amemQdrantShared {
		t.Fatal("a task with no repository/commit metadata must fall back to task-scoped amem, not repo-scoped")
	}
	if amemRepoRegistryHasEntries() {
		t.Fatal("task-scoped fallback must not register anything in the repo-scope registry")
	}
}

func resetAMEMRepoRegistry() {
	amemRepoRegistryMu.Lock()
	amemRepoRegistry = map[string]*amemRepoEntry{}
	amemRepoRegistryMu.Unlock()
}

func newGlobalScopedTestManager(t *testing.T, fake *fakeDocker) *Manager {
	t.Helper()
	manager, err := New(Options{
		Image: testOpenClawImage, OutputDir: t.TempDir(), AMEMEnabled: true, AMEMQdrantImage: "qdrant/qdrant:v1.19.0",
		AMEMScope: "global",
	})
	if err != nil {
		t.Fatal(err)
	}
	manager.client = fake
	return manager
}

// TestAMEMGlobalScopeSharesQdrantAcrossRepos is the cross-repo counterpart to
// TestAMEMRepoScopeSharesQdrantAcrossManagers's "different repository gets
// its own store" assertion: under "global" scope, two task occurrences
// against different repositories (unlike "repo" scope) must still share one
// store, since the whole point of global scope is testing whether memory
// built up in one repository transfers to a later, different repository.
func TestAMEMGlobalScopeSharesQdrantAcrossRepos(t *testing.T) {
	t.Cleanup(resetAMEMRepoRegistry)
	fake := newFakeDocker()
	first := newGlobalScopedTestManager(t, fake)
	second := newGlobalScopedTestManager(t, fake)

	firstRequest := core.HarnessRequest{RunID: "run-1", TaskID: "task-1", Repository: "owner/repo-a", BaseCommit: "aaaa"}
	if err := first.ensureAMEMQdrant(context.Background(), firstRequest); err != nil {
		t.Fatal(err)
	}
	if !first.amemQdrantShared {
		t.Fatal("expected the first manager's amemQdrant to be marked shared under global scope")
	}

	secondRequest := core.HarnessRequest{RunID: "run-1", TaskID: "task-2", Repository: "owner/repo-b", BaseCommit: "bbbb"}
	if err := second.ensureAMEMQdrant(context.Background(), secondRequest); err != nil {
		t.Fatal(err)
	}
	if fake.createCalls != 1 || len(fake.volumesCreated) != 1 || len(fake.networksCreated) != 1 {
		t.Fatalf("global-scoped ensureAMEMQdrant created its own store instead of sharing across repositories: creates=%d volumes=%d networks=%d",
			fake.createCalls, len(fake.volumesCreated), len(fake.networksCreated))
	}
	if second.amemQdrant != first.amemQdrant {
		t.Fatal("second manager did not reuse the first manager's global-scoped amemQdrantState")
	}

	// A task with no repository/commit metadata at all must still share the
	// same global store — global scope, unlike repo scope, never falls back
	// to task scope for missing metadata.
	third := newGlobalScopedTestManager(t, fake)
	thirdRequest := core.HarnessRequest{RunID: "run-1", TaskID: "task-3"}
	if err := third.ensureAMEMQdrant(context.Background(), thirdRequest); err != nil {
		t.Fatal(err)
	}
	if !third.amemQdrantShared || third.amemQdrant != first.amemQdrant {
		t.Fatal("a task without repository/commit metadata must still join the global-scoped store")
	}
	if fake.createCalls != 1 {
		t.Fatalf("global scope must not create a second store for missing repository metadata: creates = %d, want 1", fake.createCalls)
	}

	// A different run ID must get its own global store.
	otherRunManager := newGlobalScopedTestManager(t, fake)
	otherRunRequest := core.HarnessRequest{RunID: "run-2", TaskID: "task-1", Repository: "owner/repo-a", BaseCommit: "aaaa"}
	if err := otherRunManager.ensureAMEMQdrant(context.Background(), otherRunRequest); err != nil {
		t.Fatal(err)
	}
	if fake.createCalls != 2 {
		t.Fatalf("a distinct run must get its own global Qdrant store: creates = %d, want 2", fake.createCalls)
	}
}

// TestAMEMRepoAndGlobalScopeKeysDiffer guards against a regression that
// would silently collapse "repo" scope into "global" scope (or vice versa):
// for the same run and repository, the two scopes must resolve to different
// registry keys, since "repo" scope is the negative control for the
// cross-repo transfer hypothesis "global" scope exists to test.
func TestAMEMRepoAndGlobalScopeKeysDiffer(t *testing.T) {
	request := core.HarnessRequest{RunID: "run-1", TaskID: "task-1", Repository: "owner/repo", BaseCommit: "deadbeef"}
	repoKey, repoShared := amemRepoScopeKey("repo", request)
	globalKey, globalShared := amemRepoScopeKey("global", request)
	if !repoShared || !globalShared {
		t.Fatalf("both repo and global scope must report shared=true: repo=%v global=%v", repoShared, globalShared)
	}
	if repoKey == globalKey {
		t.Fatal("repo-scope and global-scope keys must differ for the same request")
	}

	// Two different repositories under global scope must resolve to the same
	// key (the whole point of global scope); the same two repositories under
	// repo scope must resolve to different keys (repo scope's existing,
	// unchanged behavior).
	other := core.HarnessRequest{RunID: "run-1", TaskID: "task-2", Repository: "owner/other-repo", BaseCommit: "beefdead"}
	otherRepoKey, _ := amemRepoScopeKey("repo", other)
	otherGlobalKey, _ := amemRepoScopeKey("global", other)
	if otherRepoKey == repoKey {
		t.Fatal("repo scope must not share a key across different repositories")
	}
	if otherGlobalKey != globalKey {
		t.Fatal("global scope must share the same key across different repositories in the same run")
	}
}
