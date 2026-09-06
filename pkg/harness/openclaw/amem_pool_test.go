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
