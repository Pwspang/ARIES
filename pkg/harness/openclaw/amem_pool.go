package openclaw

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	"github.com/moby/moby/client"
)

// amemRepoEntry is one repo-scoped amem Qdrant store, shared by every
// Manager (task occurrence) in this process's run whose task targets the
// same repository at the same commit — see amemRepoScopeKey.
type amemRepoEntry struct {
	state *amemQdrantState
	// slug names this repo's export file under CleanupSharedAMEMRepoScope's
	// "<outputRoot>/amem-memory/" directory — see amemRepoSlug.
	slug string
}

// amemRepoRegistry and its mutex are process-wide, not per-Manager: every
// task occurrence in a run gets its own fresh *Manager (internal/app/run.go's
// buildTaskExperiment), so sharing a Qdrant store across occurrences of the
// same repository needs state that outlives any one Manager. This is safe
// specifically because one ARIES process runs exactly one benchmark run —
// see CleanupSharedAMEMRepoScope's doc comment.
var (
	amemRepoRegistryMu sync.Mutex
	amemRepoRegistry   = map[string]*amemRepoEntry{}
)

// CleanupSharedAMEMRepoScope exports and tears down every repo-scoped amem
// Qdrant store created during this process's run (harness.amem.scope:
// "repo"), one per distinct repository+commit the run touched. It is a
// no-op if no repo-scoped store was ever created (repo scope disabled, or
// every task fell back to task scope for lacking repository/commit
// metadata).
//
// Must be called exactly once, after every task occurrence in the run has
// finished — see internal/app/run.go's Wiring.CleanupHarness — never from an
// individual Manager.Close(), which runs once per task occurrence and would
// tear a repo's store down out from under any other task occurrence still
// using it (or has yet to use it: a later task for the same repository would
// otherwise create a second, empty store instead of finding the shared one).
// This is only safe to centralize like this because one ARIES process runs
// exactly one benchmark run: amemRepoRegistry is process-wide state, not
// literally run-scoped, but the two coincide in practice.
//
// Builds its own Docker client rather than reusing any Manager's (each
// Manager's client is closed at that Manager's own Close(), which for the
// task occurrence that happened to create a given repo's store may run long
// before the last task occurrence sharing it finishes).
func CleanupSharedAMEMRepoScope(ctx context.Context, outputRoot string) error {
	if !amemRepoRegistryHasEntries() {
		return nil
	}
	dockerAPI, err := newDefaultDockerClient()
	if err != nil {
		return fmt.Errorf("create Docker client for amem repo-scope cleanup: %w", err)
	}
	defer func() {
		if closer, ok := dockerAPI.(interface{ Close() error }); ok {
			_ = closer.Close()
		}
	}()
	return cleanupSharedAMEMRepoScope(ctx, outputRoot, dockerAPI)
}

func amemRepoRegistryHasEntries() bool {
	amemRepoRegistryMu.Lock()
	defer amemRepoRegistryMu.Unlock()
	return len(amemRepoRegistry) != 0
}

// cleanupSharedAMEMRepoScope is CleanupSharedAMEMRepoScope's core, taking a
// dockerClient explicitly so tests can exercise it against a fake instead of
// a real Docker daemon.
func cleanupSharedAMEMRepoScope(ctx context.Context, outputRoot string, dockerAPI dockerClient) error {
	amemRepoRegistryMu.Lock()
	entries := amemRepoRegistry
	amemRepoRegistry = map[string]*amemRepoEntry{}
	amemRepoRegistryMu.Unlock()

	var errs []error
	for _, entry := range entries {
		exportDir := filepath.Join(outputRoot, "amem-memory", entry.slug)
		if err := exportAMEMQdrantState(ctx, dockerAPI, entry.state, exportDir); err != nil {
			errs = append(errs, fmt.Errorf("export amem repo-scope memory (%s): %w", entry.slug, err))
		}
		if err := teardownAMEMQdrantState(ctx, dockerAPI, entry.state); err != nil {
			errs = append(errs, fmt.Errorf("teardown amem repo-scope store (%s): %w", entry.slug, err))
		}
	}
	return errors.Join(errs...)
}

// newDefaultDockerClient mirrors New()'s own Docker client construction
// (same default socket, since cmd/aries/wiring.go never overrides
// Options.DockerSocket for the openclaw harness today).
func newDefaultDockerClient() (dockerClient, error) {
	host := defaultDockerSocket
	if !strings.Contains(host, "://") {
		host = "unix://" + host
	}
	return client.New(client.WithHost(host), client.WithUserAgent("aries-openclaw/1"))
}
