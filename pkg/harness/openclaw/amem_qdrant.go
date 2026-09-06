package openclaw

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/containerd/errdefs"
	"github.com/hyscale-lab/aries/pkg/core"
	"github.com/moby/moby/api/types/container"
	"github.com/moby/moby/api/types/mount"
	"github.com/moby/moby/api/types/network"
	"github.com/moby/moby/client"
)

const (
	// amemQdrantLoopbackPort is both Qdrant's own listening port and the port
	// amem's plugin talks to on localhost — its Qdrant client is hardcoded to
	// http://localhost:6333 in its own source (confirmed by inspecting its
	// bundled dist/index.js: no env var or config field overrides it), so the
	// per-task loopback proxy started by gatewayLauncherScript forwards this
	// exact port.
	amemQdrantLoopbackPort = "6333"
	// amemQdrantNetworkAlias is the fixed Docker network alias the Qdrant
	// container (see ensureAMEMQdrant) registers on the shared amem network,
	// and what the loopback proxy forwards to.
	amemQdrantNetworkAlias = "amem-qdrant"
	// amemQdrantCollectionName is the plugin's own default Qdrant collection
	// name (confirmed against a live gateway boot log: "default
	// collection=amem_notes (default)"). amemPluginConfig deliberately never
	// sets config.collection, so this stays accurate as long as that remains
	// true — update both together if that ever changes.
	amemQdrantCollectionName = "amem_notes"
	// amemMemoryExportTimeout bounds exportAMEMMemory's Qdrant scroll calls —
	// generous since this runs once at the very end of a task (or, under repo
	// scope, once per repository at the very end of a run), not per task.
	amemMemoryExportTimeout = 2 * time.Minute
)

// amemQdrantState holds the Qdrant resources ensureAMEMQdrant creates, so
// Manager.Close (exportAMEMMemory, teardownAMEMQdrant) can read and then
// remove them once this task occurrence ends. nil until amem is enabled and
// this Manager's one Start() call runs.
type amemQdrantState struct {
	networkID   string
	networkName string
	volumeName  string
	containerID string
}

// amemRepoScopeKey returns the repo-scope registry key for request and true
// when manager.amemScope selects sharing a Qdrant store across every task
// occurrence in this run that targets the same repository at the same
// commit — false (with an empty key) falls back to today's task-scoped
// behavior, including when the task's metadata carries no repository/commit
// identity at all (older or malformed task.toml), since sharing memory is
// only correct between tasks looking at identical code.
func amemRepoScopeKey(scope string, request core.HarnessRequest) (string, bool) {
	if scope != "repo" || request.Repository == "" || request.BaseCommit == "" {
		return "", false
	}
	hash := sha256.Sum256([]byte(request.RunID + "\x00repo\x00" + request.Repository + "\x00" + request.BaseCommit))
	return hex.EncodeToString(hash[:])[:16], true
}

// amemRepoSlug turns a task's repository+commit identity into a filesystem-
// safe name for its repo-scoped memory export (see CleanupSharedAMEMRepoScope
// in amem_pool.go).
func amemRepoSlug(request core.HarnessRequest) string {
	safe := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			return r
		default:
			return '-'
		}
	}, request.Repository)
	commit := request.BaseCommit
	if len(commit) > 12 {
		commit = commit[:12]
	}
	return safe + "-" + commit
}

// ensureAMEMQdrant idempotently gives this task occurrence a Qdrant memory
// store: task-scoped (a private volume/network/container keyed on
// runID+taskID) by default, or — when manager.amemScope is "repo" and the
// task carries repository/base_commit metadata — a store shared with every
// other task occurrence in this run against the same repository at the same
// commit, so a later task's agent can see notes an earlier one already
// stored about that codebase instead of re-discovering it from scratch.
//
// Task scoping was the original design (see git history on this file): a
// run-scoped resource name keyed only on runID collided the moment a second
// occurrence's fresh Manager (every task occurrence gets its own *Manager —
// internal/app/run.go's buildTaskExperiment) tried to create the same
// network name, and even a fix tolerating "already exists" would still be
// wrong because teardownAMEMQdrant/exportAMEMMemory run once per
// Manager.Close (also once per occurrence). Repo scope sidesteps the same
// hazard the same way task scope does — via a process-wide, mutex-protected
// registry (amemRepoRegistry in amem_pool.go) instead of per-Manager state —
// and defers teardown/export to CleanupSharedAMEMRepoScope, called once after
// every task occurrence in the run has finished (internal/app/run.go via
// Wiring.CleanupHarness), rather than from this Manager's own Close().
//
// Start() holds manager.mu for its entire duration (see its top-level
// defer), so no additional locking is needed here for manager's own fields;
// the repo-scope registry has its own mutex for the concurrent-Manager case.
func (manager *Manager) ensureAMEMQdrant(ctx context.Context, request core.HarnessRequest) error {
	if manager.amemQdrant != nil {
		return nil
	}
	if repoKey, shared := amemRepoScopeKey(manager.amemScope, request); shared {
		amemRepoRegistryMu.Lock()
		defer amemRepoRegistryMu.Unlock()
		if entry, ok := amemRepoRegistry[repoKey]; ok {
			manager.amemQdrant = entry.state
			manager.amemQdrantShared = true
			return nil
		}
		labels := map[string]string{
			"aries.managed": "true", "aries.kind": "openclaw-amem-qdrant-repo", "aries.run": request.RunID,
			"aries.repository": request.Repository, "aries.base_commit": request.BaseCommit,
		}
		state, err := manager.createAMEMQdrant(ctx, repoKey, labels)
		if err != nil {
			return err
		}
		amemRepoRegistry[repoKey] = &amemRepoEntry{state: state, slug: amemRepoSlug(request)}
		manager.amemQdrant = state
		manager.amemQdrantShared = true
		return nil
	}

	// safeTaskID (used for the OpenClaw container name elsewhere in this
	// package) truncates at 48 characters from the start of its input — fine
	// for a bare task ID, but runID alone (a timestamp+profile-name string)
	// routinely exceeds 60 characters, which would truncate away taskID
	// entirely and silently collide every occurrence in the same run right
	// back into the bug this task-scoping change fixes. A short hash of both
	// IDs together is fixed-length and collision-safe regardless of how long
	// either input is.
	scopeHash := sha256.Sum256([]byte(request.RunID + "\x00" + request.TaskID))
	scopeKey := hex.EncodeToString(scopeHash[:])[:16]
	labels := map[string]string{
		"aries.managed": "true", "aries.kind": "openclaw-amem-qdrant", "aries.run": request.RunID, "aries.task": request.TaskID,
	}
	state, err := manager.createAMEMQdrant(ctx, scopeKey, labels)
	if err != nil {
		return err
	}
	manager.amemQdrant = state
	return nil
}

// createAMEMQdrant creates a Docker volume, network, and Qdrant container
// named after scopeKey, labeled with labels. Shared by ensureAMEMQdrant's
// task-scoped and repo-scoped paths.
func (manager *Manager) createAMEMQdrant(ctx context.Context, scopeKey string, labels map[string]string) (*amemQdrantState, error) {
	volumeName := "aries-amem-data-" + scopeKey
	networkName := "aries-amem-net-" + scopeKey
	containerName := "aries-amem-qdrant-" + scopeKey

	if _, err := manager.client.VolumeCreate(ctx, client.VolumeCreateOptions{Name: volumeName, Labels: labels}); err != nil {
		return nil, fmt.Errorf("create amem Qdrant volume: %w", err)
	}
	networkResult, err := manager.client.NetworkCreate(ctx, networkName, client.NetworkCreateOptions{Labels: labels})
	if err != nil {
		return nil, fmt.Errorf("create amem Qdrant network: %w", err)
	}
	created, err := manager.client.ContainerCreate(ctx, client.ContainerCreateOptions{
		Name: containerName,
		Config: &container.Config{
			Image:  manager.amemQdrantImage,
			Labels: labels,
		},
		HostConfig: &container.HostConfig{
			NetworkMode: container.NetworkMode(networkName),
			Mounts: []mount.Mount{{
				Type: mount.TypeVolume, Source: volumeName, Target: "/qdrant/storage",
			}},
		},
		NetworkingConfig: &network.NetworkingConfig{
			EndpointsConfig: map[string]*network.EndpointSettings{
				networkName: {Aliases: []string{amemQdrantNetworkAlias}},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("create amem Qdrant container: %w", err)
	}
	if _, err := manager.client.ContainerStart(ctx, created.ID, client.ContainerStartOptions{}); err != nil {
		return nil, fmt.Errorf("start amem Qdrant container: %w", err)
	}
	return &amemQdrantState{
		networkID: networkResult.ID, networkName: networkName, volumeName: volumeName, containerID: created.ID,
	}, nil
}

// joinAMEMNetwork connects an OpenClaw task container to this task's amem
// Qdrant network (created by ensureAMEMQdrant) as a second network
// membership, alongside its primary per-task network — so the loopback proxy
// started by gatewayLauncherScript can reach amem-qdrant:6333.
func (manager *Manager) joinAMEMNetwork(ctx context.Context, containerID string) error {
	if manager.amemQdrant == nil {
		return errors.New("amem Qdrant network is not initialized")
	}
	if _, err := manager.client.NetworkConnect(ctx, manager.amemQdrant.networkID, client.NetworkConnectOptions{Container: containerID}); err != nil {
		return fmt.Errorf("join amem Qdrant network: %w", err)
	}
	return nil
}

// exportAMEMMemory dumps the final state of this task occurrence's own
// (task-scoped) amem Qdrant collection to its own JSON artifact. Under repo
// scope this is a no-op (manager.amemQdrantShared is true): that store
// outlives this one task occurrence, so its export happens once, in bulk,
// from CleanupSharedAMEMRepoScope — see that function's doc comment.
//
// Every stored note's full payload, including its "links" array (the ids of
// other notes it's linked to; this is the graph edge data amem's own
// automatic contradiction/linking logic writes, confirmed by inspecting the
// plugin's bundled dist/index.js: notes are stored and patched with a
// top-level `links` field) is written to manager.amemExportDir (set in
// Start(); NOT manager.outputDir directly, which is shared by every
// occurrence in the run and would have every occurrence overwrite the same
// file). Called once from Manager.Close, before teardownAMEMQdrant removes
// the container: this task's Qdrant volume is the only place this data ever
// lives, so this is the one chance to capture it.
//
// This reaches Qdrant's REST API directly over the container's Docker
// network IP rather than through the per-task loopback proxy (which only
// exists inside OpenClaw task containers, none of which may still be
// running by the time Close is called) — valid because the ARIES process
// itself talks to the same Docker daemon these containers run under
// (manager.client), so it can reach their bridge-network IPs directly.
func (manager *Manager) exportAMEMMemory(ctx context.Context) error {
	if manager.amemQdrant == nil || manager.amemQdrantShared {
		return nil
	}
	return exportAMEMQdrantState(ctx, manager.client, manager.amemQdrant, manager.amemExportDir)
}

// exportAMEMQdrantState is the free-function core of exportAMEMMemory,
// reused by CleanupSharedAMEMRepoScope (amem_pool.go) to export a
// repo-scoped store's accumulated memory once, at the end of a run, using a
// dedicated Docker client rather than any one Manager's (which may already
// be closed by the time the last task occurrence for that repo finishes).
func exportAMEMQdrantState(ctx context.Context, dockerClient dockerClient, state *amemQdrantState, exportDir string) error {
	inspection, err := dockerClient.ContainerInspect(ctx, state.containerID, client.ContainerInspectOptions{})
	if err != nil {
		return fmt.Errorf("inspect amem Qdrant container for export: %w", err)
	}
	if inspection.Container.NetworkSettings == nil {
		return errors.New("amem Qdrant container has no network settings")
	}
	endpoint, ok := inspection.Container.NetworkSettings.Networks[state.networkName]
	if !ok || !endpoint.IPAddress.IsValid() {
		return errors.New("amem Qdrant container has no address on its network")
	}
	baseURL := "http://" + net.JoinHostPort(endpoint.IPAddress.String(), amemQdrantLoopbackPort)

	httpClient := &http.Client{Timeout: 30 * time.Second}
	var notes []json.RawMessage
	var offset json.RawMessage
	for {
		requestBody, err := json.Marshal(struct {
			Limit       int             `json:"limit"`
			WithPayload bool            `json:"with_payload"`
			WithVector  bool            `json:"with_vector"`
			Offset      json.RawMessage `json:"offset,omitempty"`
		}{Limit: 200, WithPayload: true, WithVector: false, Offset: offset})
		if err != nil {
			return fmt.Errorf("encode amem Qdrant scroll request: %w", err)
		}
		url := baseURL + "/collections/" + amemQdrantCollectionName + "/points/scroll"
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(requestBody))
		if err != nil {
			return fmt.Errorf("build amem Qdrant scroll request: %w", err)
		}
		request.Header.Set("Content-Type", "application/json")
		response, err := httpClient.Do(request)
		if err != nil {
			return fmt.Errorf("scroll amem Qdrant collection: %w", err)
		}
		var decoded struct {
			Status string `json:"status"`
			Result struct {
				Points         []json.RawMessage `json:"points"`
				NextPageOffset json.RawMessage   `json:"next_page_offset"`
			} `json:"result"`
		}
		decodeErr := json.NewDecoder(response.Body).Decode(&decoded)
		closeErr := response.Body.Close()
		if decodeErr != nil {
			return fmt.Errorf("decode amem Qdrant scroll response: %w", decodeErr)
		}
		if closeErr != nil {
			return fmt.Errorf("close amem Qdrant scroll response: %w", closeErr)
		}
		if response.StatusCode == http.StatusNotFound {
			// No notes were ever stored — the collection is never created
			// until the first memory_add call (confirmed empirically).
			return writeAMEMMemoryExport(exportDir, nil)
		}
		if decoded.Status != "ok" {
			return fmt.Errorf("amem Qdrant scroll returned status %q", decoded.Status)
		}
		notes = append(notes, decoded.Result.Points...)
		if len(decoded.Result.NextPageOffset) == 0 || string(decoded.Result.NextPageOffset) == "null" {
			break
		}
		offset = decoded.Result.NextPageOffset
	}
	return writeAMEMMemoryExport(exportDir, notes)
}

func writeAMEMMemoryExport(exportDir string, notes []json.RawMessage) error {
	if notes == nil {
		notes = []json.RawMessage{}
	}
	content, err := json.MarshalIndent(notes, "", "  ")
	if err != nil {
		return fmt.Errorf("encode amem memory export: %w", err)
	}
	content = append(content, '\n')
	// exportDir (manager.outputDir + task ID, or the repo-scoped cleanup
	// export directory) already exists in the common case, but MkdirAll
	// defensively in case ensureAMEMQdrant succeeded before that happened.
	if err := os.MkdirAll(exportDir, 0o700); err != nil {
		return fmt.Errorf("create amem memory export directory: %w", err)
	}
	path := filepath.Join(exportDir, "amem-memory.json")
	if err := os.WriteFile(path, content, 0o600); err != nil {
		return fmt.Errorf("write amem memory export: %w", err)
	}
	return nil
}

// teardownAMEMQdrant removes this task occurrence's own (task-scoped) amem
// Qdrant container, network, and volume. Under repo scope this is a no-op
// (manager.amemQdrantShared is true): that store outlives this one task
// occurrence, so it's torn down once, in bulk, from
// CleanupSharedAMEMRepoScope instead — see that function's doc comment.
// Called once from Manager.Close, at the end of this task occurrence.
func (manager *Manager) teardownAMEMQdrant(ctx context.Context) error {
	if manager.amemQdrant == nil {
		return nil
	}
	if manager.amemQdrantShared {
		manager.amemQdrant = nil
		return nil
	}
	state := manager.amemQdrant
	manager.amemQdrant = nil
	return teardownAMEMQdrantState(ctx, manager.client, state)
}

// teardownAMEMQdrantState is the free-function core of teardownAMEMQdrant,
// reused by CleanupSharedAMEMRepoScope to tear down a repo-scoped store once
// every task occurrence sharing it has finished.
func teardownAMEMQdrantState(ctx context.Context, dockerClient dockerClient, state *amemQdrantState) error {
	var errs []error
	timeout := gracefulStopSeconds
	if _, err := dockerClient.ContainerStop(ctx, state.containerID, client.ContainerStopOptions{Timeout: &timeout}); err != nil && !errdefs.IsNotFound(err) {
		errs = append(errs, fmt.Errorf("stop amem Qdrant container: %w", err))
	}
	if _, err := dockerClient.ContainerRemove(ctx, state.containerID, client.ContainerRemoveOptions{Force: true}); err != nil && !errdefs.IsNotFound(err) {
		errs = append(errs, fmt.Errorf("remove amem Qdrant container: %w", err))
	}
	if _, err := dockerClient.NetworkRemove(ctx, state.networkID, client.NetworkRemoveOptions{}); err != nil && !errdefs.IsNotFound(err) {
		errs = append(errs, fmt.Errorf("remove amem Qdrant network: %w", err))
	}
	if _, err := dockerClient.VolumeRemove(ctx, state.volumeName, client.VolumeRemoveOptions{Force: true}); err != nil && !errdefs.IsNotFound(err) {
		errs = append(errs, fmt.Errorf("remove amem Qdrant volume: %w", err))
	}
	return errors.Join(errs...)
}
