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
	"time"

	"github.com/containerd/errdefs"
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
	// amemQdrantNetworkAlias is the fixed Docker network alias the run-scoped
	// Qdrant container (see ensureAMEMQdrant) registers on the shared amem
	// network, and what the loopback proxy forwards to.
	amemQdrantNetworkAlias = "amem-qdrant"
	// amemQdrantCollectionName is the plugin's own default Qdrant collection
	// name (confirmed against a live gateway boot log: "default
	// collection=amem_notes (default)"). amemPluginConfig deliberately never
	// sets config.collection, so this stays accurate as long as that remains
	// true — update both together if that ever changes.
	amemQdrantCollectionName = "amem_notes"
	// amemMemoryExportTimeout bounds exportAMEMMemory's Qdrant scroll calls —
	// generous since this runs once at the very end of a run, not per task.
	amemMemoryExportTimeout = 2 * time.Minute
)

// amemQdrantState holds the task-scoped Qdrant resources ensureAMEMQdrant
// creates, so Manager.Close (exportAMEMMemory, teardownAMEMQdrant) can read
// and then remove them once this task occurrence ends. nil until amem is
// enabled and this Manager's one Start() call runs.
type amemQdrantState struct {
	networkID   string
	networkName string
	volumeName  string
	containerID string
}

// ensureAMEMQdrant idempotently creates a Docker volume, network, and Qdrant
// container scoped to this one task occurrence (runID+taskID), giving amem a
// private memory store per task attempt.
//
// This is deliberately task-scoped, not run-scoped: internal/app/run.go's
// buildTaskExperiment constructs a brand-new *openclaw.Manager for every task
// occurrence (confirmed by reading its dispatch loop) — there is no single
// Manager instance shared across a run's occurrences the way an earlier
// version of this file assumed. A run-scoped resource name keyed only on
// runID therefore collided the moment a second occurrence's fresh Manager
// (with its own nil amemQdrant) tried to create the same network name — this
// failed deterministically from the second occurrence onward regardless of
// execution.concurrency, and even a fix that tolerated "already exists"
// would still be wrong: teardownAMEMQdrant/exportAMEMMemory run once per
// Manager.Close (also once per occurrence, not once per run — see
// internal/app/run.go's closeOccurrenceClients), so the first occurrence to
// finish would tear down and export Qdrant out from under every other
// occurrence still using it. Scoping per-task instead sidesteps all of that:
// each Manager's create/use/teardown cycle is now fully self-contained, no
// cross-Manager coordination needed. The tradeoff is that amem's memory no
// longer links/persists across different tasks in the same benchmark run —
// acceptable since Deep Research Bench tasks are independent research
// questions with no shared context to begin with.
//
// Start() holds manager.mu for its entire duration (see its top-level
// defer), so no additional locking is needed here.
func (manager *Manager) ensureAMEMQdrant(ctx context.Context, runID, taskID string) error {
	if manager.amemQdrant != nil {
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
	scopeHash := sha256.Sum256([]byte(runID + "\x00" + taskID))
	scopeKey := hex.EncodeToString(scopeHash[:])[:16]
	labels := map[string]string{
		"aries.managed": "true", "aries.kind": "openclaw-amem-qdrant", "aries.run": runID, "aries.task": taskID,
	}
	volumeName := "aries-amem-data-" + scopeKey
	networkName := "aries-amem-net-" + scopeKey
	containerName := "aries-amem-qdrant-" + scopeKey

	if _, err := manager.client.VolumeCreate(ctx, client.VolumeCreateOptions{Name: volumeName, Labels: labels}); err != nil {
		return fmt.Errorf("create amem Qdrant volume: %w", err)
	}
	networkResult, err := manager.client.NetworkCreate(ctx, networkName, client.NetworkCreateOptions{Labels: labels})
	if err != nil {
		return fmt.Errorf("create amem Qdrant network: %w", err)
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
		return fmt.Errorf("create amem Qdrant container: %w", err)
	}
	if _, err := manager.client.ContainerStart(ctx, created.ID, client.ContainerStartOptions{}); err != nil {
		return fmt.Errorf("start amem Qdrant container: %w", err)
	}
	manager.amemQdrant = &amemQdrantState{
		networkID: networkResult.ID, networkName: networkName, volumeName: volumeName, containerID: created.ID,
	}
	return nil
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

// exportAMEMMemory dumps the final state of amem's Qdrant collection — every
// stored note's full payload, including its "links" array (the ids of other
// notes it's linked to; this is the graph edge data amem's own automatic
// contradiction/linking logic writes, confirmed by inspecting the plugin's
// bundled dist/index.js: notes are stored and patched with a top-level
// `links` field) — to this task occurrence's own JSON artifact
// (manager.amemExportDir, set in Start(); NOT manager.outputDir directly,
// which is shared by every occurrence in the run and would have every
// occurrence overwrite the same file). Called once from Manager.Close,
// before teardownAMEMQdrant removes the container: this task's Qdrant volume
// is the only place this data ever lives (see ensureAMEMQdrant's doc comment
// on why it's deliberately not persisted beyond one task attempt), so this
// is the one chance to capture it.
//
// This reaches Qdrant's REST API directly over the container's Docker
// network IP rather than through the per-task loopback proxy (which only
// exists inside OpenClaw task containers, none of which may still be
// running by the time Close is called) — valid because the ARIES process
// itself talks to the same Docker daemon these containers run under
// (manager.client), so it can reach their bridge-network IPs directly.
func (manager *Manager) exportAMEMMemory(ctx context.Context) error {
	if manager.amemQdrant == nil {
		return nil
	}
	inspection, err := manager.client.ContainerInspect(ctx, manager.amemQdrant.containerID, client.ContainerInspectOptions{})
	if err != nil {
		return fmt.Errorf("inspect amem Qdrant container for export: %w", err)
	}
	if inspection.Container.NetworkSettings == nil {
		return errors.New("amem Qdrant container has no network settings")
	}
	endpoint, ok := inspection.Container.NetworkSettings.Networks[manager.amemQdrant.networkName]
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
			// No notes were ever stored this run — the collection is never
			// created until the first memory_add call (confirmed empirically).
			return writeAMEMMemoryExport(manager.amemExportDir, nil)
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
	return writeAMEMMemoryExport(manager.amemExportDir, notes)
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
	// exportDir (manager.outputDir + task ID) already exists by the time a
	// task attempt reaches Close in the common case (Start's artifactDir
	// creation is an ancestor of it), but MkdirAll defensively in case
	// ensureAMEMQdrant succeeded before that happened.
	if err := os.MkdirAll(exportDir, 0o700); err != nil {
		return fmt.Errorf("create amem memory export directory: %w", err)
	}
	path := filepath.Join(exportDir, "amem-memory.json")
	if err := os.WriteFile(path, content, 0o600); err != nil {
		return fmt.Errorf("write amem memory export: %w", err)
	}
	return nil
}

// teardownAMEMQdrant removes the task-scoped Qdrant container, network, and
// volume created by ensureAMEMQdrant. Called once from Manager.Close, at the
// end of this task occurrence — deliberately not persisted beyond it, so a
// later task (in this run or another) never inherits an earlier task's
// stored memories (see amemPluginConfig's doc comment on why memory linking
// is scoped the way it is).
func (manager *Manager) teardownAMEMQdrant(ctx context.Context) error {
	if manager.amemQdrant == nil {
		return nil
	}
	state := manager.amemQdrant
	manager.amemQdrant = nil
	var errs []error
	timeout := gracefulStopSeconds
	if _, err := manager.client.ContainerStop(ctx, state.containerID, client.ContainerStopOptions{Timeout: &timeout}); err != nil && !errdefs.IsNotFound(err) {
		errs = append(errs, fmt.Errorf("stop amem Qdrant container: %w", err))
	}
	if _, err := manager.client.ContainerRemove(ctx, state.containerID, client.ContainerRemoveOptions{Force: true}); err != nil && !errdefs.IsNotFound(err) {
		errs = append(errs, fmt.Errorf("remove amem Qdrant container: %w", err))
	}
	if _, err := manager.client.NetworkRemove(ctx, state.networkID, client.NetworkRemoveOptions{}); err != nil && !errdefs.IsNotFound(err) {
		errs = append(errs, fmt.Errorf("remove amem Qdrant network: %w", err))
	}
	if _, err := manager.client.VolumeRemove(ctx, state.volumeName, client.VolumeRemoveOptions{Force: true}); err != nil && !errdefs.IsNotFound(err) {
		errs = append(errs, fmt.Errorf("remove amem Qdrant volume: %w", err))
	}
	return errors.Join(errs...)
}
