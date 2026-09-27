package hermes

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/containerd/errdefs"
	"github.com/hyscale-lab/aries/pkg/core"
	"github.com/moby/moby/api/types/container"
	"github.com/moby/moby/client"
	"github.com/sirupsen/logrus"
)

// The MemRL provider (docker/hermes-memrl) keeps its SQLite store under
// HERMES_HOME. ARIES owns one run-scoped copy on the host and hands it to each
// task: staged into the runtime archive at Start, copied back out after the
// one-shot exits. Hermes containers take no mounts, so this copy is the only
// path between tasks, and config validation keeps it sequential.
//
// MemRL learns from the environment's success signal, which only exists after
// evaluation, when the task's container is gone. So the provider runs with
// external rewards: at exit it parks the session under the task ID, ARIES
// records the task's verdict in rewards.json once evaluation finishes, and the
// next task's provider applies it before its first recall.
const (
	memrlStoreName          = "memrl.db"
	memrlRewardsName        = "rewards.json"
	memrlStoreContainerDir  = stateContainerPath + "/memrl"
	memrlStoreContainerPath = memrlStoreContainerDir + "/" + memrlStoreName
	memrlConfigBlock        = "\nmemory:\n  provider: \"memrl\"\n"
	maxMemRLRewards         = 1 << 20

	// Batched training: each task parks its session in memrlPendingName, the
	// run queues them under memrl/pending, and FinalizeMemRLBatch applies
	// them and files them under memrl/batches/NNN.
	memrlPendingName     = "pending.json"
	memrlPendingDir      = "pending"
	memrlBatchesDir      = "batches"
	memrlFinalizePath    = stagedRoot + "/run-memrl-finalize"
	memrlFinalizeTimeout = 30 * time.Minute
)

// memrlEnvironment tells the provider to wait for ARIES's verdict and which
// task its session belongs to. A frozen provider recalls greedily (no
// exploration) and never learns; a batched one parks its session in a file
// for FinalizeMemRLBatch. No value is secret.
func memrlEnvironment(taskID string, frozen, batch bool) []string {
	environment := []string{"MEMRL_REWARD_SOURCE=external", "MEMRL_TASK_ID=" + taskID}
	if frozen {
		environment = append(environment, "MEMRL_FROZEN=1", "MEMRL_EPSILON=0")
	}
	if batch {
		environment = append(environment, "MEMRL_BATCH=1")
	}
	return environment
}

// readMemRLState returns the store and rewards a task starts from:
//   - frozen: the frozen store, required, and no rewards, since nothing learns;
//   - batched: the run's store as the last batch update left it (none before
//     the first), and no rewards, since the batch update applies them;
//   - sequential: the run's store and rewards as the previous task left them.
func (manager *Manager) readMemRLState() ([]byte, []byte, error) {
	if manager.memrlFrozenStore != "" {
		store, err := readMemRLStore(filepath.Join(manager.memrlFrozenStore, memrlStoreName))
		if err == nil && store == nil {
			err = fmt.Errorf("MemRL frozen store %s has no %s", manager.memrlFrozenStore, memrlStoreName)
		}
		return store, nil, err
	}
	store, err := readMemRLStore(manager.memrlStorePath)
	if err != nil || manager.memrlBatch {
		return store, nil, err
	}
	rewards, err := readMemRLRewards(filepath.Join(filepath.Dir(manager.memrlStorePath), memrlRewardsName))
	if err != nil {
		clear(store)
		return nil, nil, err
	}
	return store, rewards, nil
}

// readMemRLRewards returns the run's recorded verdicts, or nil before the
// first one.
func readMemRLRewards(path string) ([]byte, error) {
	content, err := readStablePrivateFile(path, 0o600)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read MemRL rewards: %w", err)
	}
	var rewards map[string]*float64
	if len(content) > maxMemRLRewards || json.Unmarshal(content, &rewards) != nil {
		return nil, errors.New("MemRL rewards are not a bounded JSON object of task rewards")
	}
	return content, nil
}

// memrlReward maps an evaluation to MemRL's success signal: +1 when the
// benchmark awarded full reward, -1 otherwise. It is nil when the benchmark
// reached no verdict (not run, blocked, canceled, or failed with an error of
// its own), so the provider drops that session instead of learning from it.
func memrlReward(evaluation core.Evaluation) *float64 {
	if evaluation.Error != "" || evaluation.Status != core.StatusSucceeded && evaluation.Status != core.StatusFailed {
		return nil
	}
	reward := -1.0
	if evaluation.Reward >= 1 {
		reward = 1
	}
	return &reward
}

// memrlRewardsMu serializes rewards.json updates from a batch's concurrent
// tasks, which all finish in this process.
var memrlRewardsMu sync.Mutex

// RecordMemRLOutcome adds one task's verdict to the run's rewards.json under
// outputDir/memrl. A sequential run records each task before the next one
// starts and reads it; a batched run records the whole batch before its
// update.
func RecordMemRLOutcome(outputDir string, task core.TaskResult) error {
	if err := validateTaskID(task.TaskID); err != nil {
		return err
	}
	memrlRewardsMu.Lock()
	defer memrlRewardsMu.Unlock()
	path := filepath.Join(outputDir, "memrl", memrlRewardsName)
	rewards := map[string]*float64{}
	existing, err := readMemRLRewards(path)
	if err != nil {
		return err
	}
	if existing != nil {
		if err := json.Unmarshal(existing, &rewards); err != nil {
			return err
		}
	}
	rewards[task.TaskID] = memrlReward(task.Evaluation)
	encoded, err := json.MarshalIndent(rewards, "", "  ")
	if err != nil {
		return err
	}
	if err := replacePrivateFile(path, append(encoded, '\n')); err != nil {
		return fmt.Errorf("record MemRL reward: %w", err)
	}
	return nil
}

var sqliteHeader = []byte("SQLite format 3\x00")

// readMemRLStore returns the run-scoped store, or nil before the first task
// has produced one.
func readMemRLStore(path string) ([]byte, error) {
	content, err := readStablePrivateFile(path, 0o600)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read MemRL store: %w", err)
	}
	if !bytes.HasPrefix(content, sqliteHeader) {
		clear(content)
		return nil, errors.New("MemRL store is not an SQLite database")
	}
	return content, nil
}

// exportMemRLStore copies the store out of the still-running container and
// retains it as a task artifact. A sequential run also makes it the run-scoped
// copy the next task starts from; frozen and batched tasks never hand their
// store on. It is called only after the one-shot process has exited, so the
// provider's exit commit has finished writing. A missing store means the
// provider never committed (for example, the task prompt was trivial) and
// changes nothing.
func (manager *Manager) exportMemRLStore(ctx context.Context, active *session) (string, error) {
	result, err := manager.client.CopyFromContainer(ctx, active.containerID, client.CopyFromContainerOptions{SourcePath: memrlStoreContainerPath})
	if errdefs.IsNotFound(err) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("collect MemRL store: %w", err)
	}
	defer result.Content.Close()
	content, err := singleRegularFile(result.Content, memrlStoreName)
	if err != nil {
		return "", fmt.Errorf("extract MemRL store: %w", err)
	}
	if !bytes.HasPrefix(content, sqliteHeader) {
		return "", errors.New("exported MemRL store is not an SQLite database")
	}
	artifact := filepath.Join(active.artifactDir, "memrl", memrlStoreName)
	if err := writeArtifact(artifact, content); err != nil {
		return "", fmt.Errorf("retain MemRL store: %w", err)
	}
	if manager.memrlFrozenStore == "" && !manager.memrlBatch {
		if err := replacePrivateFile(manager.memrlStorePath, content); err != nil {
			return "", fmt.Errorf("update run MemRL store: %w", err)
		}
	}
	return artifact, nil
}

// exportMemRLPending collects a batched task's parked session and queues it,
// under the task's execution ID, for the batch update. A task that parked
// nothing (a trivial prompt, or a one-shot that never reached its exit
// commit) queues nothing.
func (manager *Manager) exportMemRLPending(ctx context.Context, active *session) (string, error) {
	result, err := manager.client.CopyFromContainer(ctx, active.containerID, client.CopyFromContainerOptions{SourcePath: memrlStoreContainerDir + "/" + memrlPendingName})
	if errdefs.IsNotFound(err) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("collect parked MemRL session: %w", err)
	}
	defer result.Content.Close()
	content, err := singleRegularFile(result.Content, memrlPendingName)
	if err != nil {
		return "", fmt.Errorf("extract parked MemRL session: %w", err)
	}
	var parked map[string]any
	if json.Unmarshal(content, &parked) != nil || parked["task_id"] != active.taskID {
		return "", errors.New("parked MemRL session is not a JSON object for this task")
	}
	artifact := filepath.Join(active.artifactDir, "memrl", memrlPendingName)
	if err := writeArtifact(artifact, content); err != nil {
		return "", fmt.Errorf("retain parked MemRL session: %w", err)
	}
	queued := filepath.Join(filepath.Dir(manager.memrlStorePath), memrlPendingDir, active.taskID+".json")
	if err := replacePrivateFile(queued, content); err != nil {
		return "", fmt.Errorf("queue parked MemRL session: %w", err)
	}
	return artifact, nil
}

// FinalizeMemRLBatch is the mini-batch update. It stages the run's store, the
// batch's parked sessions, and their rewards into one short-lived container
// of the MemRL image. Inside, the provider's finalize step applies every
// reward in execution order: utility updates for the memories each task
// recalled, and a new memory per task, whose script or reflection costs a
// call to the task model. The updated store replaces the run's store, and
// the applied sessions move to memrl/batches/NNN with the step's log. With
// nothing parked it does nothing. Like a task's Hermes container, it gets the
// model credential only as a staged file.
func (manager *Manager) FinalizeMemRLBatch(ctx context.Context, model core.ModelConfig) (returnErr error) {
	runDir := filepath.Dir(manager.memrlStorePath)
	pendingDir := filepath.Join(runDir, memrlPendingDir)
	entries, err := os.ReadDir(pendingDir)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("list parked MemRL sessions: %w", err)
	}
	files := map[string]stagedFile{}
	var names []string
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, ".json") || validateTaskID(strings.TrimSuffix(name, ".json")) != nil {
			return fmt.Errorf("unexpected entry %q among parked MemRL sessions", name)
		}
		content, err := readStablePrivateFile(filepath.Join(pendingDir, name), 0o600)
		if err != nil {
			return fmt.Errorf("read parked MemRL session %s: %w", name, err)
		}
		files[strings.TrimPrefix(memrlStoreContainerDir+"/"+memrlPendingDir+"/"+name, "/")] = stagedFile{content: content, mode: 0o600}
		names = append(names, name)
	}
	if len(names) == 0 {
		return nil
	}
	store, err := readMemRLStore(manager.memrlStorePath)
	if err != nil {
		return err
	}
	if store != nil {
		files[strings.TrimPrefix(memrlStoreContainerPath, "/")] = stagedFile{content: store, mode: 0o600}
	}
	rewards, err := readMemRLRewards(filepath.Join(runDir, memrlRewardsName))
	if err != nil {
		return err
	}
	if rewards == nil {
		return errors.New("MemRL batch has parked sessions but no recorded rewards")
	}
	files[strings.TrimPrefix(memrlStoreContainerDir+"/"+memrlRewardsName, "/")] = stagedFile{content: rewards, mode: 0o600}
	configuration, err := renderConfig(model, manager.maxTurns, false, false, false, 0)
	if err != nil {
		return err
	}
	apiKeySource, ok := manager.apiKeyLookup(model.APIKeyEnv)
	if !ok {
		clear(apiKeySource)
		return fmt.Errorf("Hermes API-key environment %q is not set", model.APIKeyEnv)
	}
	apiKey := bytes.Clone(apiKeySource)
	clear(apiKeySource)
	if err := validateAPIKey(apiKey); err != nil || bytes.Contains(configuration, apiKey) {
		clear(apiKey)
		return errors.Join(err, errors.New("MemRL batch update has no usable model credential"))
	}
	files[strings.TrimPrefix(configContainerPath, "/")] = stagedFile{content: configuration, mode: 0o600}
	files[strings.TrimPrefix(modelKeyPath, "/")] = stagedFile{content: apiKey, mode: 0o600}
	files[strings.TrimPrefix(memrlFinalizePath, "/")] = stagedFile{content: memrlFinalizeScript(model.APIKeyEnv), mode: 0o555}
	archive, err := stageArchive(files, strings.TrimPrefix(memrlStoreContainerDir, "/"), strings.TrimPrefix(memrlStoreContainerDir+"/"+memrlPendingDir, "/"))
	if err != nil {
		clear(apiKey)
		return err
	}
	defer clear(archive)
	id, err := manager.newID()
	if err != nil {
		clear(apiKey)
		return fmt.Errorf("generate MemRL batch update ID: %w", err)
	}
	active := &session{
		runID: filepath.Base(manager.outputDir), taskID: "memrl-batch", attemptID: id,
		containerName: "aries-hermes-memrl-" + id, model: model, apiKey: apiKey,
	}
	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), manager.cleanupTimeout)
		defer cancel()
		if err := manager.stopSession(cleanupCtx, active); err != nil {
			returnErr = errors.Join(returnErr, fmt.Errorf("remove MemRL batch update container: %w", err))
		}
	}()
	created, err := manager.client.ContainerCreate(ctx, client.ContainerCreateOptions{
		Name: active.containerName,
		Config: &container.Config{
			Image:      manager.image,
			Env:        []string{"HERMES_HOME=" + stateContainerPath, "HF_HUB_OFFLINE=1"},
			Entrypoint: append([]string(nil), idleEntrypoint...),
			Cmd:        append([]string(nil), idleCommand...),
			Labels: map[string]string{
				"aries.managed": "true", "aries.kind": "hermes-memrl-batch", "aries.component": "harness",
				"aries.run": active.runID, "aries.attempt": id,
			},
		},
		HostConfig: &container.HostConfig{NetworkMode: "bridge"},
	})
	if err != nil {
		return fmt.Errorf("create MemRL batch update container: %w", err)
	}
	active.containerID = created.ID
	if strings.TrimSpace(active.containerID) == "" {
		return errors.New("Docker returned an empty MemRL batch update container ID")
	}
	if _, err := manager.client.CopyToContainer(ctx, active.containerID, client.CopyToContainerOptions{
		DestinationPath: "/", Content: bytes.NewReader(archive), CopyUIDGID: true,
	}); err != nil {
		return fmt.Errorf("stage MemRL batch update: %w", err)
	}
	if _, err := manager.client.ContainerStart(ctx, active.containerID, client.ContainerStartOptions{}); err != nil {
		return fmt.Errorf("start MemRL batch update container: %w", err)
	}
	execCtx, cancel := context.WithTimeout(ctx, memrlFinalizeTimeout)
	result, err := manager.execAttached(execCtx, active.containerID, []string{memrlFinalizePath}, "/opt/hermes")
	cancel()
	batches, readErr := os.ReadDir(filepath.Join(runDir, memrlBatchesDir))
	if readErr != nil && !errors.Is(readErr, os.ErrNotExist) {
		return errors.Join(err, readErr)
	}
	batchDir := filepath.Join(runDir, memrlBatchesDir, fmt.Sprintf("%03d", len(batches)+1))
	logContent := redactSession(append(append([]byte(nil), result.stdout...), result.stderr...), active)
	if logErr := writeArtifact(filepath.Join(batchDir, "finalize.log"), logContent); logErr != nil {
		return errors.Join(err, fmt.Errorf("retain MemRL batch update log: %w", logErr))
	}
	if err == nil && result.exitCode != 0 {
		err = fmt.Errorf("MemRL batch update exited with status %d (see %s)", result.exitCode, filepath.Join(batchDir, "finalize.log"))
	}
	if err != nil {
		return err
	}
	copied, err := manager.client.CopyFromContainer(ctx, active.containerID, client.CopyFromContainerOptions{SourcePath: memrlStoreContainerPath})
	if err != nil {
		return fmt.Errorf("collect updated MemRL store: %w", err)
	}
	defer copied.Content.Close()
	updated, err := singleRegularFile(copied.Content, memrlStoreName)
	if err != nil {
		return fmt.Errorf("extract updated MemRL store: %w", err)
	}
	if !bytes.HasPrefix(updated, sqliteHeader) {
		return errors.New("updated MemRL store is not an SQLite database")
	}
	if err := replacePrivateFile(manager.memrlStorePath, updated); err != nil {
		return fmt.Errorf("update run MemRL store: %w", err)
	}
	for _, name := range names {
		if err := os.Rename(filepath.Join(pendingDir, name), filepath.Join(batchDir, name)); err != nil {
			return fmt.Errorf("retire applied MemRL session %s: %w", name, err)
		}
	}
	manager.logger.WithContext(ctx).WithFields(logrus.Fields{"sessions": len(names), "batch": filepath.Base(batchDir)}).Info("MemRL batch update applied")
	return nil
}

// memrlFinalizeScript exports the staged credential under its required name,
// like agentWrapperScript, and runs the provider's batch update.
func memrlFinalizeScript(apiKeyEnv string) []byte {
	return []byte(`#!/bin/sh
set -eu
` + apiKeyEnv + `="$(cat ` + modelKeyPath + `)"
export ` + apiKeyEnv + `
exec /opt/hermes/.venv/bin/python -m plugins.memory.memrl.finalize
`)
}

// singleRegularFile reads a Docker copy archive that must hold exactly one
// bounded regular file with the given name.
func singleRegularFile(stream io.Reader, name string) ([]byte, error) {
	reader := tar.NewReader(io.LimitReader(stream, maxDockerOutput+64<<10))
	var content []byte
	for {
		header, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		if content != nil || header.Typeflag != tar.TypeReg || header.Name != name || header.Size < 1 || header.Size > maxDockerOutput {
			return nil, fmt.Errorf("archive must contain only the regular file %q within its bound", name)
		}
		content, err = io.ReadAll(io.LimitReader(reader, header.Size+1))
		if err != nil || int64(len(content)) != header.Size {
			return nil, errors.New("archive entry is truncated")
		}
	}
	if content == nil {
		return nil, fmt.Errorf("archive does not contain %q", name)
	}
	return content, nil
}

// replacePrivateFile atomically replaces path with a 0600 file so a crash
// never leaves a torn store for the next task.
func replacePrivateFile(path string, content []byte) error {
	directory := filepath.Dir(path)
	if err := ensurePrivateDirectory(directory); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(path)+".*")
	if err != nil {
		return err
	}
	name := temporary.Name()
	_, writeErr := temporary.Write(content)
	syncErr := temporary.Sync()
	closeErr := temporary.Close()
	if err := errors.Join(writeErr, syncErr, closeErr); err != nil {
		_ = os.Remove(name)
		return err
	}
	if err := os.Rename(name, path); err != nil {
		_ = os.Remove(name)
		return err
	}
	return nil
}
