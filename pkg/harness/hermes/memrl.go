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

	"github.com/containerd/errdefs"
	"github.com/hyscale-lab/aries/pkg/core"
	"github.com/moby/moby/client"
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
)

// memrlEnvironment tells the provider to wait for ARIES's verdict and which
// task its session belongs to. Similarity retrieval zeroes the utility weight
// and exploration, so recall ranks by similarity alone. No value is secret.
func memrlEnvironment(taskID, retrieval string) []string {
	environment := []string{"MEMRL_REWARD_SOURCE=external", "MEMRL_TASK_ID=" + taskID}
	if retrieval == "similarity" {
		environment = append(environment, "MEMRL_LAM=0", "MEMRL_EPSILON=0")
	}
	return environment
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

// RecordMemRLOutcome adds one task's verdict to the run's rewards.json under
// outputDir/memrl. The run is sequential when MemRL is enabled, so each task
// is recorded before the next one starts and reads it.
func RecordMemRLOutcome(outputDir string, task core.TaskResult) error {
	if err := validateTaskID(task.TaskID); err != nil {
		return err
	}
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

// exportMemRLStore copies the store out of the still-running container,
// retains it as a task artifact, and replaces the run-scoped copy. It is
// called only after the one-shot process has exited, so the provider's exit
// commit has finished writing. A missing store means the provider never
// committed (for example, the task prompt was trivial) and changes nothing.
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
	if err := replacePrivateFile(manager.memrlStorePath, content); err != nil {
		return "", fmt.Errorf("update run MemRL store: %w", err)
	}
	return artifact, nil
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
