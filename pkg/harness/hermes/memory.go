package hermes

import (
	"archive/tar"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"syscall"

	"github.com/containerd/errdefs"
	"github.com/moby/moby/client"
)

// A Hermes memory manager is a memory provider plugin baked into the Hermes
// image (docs/design/harness.md has the contract). ARIES names it in
// config.yaml, passes the profile's environment through, and carries one
// opaque state directory from task to task: staged at ARIES_MEMORY_DIR on
// Start and copied back out after the one-shot exits. ARIES never interprets
// the directory's contents. Hermes containers take no mounts, so this copy is
// the only path between tasks, and config validation keeps it sequential
// unless the state is frozen.
const (
	memoryContainerDir    = stateContainerPath + "/memory"
	memoryRunDirName      = "memory"
	maxMemoryStateBytes   = 256 << 20
	maxMemoryStateEntries = 4096
)

var (
	memoryProviderPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)
	memoryEnvKeyPattern   = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
)

// MemoryOptions selects a Hermes memory manager. A zero Provider disables it.
type MemoryOptions struct {
	// Provider is the plugin name under /opt/hermes/plugins/memory in the
	// image, rendered as memory.provider in config.yaml.
	Provider string
	// Env is passed to the Hermes container verbatim; it holds provider
	// tuning, never secrets.
	Env map[string]string
	// FrozenState, when set, is a state directory every task starts from
	// and none writes back.
	FrozenState string
}

func (options MemoryOptions) validate(secretEnvs ...string) error {
	if options.Provider == "" {
		if len(options.Env) != 0 || options.FrozenState != "" {
			return errors.New("Hermes memory env and frozen state require a provider")
		}
		return nil
	}
	if !memoryProviderPattern.MatchString(options.Provider) || len(options.Provider) > 64 {
		return fmt.Errorf("Hermes memory provider %q is not a plugin name", options.Provider)
	}
	for key := range options.Env {
		if err := validateMemoryEnvKey(key, secretEnvs...); err != nil {
			return err
		}
	}
	return nil
}

// validateMemoryEnvKey rejects a memory env name that is malformed or would
// override a variable ARIES itself sets in the Hermes container, including
// the named credential variables.
func validateMemoryEnvKey(key string, secretEnvs ...string) error {
	switch {
	case !memoryEnvKeyPattern.MatchString(key):
		return fmt.Errorf("Hermes memory env name %q must match %s", key, memoryEnvKeyPattern)
	case strings.HasPrefix(key, "ARIES_"), strings.HasPrefix(key, "HERMES_"), strings.HasPrefix(key, "TERMINAL_"),
		key == "SEARXNG_URL", key == tavilyAPIKeyEnv, key == "PATH", key == "HOME",
		key == "HF_HUB_OFFLINE", key == "TRANSFORMERS_OFFLINE", slices.Contains(secretEnvs, key):
		return fmt.Errorf("Hermes memory env name %q is reserved", key)
	}
	return nil
}

// memoryConfigBlock selects the provider and turns off Hermes's built-in
// MEMORY.md/USER.md store, whose tool the "memory" toolset also exposes, so the
// provider is the only memory the agent has.
func memoryConfigBlock(provider string) []byte {
	return []byte("\nmemory:\n  provider: \"" + provider + "\"\n  memory_enabled: false\n  user_profile_enabled: false\n")
}

// memoryEnvironment tells the provider where its state lives and which task
// execution it serves, followed by the profile's env in a stable order.
func memoryEnvironment(taskID string, env map[string]string) []string {
	environment := []string{"ARIES_MEMORY_DIR=" + memoryContainerDir, "ARIES_MEMORY_TASK_ID=" + taskID}
	keys := make([]string, 0, len(env))
	for key := range env {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		environment = append(environment, key+"="+env[key])
	}
	return environment
}

// memoryState is a bounded snapshot of a state directory: slash-separated
// paths relative to its root, directories listed parents first.
type memoryState struct {
	directories []string
	files       map[string][]byte
	size        int64
}

func (state *memoryState) add(name string, directory bool, content []byte) error {
	if name == "" || name == "." || path.IsAbs(name) || path.Clean(name) != name || name == ".." || strings.HasPrefix(name, "../") {
		return fmt.Errorf("memory state path %q is not a clean relative path", name)
	}
	if len(state.directories)+len(state.files) >= maxMemoryStateEntries {
		return fmt.Errorf("memory state has more than %d entries", maxMemoryStateEntries)
	}
	if directory {
		state.directories = append(state.directories, name)
		return nil
	}
	state.size += int64(len(content))
	if state.size > maxMemoryStateBytes {
		return fmt.Errorf("memory state exceeds %d bytes", maxMemoryStateBytes)
	}
	state.files[name] = content
	return nil
}

// memoryStateSource is the directory a task starts from: the frozen state,
// or the run's state as the previous task left it.
func (manager *Manager) memoryStateSource() string {
	if manager.memory.FrozenState != "" {
		return manager.memory.FrozenState
	}
	return filepath.Join(manager.outputDir, memoryRunDirName)
}

// readMemoryState snapshots a host state directory. A missing run state is
// the empty state of a run's first task; a missing frozen state is an error.
func (manager *Manager) readMemoryState() (*memoryState, error) {
	root := manager.memoryStateSource()
	state := &memoryState{files: map[string][]byte{}}
	info, err := os.Lstat(root)
	if errors.Is(err, os.ErrNotExist) && manager.memory.FrozenState == "" {
		return state, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read Hermes memory state: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("Hermes memory state %s is not a directory", root)
	}
	err = filepath.WalkDir(root, func(current string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || current == root {
			return walkErr
		}
		relative, err := filepath.Rel(root, current)
		if err != nil {
			return err
		}
		switch {
		case entry.IsDir():
			return state.add(filepath.ToSlash(relative), true, nil)
		case entry.Type().IsRegular():
			content, err := readMemoryFile(current)
			if err != nil {
				return err
			}
			return state.add(filepath.ToSlash(relative), false, content)
		default:
			return fmt.Errorf("memory state entry %s is not a regular file or directory", relative)
		}
	})
	if err != nil {
		return nil, fmt.Errorf("read Hermes memory state: %w", err)
	}
	return state, nil
}

func readMemoryFile(name string) ([]byte, error) {
	fd, err := syscall.Open(name, syscall.O_RDONLY|syscall.O_CLOEXEC|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return nil, err
	}
	file := os.NewFile(uintptr(fd), name)
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() > maxMemoryStateBytes {
		return nil, fmt.Errorf("memory state file %s is not a bounded regular file", name)
	}
	content, err := io.ReadAll(io.LimitReader(file, info.Size()+1))
	if err != nil || int64(len(content)) != info.Size() {
		return nil, fmt.Errorf("memory state file %s changed while being read", name)
	}
	return content, nil
}

// stageMemoryState adds the state under ARIES_MEMORY_DIR to a runtime
// archive's files and returns the directories the archive must create.
func stageMemoryState(files map[string]stagedFile, state *memoryState) []string {
	root := strings.TrimPrefix(memoryContainerDir, "/")
	directories := []string{root}
	for _, name := range state.directories {
		directories = append(directories, root+"/"+name)
	}
	for name, content := range state.files {
		files[root+"/"+name] = stagedFile{content: content, mode: 0o600}
	}
	return directories
}

// exportMemoryState copies the state directory out of the still-running
// container, retains it as the task artifact harness/memory, and, unless the
// state is frozen, makes it the run's state for the next task. It is called
// only after the one-shot process has exited, so the provider's exit commit
// has finished writing. A missing directory means the provider never
// committed and changes nothing.
func (manager *Manager) exportMemoryState(ctx context.Context, active *session) (string, error) {
	result, err := manager.client.CopyFromContainer(ctx, active.containerID, client.CopyFromContainerOptions{SourcePath: memoryContainerDir})
	if errdefs.IsNotFound(err) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("collect Hermes memory state: %w", err)
	}
	defer result.Content.Close()
	state, err := memoryStateFromArchive(result.Content, path.Base(memoryContainerDir))
	if err != nil {
		return "", fmt.Errorf("extract Hermes memory state: %w", err)
	}
	artifact := filepath.Join(active.artifactDir, memoryRunDirName)
	if err := replaceMemoryState(artifact, state); err != nil {
		return "", fmt.Errorf("retain Hermes memory state: %w", err)
	}
	if manager.memory.FrozenState == "" {
		if err := replaceMemoryState(filepath.Join(manager.outputDir, memoryRunDirName), state); err != nil {
			return "", fmt.Errorf("update run Hermes memory state: %w", err)
		}
	}
	return artifact, nil
}

// memoryStateFromArchive reads a Docker copy archive of one directory named
// root, accepting only directories and bounded regular files beneath it.
func memoryStateFromArchive(stream io.Reader, root string) (*memoryState, error) {
	state := &memoryState{files: map[string][]byte{}}
	reader := tar.NewReader(io.LimitReader(stream, maxMemoryStateBytes+int64(maxMemoryStateEntries+2)*1024))
	sawRoot := false
	for {
		header, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		name := strings.TrimSuffix(header.Name, "/")
		if name == root && header.Typeflag == tar.TypeDir {
			sawRoot = true
			continue
		}
		relative, ok := strings.CutPrefix(name, root+"/")
		if !ok {
			return nil, fmt.Errorf("archive entry %q is outside %s", header.Name, root)
		}
		switch header.Typeflag {
		case tar.TypeDir:
			if err := state.add(relative, true, nil); err != nil {
				return nil, err
			}
		case tar.TypeReg:
			if header.Size < 0 || header.Size > maxMemoryStateBytes-state.size {
				return nil, fmt.Errorf("memory state exceeds %d bytes", maxMemoryStateBytes)
			}
			content, err := io.ReadAll(io.LimitReader(reader, header.Size+1))
			if err != nil || int64(len(content)) != header.Size {
				return nil, errors.New("archive entry is truncated")
			}
			if err := state.add(relative, false, content); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("memory state entry %q is not a regular file or directory", relative)
		}
	}
	if !sawRoot {
		return nil, fmt.Errorf("archive does not contain the directory %q", root)
	}
	for name := range state.files {
		if parent := path.Dir(name); parent != "." && !slices.Contains(state.directories, parent) {
			return nil, fmt.Errorf("memory state file %q has no directory entry for its parent", name)
		}
	}
	return state, nil
}

// replaceMemoryState writes the state into a fresh private sibling directory
// and swaps it into place, so a failed write never leaves a torn state for
// the next task.
func replaceMemoryState(root string, state *memoryState) error {
	parent := filepath.Dir(root)
	if err := ensurePrivateDirectory(parent); err != nil {
		return err
	}
	temporary, err := os.MkdirTemp(parent, "."+filepath.Base(root)+".new-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(temporary)
	if err := writeMemoryState(temporary, state); err != nil {
		return err
	}
	previous := ""
	if _, err := os.Lstat(root); err == nil {
		previous = temporary + ".old"
		if err := os.Rename(root, previous); err != nil {
			return err
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.Rename(temporary, root); err != nil {
		if previous != "" {
			err = errors.Join(err, os.Rename(previous, root))
		}
		return err
	}
	if previous != "" {
		return os.RemoveAll(previous)
	}
	return nil
}

func writeMemoryState(root string, state *memoryState) error {
	if err := os.Chmod(root, 0o700); err != nil {
		return err
	}
	for _, name := range state.directories {
		if err := os.Mkdir(filepath.Join(root, filepath.FromSlash(name)), 0o700); err != nil {
			return err
		}
	}
	names := make([]string, 0, len(state.files))
	for name := range state.files {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		file, err := os.OpenFile(filepath.Join(root, filepath.FromSlash(name)), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			return err
		}
		_, writeErr := file.Write(state.files[name])
		syncErr := file.Sync()
		if err := errors.Join(writeErr, syncErr, file.Close()); err != nil {
			return err
		}
	}
	return nil
}
