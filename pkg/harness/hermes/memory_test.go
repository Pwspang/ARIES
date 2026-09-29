package hermes

import (
	"archive/tar"
	"bytes"
	"context"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func newMemoryManager(t *testing.T, fake *fakeDocker, memory MemoryOptions) *Manager {
	t.Helper()
	manager := newTestManager(t, fake, []byte("model-secret"))
	manager.memory = memory
	return manager
}

// memoryArchive builds a Docker copy archive of the container's memory
// directory holding the given files, with a directory entry per parent.
func memoryArchive(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var archive bytes.Buffer
	writer := tar.NewWriter(&archive)
	written := map[string]bool{}
	directory := func(name string) {
		if !written[name] {
			written[name] = true
			if err := writer.WriteHeader(&tar.Header{Name: name + "/", Typeflag: tar.TypeDir, Mode: 0o700}); err != nil {
				t.Fatal(err)
			}
		}
	}
	directory("memory")
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		parts := strings.Split(name, "/")
		for index := 1; index < len(parts); index++ {
			directory("memory/" + strings.Join(parts[:index], "/"))
		}
		if err := writer.WriteHeader(&tar.Header{Name: "memory/" + name, Typeflag: tar.TypeReg, Mode: 0o600, Size: int64(len(files[name]))}); err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Write([]byte(files[name])); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return archive.Bytes()
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

// readTree returns every regular file under root by slash-relative path.
func readTree(t *testing.T, root string) map[string]string {
	t.Helper()
	files := map[string]string{}
	err := filepath.WalkDir(root, func(current string, entry os.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		content, err := os.ReadFile(current)
		relative, _ := filepath.Rel(root, current)
		files[filepath.ToSlash(relative)] = string(content)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return files
}

func writeTree(t *testing.T, root string, files map[string]string) {
	t.Helper()
	for name, content := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

func TestMemoryDisabledLeavesConfigArchiveAndEnvironmentUnchanged(t *testing.T) {
	fake := newFakeDocker()
	fake.memoryArchive = memoryArchive(t, map[string]string{"ignored": "x"})
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
	if _, ok := archiveEntries(t, fake.archive)[strings.TrimPrefix(memoryContainerDir, "/")]; ok {
		t.Fatal("archive stages a memory directory")
	}
	for _, variable := range fake.created.Config.Env {
		if strings.HasPrefix(variable, "ARIES_MEMORY_") {
			t.Fatalf("environment sets %s without a memory provider", variable)
		}
	}
	if len(fake.copyFrom) != 0 {
		t.Fatalf("copied from container: %v", fake.copyFrom)
	}
}

func TestMemoryProviderEnvironmentAndFirstTaskState(t *testing.T) {
	fake := newFakeDocker()
	exported := map[string]string{"notes.db": "after task one", "index/vectors.bin": "\x00\x01", "index/empty": ""}
	fake.memoryArchive = memoryArchive(t, exported)
	manager := newMemoryManager(t, fake, MemoryOptions{Provider: "my-notes", Env: map[string]string{"NOTES_TOP_K": "3", "NOTES_A": "x y"}})
	request := testRequest(t)
	if err := manager.Start(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	config, _ := stagedFileContent(t, fake.archive, strings.TrimPrefix(configContainerPath, "/"))
	if !strings.Contains(string(config), "\nmemory:\n  provider: \"my-notes\"\n") {
		t.Fatalf("config lacks the memory provider:\n%s", config)
	}
	environment := fake.created.Config.Env
	want := []string{"ARIES_MEMORY_DIR=" + memoryContainerDir, "ARIES_MEMORY_TASK_ID=" + request.TaskID, "NOTES_A=x y", "NOTES_TOP_K=3"}
	if index := slices.Index(environment, want[0]); index < 0 || !slices.Equal(environment[index:index+len(want)], want) {
		t.Fatalf("environment = %v, want suffix %v", environment, want)
	}
	directory, ok := archiveEntries(t, fake.archive)[strings.TrimPrefix(memoryContainerDir, "/")]
	if !ok || directory.Uid != runtimeUID || directory.Typeflag != tar.TypeDir || directory.Mode != 0o700 {
		t.Fatalf("memory directory entry = %#v", directory)
	}

	result, err := manager.Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	runState := filepath.Join(manager.outputDir, memoryRunDirName)
	if got := readTree(t, runState); !maps.Equal(got, exported) {
		t.Fatalf("run state = %v", got)
	}
	if info, err := os.Stat(filepath.Join(runState, "index")); err != nil || info.Mode().Perm() != 0o700 {
		t.Fatalf("run state directory mode = %v, %v", info, err)
	}
	artifact := filepath.Join(manager.outputDir, request.TaskID, "harness", "memory")
	if got := readTree(t, artifact); !maps.Equal(got, exported) {
		t.Fatalf("artifact = %v", got)
	}
	if !slices.Contains(result.LogPaths, artifact) {
		t.Fatalf("log paths %v lack %s", result.LogPaths, artifact)
	}
}

func TestMemoryNextTaskReceivesRunStateAndKeepsItWhenNothingCommits(t *testing.T) {
	fake := newFakeDocker()
	manager := newMemoryManager(t, fake, MemoryOptions{Provider: "memrl"})
	previous := map[string]string{"memrl.db": "from the previous task", "a/b/c.json": "{}"}
	writeTree(t, filepath.Join(manager.outputDir, memoryRunDirName), previous)
	if err := manager.Start(context.Background(), testRequest(t)); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	entries := archiveEntries(t, fake.archive)
	root := strings.TrimPrefix(memoryContainerDir, "/")
	for name, content := range previous {
		staged, ok := stagedFileContent(t, fake.archive, root+"/"+name)
		if !ok || string(staged) != content {
			t.Fatalf("staged %s = %q", name, staged)
		}
		if header := entries[root+"/"+name]; header.Uid != runtimeUID || header.Mode != 0o600 {
			t.Fatalf("staged %s header = %#v", name, header)
		}
	}
	for _, name := range []string{"a", "a/b"} {
		if header, ok := entries[root+"/"+name]; !ok || header.Typeflag != tar.TypeDir || header.Uid != runtimeUID {
			t.Fatalf("staged directory %s = %#v", name, header)
		}
	}
	// The provider never committed: the run state must be left as it was.
	if _, err := manager.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if got := readTree(t, filepath.Join(manager.outputDir, memoryRunDirName)); !maps.Equal(got, previous) {
		t.Fatalf("run state changed to %v", got)
	}
}

func TestMemoryExportReplacesTheWholeRunState(t *testing.T) {
	fake := newFakeDocker()
	fake.memoryArchive = memoryArchive(t, map[string]string{"kept": "new"})
	manager := newMemoryManager(t, fake, MemoryOptions{Provider: "memrl"})
	writeTree(t, filepath.Join(manager.outputDir, memoryRunDirName), map[string]string{"kept": "old", "deleted/by-provider": "old"})
	if err := manager.Start(context.Background(), testRequest(t)); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	if _, err := manager.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if got := readTree(t, filepath.Join(manager.outputDir, memoryRunDirName)); !maps.Equal(got, map[string]string{"kept": "new"}) {
		t.Fatalf("run state = %v", got)
	}
	if leftovers, _ := filepath.Glob(filepath.Join(manager.outputDir, ".memory*")); len(leftovers) != 0 {
		t.Fatalf("replacement left %v", leftovers)
	}
}

func TestMemoryFrozenStateIsStagedButNeverWritten(t *testing.T) {
	fake := newFakeDocker()
	fake.memoryArchive = memoryArchive(t, map[string]string{"memrl.db": "learned during the task"})
	frozen := filepath.Join(t.TempDir(), "stores", "trained")
	trained := map[string]string{"memrl.db": "trained"}
	writeTree(t, frozen, trained)
	manager := newMemoryManager(t, fake, MemoryOptions{Provider: "memrl", FrozenState: frozen})
	request := testRequest(t)
	if err := manager.Start(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	if staged, _ := stagedFileContent(t, fake.archive, strings.TrimPrefix(memoryContainerDir, "/")+"/memrl.db"); string(staged) != "trained" {
		t.Fatalf("staged = %q", staged)
	}
	if _, err := manager.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if got := readTree(t, frozen); !maps.Equal(got, trained) {
		t.Fatalf("frozen state changed to %v", got)
	}
	if _, err := os.Stat(filepath.Join(manager.outputDir, memoryRunDirName)); !os.IsNotExist(err) {
		t.Fatalf("a frozen run wrote run state: %v", err)
	}
	// The task's own copy is still retained for analysis.
	if got := readTree(t, filepath.Join(manager.outputDir, request.TaskID, "harness", "memory")); got["memrl.db"] != "learned during the task" {
		t.Fatalf("artifact = %v", got)
	}
}

func TestMemoryMissingFrozenStateFailsTheTask(t *testing.T) {
	manager := newMemoryManager(t, newFakeDocker(), MemoryOptions{Provider: "memrl", FrozenState: filepath.Join(t.TempDir(), "missing")})
	if err := manager.Start(context.Background(), testRequest(t)); err == nil {
		t.Fatal("start succeeded without the frozen state")
	}
}

func TestMemoryRejectsHostStateThatIsNotPlainFiles(t *testing.T) {
	manager := newMemoryManager(t, newFakeDocker(), MemoryOptions{Provider: "memrl"})
	root := filepath.Join(manager.outputDir, memoryRunDirName)
	writeTree(t, root, map[string]string{"memrl.db": "x"})
	if err := os.Symlink("/etc/passwd", filepath.Join(root, "link")); err != nil {
		t.Fatal(err)
	}
	if err := manager.Start(context.Background(), testRequest(t)); err == nil || !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("start with a symlinked state entry: %v", err)
	}
}

func TestMemoryRejectsUnsafeExports(t *testing.T) {
	build := func(headers ...tar.Header) []byte {
		var archive bytes.Buffer
		writer := tar.NewWriter(&archive)
		for _, header := range headers {
			if err := writer.WriteHeader(&header); err != nil {
				t.Fatal(err)
			}
			if header.Size > 0 {
				if _, err := writer.Write(bytes.Repeat([]byte("x"), int(header.Size))); err != nil {
					t.Fatal(err)
				}
			}
		}
		if err := writer.Close(); err != nil {
			t.Fatal(err)
		}
		return archive.Bytes()
	}
	root := tar.Header{Name: "memory/", Typeflag: tar.TypeDir, Mode: 0o700}
	for name, archive := range map[string][]byte{
		"symlink":         build(root, tar.Header{Name: "memory/link", Typeflag: tar.TypeSymlink, Linkname: "/etc/passwd"}),
		"hard link":       build(root, tar.Header{Name: "memory/link", Typeflag: tar.TypeLink, Linkname: "memory/a"}),
		"escape":          build(root, tar.Header{Name: "memory/../escape", Typeflag: tar.TypeReg, Mode: 0o600, Size: 1}),
		"outside root":    build(root, tar.Header{Name: "other/file", Typeflag: tar.TypeReg, Mode: 0o600, Size: 1}),
		"no root":         build(tar.Header{Name: "memory/file", Typeflag: tar.TypeReg, Mode: 0o600, Size: 1}),
		"missing parent":  build(root, tar.Header{Name: "memory/a/file", Typeflag: tar.TypeReg, Mode: 0o600, Size: 1}),
		"oversize header": build(root, tar.Header{Name: "memory/big", Typeflag: tar.TypeReg, Mode: 0o600, Size: maxMemoryStateBytes + 1}),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := memoryStateFromArchive(bytes.NewReader(archive), "memory"); err == nil {
				t.Fatal("unsafe archive was accepted")
			}
		})
	}
	state, err := memoryStateFromArchive(bytes.NewReader(build(root, tar.Header{Name: "memory/a/", Typeflag: tar.TypeDir, Mode: 0o700}, tar.Header{Name: "memory/a/f", Typeflag: tar.TypeReg, Mode: 0o600, Size: 2})), "memory")
	if err != nil || !slices.Equal(state.directories, []string{"a"}) || string(state.files["a/f"]) != "xx" {
		t.Fatalf("state = %#v, %v", state, err)
	}
}

func TestMemoryRejectedExportLeavesRunStateAlone(t *testing.T) {
	fake := newFakeDocker()
	var archive bytes.Buffer
	writer := tar.NewWriter(&archive)
	_ = writer.WriteHeader(&tar.Header{Name: "memory/", Typeflag: tar.TypeDir, Mode: 0o700})
	_ = writer.WriteHeader(&tar.Header{Name: "memory/link", Typeflag: tar.TypeSymlink, Linkname: "/"})
	_ = writer.Close()
	fake.memoryArchive = archive.Bytes()
	manager := newMemoryManager(t, fake, MemoryOptions{Provider: "memrl"})
	previous := map[string]string{"memrl.db": "previous"}
	writeTree(t, filepath.Join(manager.outputDir, memoryRunDirName), previous)
	if err := manager.Start(context.Background(), testRequest(t)); err != nil {
		t.Fatal(err)
	}
	defer manager.Stop(context.Background())
	if _, err := manager.Run(context.Background(), "task"); err == nil || !strings.Contains(err.Error(), "memory state") {
		t.Fatalf("run with an unsafe export: %v", err)
	}
	if got := readTree(t, filepath.Join(manager.outputDir, memoryRunDirName)); !maps.Equal(got, previous) {
		t.Fatalf("run state changed to %v", got)
	}
}

// A canceled one-shot may still be writing, so its state is not trusted.
func TestMemorySkipsExportWhenRunDidNotExit(t *testing.T) {
	fake := newFakeDocker()
	fake.memoryArchive = memoryArchive(t, map[string]string{"memrl.db": "possibly torn"})
	manager := newMemoryManager(t, fake, MemoryOptions{Provider: "memrl"})
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
	if _, err := os.Stat(filepath.Join(manager.outputDir, memoryRunDirName)); !os.IsNotExist(err) {
		t.Fatalf("run state written after cancellation: %v", err)
	}
}

func TestMemoryOptionsValidation(t *testing.T) {
	if err := (MemoryOptions{Provider: "memrl", Env: map[string]string{"MEMRL_K2": "3"}}).validate("DEEPSEEK_API_KEY"); err != nil {
		t.Fatal(err)
	}
	for name, options := range map[string]MemoryOptions{
		"env without provider":    {Env: map[string]string{"A": "1"}},
		"frozen without provider": {FrozenState: "/tmp/x"},
		"YAML in provider":        {Provider: "memrl\"\nx: y"},
		"uppercase provider":      {Provider: "MemRL"},
		"model key":               {Provider: "memrl", Env: map[string]string{"DEEPSEEK_API_KEY": "x"}},
		"ARIES name":              {Provider: "memrl", Env: map[string]string{"ARIES_MEMORY_DIR": "/"}},
		"Hermes name":             {Provider: "memrl", Env: map[string]string{"HERMES_HOME": "/"}},
		"terminal name":           {Provider: "memrl", Env: map[string]string{"TERMINAL_CWD": "/"}},
		"Tavily key":              {Provider: "memrl", Env: map[string]string{tavilyAPIKeyEnv: "x"}},
		"offline flag":            {Provider: "memrl", Env: map[string]string{"HF_HUB_OFFLINE": "0"}},
		"lowercase name":          {Provider: "memrl", Env: map[string]string{"k": "1"}},
	} {
		if err := options.validate("DEEPSEEK_API_KEY"); err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
}

func TestMemoryEnvCarryingTheModelKeyIsRejected(t *testing.T) {
	manager := newMemoryManager(t, newFakeDocker(), MemoryOptions{Provider: "memrl", Env: map[string]string{"NOTES_TOKEN": "model-secret"}})
	if err := manager.Start(context.Background(), testRequest(t)); err == nil || !strings.Contains(err.Error(), "API-key value") {
		t.Fatalf("start with the key in memory env: %v", err)
	}
}
