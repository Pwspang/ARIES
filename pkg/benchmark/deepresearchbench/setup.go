package deepresearchbench

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

var renameCheckout = os.Rename

const lfsPointerPrefix = "version https://git-lfs.github.com/spec"

// DefaultRoot is the conventional local checkout path for Deep Research Bench.
const DefaultRoot = ".cache/deep-research-bench"

// Setup creates an exact shallow detached checkout at root. An existing root
// is accepted only when it is already at the pinned revision.
func Setup(ctx context.Context, root, repositoryURL, revision string) error {
	if repositoryURL == "" {
		return errors.New("deepresearchbench repository URL is required")
	}
	if revision == "" {
		return errors.New("deepresearchbench revision is required")
	}
	root = filepath.Clean(root)
	if root == "." || root == string(filepath.Separator) {
		return fmt.Errorf("unsafe deepresearchbench setup root %q", root)
	}
	info, err := os.Stat(root)
	if err == nil {
		if !info.IsDir() {
			return fmt.Errorf("deepresearchbench setup root %q is not a directory", root)
		}
		return VerifyRevision(ctx, root, revision)
	}
	if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect deepresearchbench setup root %q: %w", root, err)
	}

	parent := filepath.Dir(root)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return fmt.Errorf("create deepresearchbench cache parent: %w", err)
	}
	temporary, err := os.MkdirTemp(parent, ".deep-research-bench-setup-")
	if err != nil {
		return fmt.Errorf("create temporary deepresearchbench checkout: %w", err)
	}
	defer os.RemoveAll(temporary)

	commands := [][]string{
		{"init", "--quiet"},
		{"remote", "add", "origin", repositoryURL},
		{"fetch", "--depth=1", "origin", revision},
		{"checkout", "--quiet", "--detach", "FETCH_HEAD"},
	}
	for _, args := range commands {
		if err := runGit(ctx, temporary, args...); err != nil {
			return err
		}
	}
	if err := VerifyRevision(ctx, temporary, revision); err != nil {
		return err
	}
	if err := pullLFSIfNeeded(ctx, temporary); err != nil {
		return err
	}
	return installCheckout(ctx, temporary, root, revision)
}

// pullLFSIfNeeded runs `git lfs pull` only when the checkout actually
// contains unresolved Git LFS pointer files. The upstream Hugging Face
// dataset repository does not currently LFS-track its JSONL files, so this
// is a defensive no-op guard rather than a hard requirement.
func pullLFSIfNeeded(ctx context.Context, dir string) error {
	needsLFS := false
	err := filepath.WalkDir(dir, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if entry.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		if needsLFS {
			return nil
		}
		pointer, err := isLFSPointerFile(path)
		if err != nil {
			return err
		}
		if pointer {
			needsLFS = true
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("scan deepresearchbench checkout for LFS pointers: %w", err)
	}
	if !needsLFS {
		return nil
	}
	return runGit(ctx, dir, "lfs", "pull")
}

func isLFSPointerFile(path string) (bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return false, fmt.Errorf("open %q: %w", path, err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	if !scanner.Scan() {
		return false, nil
	}
	return strings.HasPrefix(scanner.Text(), lfsPointerPrefix), nil
}

func installCheckout(ctx context.Context, temporary, root, revision string) error {
	if err := renameCheckout(temporary, root); err != nil {
		// Another setup may have atomically installed the same pinned checkout
		// after our initial absence check. Accept only a freshly reverified
		// destination; a wrong, dirty, or partial winner remains an error.
		if verifyErr := VerifyRevision(ctx, root, revision); verifyErr == nil {
			return nil
		} else {
			return fmt.Errorf("install deepresearchbench checkout at %q: %w", root, errors.Join(err, verifyErr))
		}
	}
	return nil
}

// VerifyRevision confirms that root is the exact clean pinned checkout.
func VerifyRevision(ctx context.Context, root, revision string) error {
	cmd := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "HEAD")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("verify deepresearchbench checkout %q: %w: %s", root, err, output)
	}
	got := strings.TrimSpace(string(output))
	if got != revision {
		return fmt.Errorf("deepresearchbench checkout %q is revision %q; want pinned %q", root, got, revision)
	}
	status := exec.CommandContext(ctx, "git", "-C", root, "status", "--porcelain", "--untracked-files=all")
	output, err = status.CombinedOutput()
	if err != nil {
		return fmt.Errorf("inspect deepresearchbench checkout %q: %w: %s", root, err, output)
	}
	if len(output) != 0 {
		return fmt.Errorf("deepresearchbench checkout %q has local changes; the pinned dataset must be clean", root)
	}
	return nil
}

func runGit(ctx context.Context, dir string, args ...string) error {
	cmdArgs := append([]string{"-C", dir}, args...)
	cmd := exec.CommandContext(ctx, "git", cmdArgs...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("git %v: %w: %s", args, err, output)
	}
	return nil
}
