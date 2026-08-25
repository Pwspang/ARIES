package openclaw

import (
	"archive/tar"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/containerd/errdefs"
	"github.com/moby/moby/client"
)

// exportLosslessClawMemory copies lossless-claw's SQLite state file out of
// the still-running OpenClaw container into this task's output directory, as
// a plain docker-cp — unlike amem's Qdrant-REST-scroll export
// (amem_qdrant.go's exportAMEMMemory), lossless-claw needs no separate
// sidecar to reach: its state lives inside the OpenClaw container's own
// filesystem at stateContainerPath+"/lcm.db".
//
// This must run from collectArtifacts (called per harness turn, while
// active.containerID is still alive), not from Manager.Close: by Close time,
// Stop has already stopped and removed the OpenClaw container (see Close's
// doc comment), so a docker-cp attempted there would always fail with "not
// found" and silently export nothing.
func (manager *Manager) exportLosslessClawMemory(ctx context.Context, active *session) error {
	result, err := manager.client.CopyFromContainer(ctx, active.containerID, client.CopyFromContainerOptions{SourcePath: stateContainerPath + "/lcm.db"})
	if err != nil {
		if errdefs.IsNotFound(err) {
			// Tolerated: a very short or failed turn may never have created
			// lcm.db at all.
			return nil
		}
		return fmt.Errorf("collect lossless-claw memory: %w", err)
	}
	defer result.Content.Close()
	archive, err := io.ReadAll(io.LimitReader(result.Content, maxDockerOutput+1))
	if err != nil || len(archive) > maxDockerOutput {
		return errors.New("lossless-claw memory archive exceeded its bound")
	}
	content, err := extractSingleFile(archive, "lcm.db")
	if err != nil {
		return fmt.Errorf("extract lossless-claw memory: %w", err)
	}
	if content == nil {
		return nil
	}
	if err := os.MkdirAll(manager.losslessClawExportDir, 0o700); err != nil {
		return fmt.Errorf("create lossless-claw memory export directory: %w", err)
	}
	path := filepath.Join(manager.losslessClawExportDir, "lcm.db")
	if err := os.WriteFile(path, content, 0o600); err != nil {
		return fmt.Errorf("write lossless-claw memory export: %w", err)
	}
	return nil
}

// extractSingleFile reads the first regular-file tar entry whose base name
// matches name out of archive, bounded by maxDockerOutput. Returns nil, nil
// if no matching entry is found.
func extractSingleFile(archive []byte, name string) ([]byte, error) {
	reader := tar.NewReader(bytes.NewReader(archive))
	for {
		header, err := reader.Next()
		if errors.Is(err, io.EOF) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		if (header.Typeflag != tar.TypeReg && header.Typeflag != tar.TypeRegA) || header.Size < 0 || header.Size > maxDockerOutput {
			continue
		}
		if filepath.Base(filepath.Clean(header.Name)) != name {
			continue
		}
		content, err := io.ReadAll(io.LimitReader(reader, header.Size+1))
		if err != nil || int64(len(content)) != header.Size {
			return nil, errors.New("archive entry is truncated")
		}
		return content, nil
	}
}
