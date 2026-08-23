package docker

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// dockerSocketEnvVar overrides the Docker Engine socket when it isn't
// reachable at the conventional /var/run/docker.sock path — for example
// rootless Podman, which exposes a per-user socket instead of the
// system-wide one that path resolves to.
const dockerSocketEnvVar = "ARIES_DOCKER_SOCKET"

// ResolveDockerHost turns a configured socket into a host string suitable for
// client.WithHost. An empty configured value falls back to
// ARIES_DOCKER_SOCKET, then to defaultDockerSocket. A value that already
// carries a scheme (e.g. "unix://", "tcp://") is returned unchanged; a bare
// path is resolved to an absolute path and prefixed with "unix://".
func ResolveDockerHost(configured string) (string, error) {
	socket := configured
	if socket == "" {
		socket = os.Getenv(dockerSocketEnvVar)
	}
	if socket == "" {
		socket = defaultDockerSocket
	}
	if strings.Contains(socket, "://") {
		return socket, nil
	}
	absolute, err := filepath.Abs(socket)
	if err != nil {
		return "", fmt.Errorf("resolve Docker socket path: %w", err)
	}
	return "unix://" + absolute, nil
}
