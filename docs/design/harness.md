# AgentHarness

`AgentHarness` owns agent execution. It receives a task instruction, model
configuration, and temporary bridge connection details, then starts the agent,
runs it within the configured deadline, and stops it positively before bridge
revocation or evaluation can proceed.

## Boundary and lifecycle

The harness does not own the benchmark task container, verifier material,
bridge listener, or evaluation. It may retain its own logs, telemetry, and
rendered configuration as private run artifacts. Model credentials are supplied
at runtime and must not be written into profiles, structured logs, Docker
metadata, or results.

The current implementation is OpenClaw in a pinned container image. ARIES starts
the harness only after the sandbox and bridge are ready. On success, failure, or
cancellation, it stops the harness and confirms absence. Evaluation remains a
separate Benchmark outcome rather than an interpretation of harness success.

OpenClaw container lifecycle, gateway protocol, and voice-session semantics are
separate concrete responsibilities. `openclaw.Manager` owns the container and
publishes one ephemeral host-loopback port. `openclaw/gateway.Client` owns one
authenticated WebSocket protocol connection and bounded fail-closed frame
dispatch. `openclaw/realtime.Runner` consumes that client for talk/chat/tool
session semantics; the gateway package intentionally contains no voice event
parsers or Docker lifecycle.

Text and realtime modes share the authenticated Gateway transport. Text sends
exactly one pinned-version `agent` request and correlates accepted and terminal
responses by both frame request ID and non-empty run ID; an ambiguous send,
disconnect, timeout, or protocol mismatch is never retried. Realtime requires
read and write scopes, while text requires write scope. Only sanitized role and
sorted scope metadata may leave the authentication boundary.

Realtime converts the task instruction to staged audio, owns one talk session,
and may invoke nested agent runs through the same authenticated client. Its
audio, transcript, result, and optional event records remain private harness
artifacts. The separate realtime/TTS credential is staged privately for voice
mode and is not part of model configuration, Gateway authentication, or
structured results.
Both modes remain concrete behavior of the single `AgentHarness` role; realtime
does not create a fifth Runner role or take ownership from the benchmark,
sandbox, or bridge.

## Hermes

Hermes is the second supported harness and runs the pinned upstream image
unmodified. It is text mode only; `harness.mode: "realtime"` remains OpenClaw's.

`hermes.Manager` owns one container held at an idle command, so ARIES decides
when the agent starts rather than the image entrypoint. One task instruction is
delivered by executing a staged wrapper that runs the Hermes one-shot
(`hermes --ignore-rules --yolo --model … --provider … -z …`) and reports its
status through a delimited exit trailer. The instruction is passed as a single
argument vector element, never interpolated into a shell string. `--toolsets` is
deliberately not passed: on the pinned version its validator can return a bare
`None` that the caller unpacks, so the agent would exit before doing any work.
Toolsets come from the rendered configuration instead.

Hermes exposes no control protocol, so there is no gateway client. The final
response is the one-shot's standard output, and the message-level trajectory is
Hermes's own SQLite session store exported to standard output. Container logs,
both output streams, the redacted configuration, and the session export are
private harness artifacts.

Hermes reads its tool backend only from environment variables, so ARIES sets
`TERMINAL_ENV=ssh` with the bridge's host, port, user, and identity path; this
is upstream's native SSH environment, not an ARIES modification. `HERMES_HOME`
is relocated to a staged private directory so the image's declared `/opt/data`
volume holds no run state; that one anonymous volume is still created by Docker
and is the only mount the harness tolerates. The model credential is written to
the rendered configuration as a `${NAME}` reference, staged separately as a
private key file, and exported by the wrapper inside the container, so no
credential value reaches the configuration, Docker metadata, or results.

Two upstream details are load-bearing. First, `/opt/hermes/bin/hermes` sits
earliest on `PATH` and is a privilege-drop shim: invoked as root it re-execs the
real binary as the image's unprivileged `hermes` user. Everything ARIES stages
is therefore owned by that user, asserted by the container's own start command
because the Engine's copy API resets archive ownership to root. Staging as root
instead leaves the agent unable to read its own configuration, and the failure
surfaces far from its cause. The readiness probe runs `hermes --version` rather
than only stat-ing the staged files, because those checks run as root and pass
regardless of ownership.

Second, Hermes requires `/bin/bash` in the task image, because every tool call
it issues is `bash -c` on the remote.

`harness.memrl.enabled` switches on the MemRL memory provider. The provider is
baked into the `docker/hermes-memrl` image, which a versions file such as
`configs/versions-memrl.json` pins. ARIES adds `memory.provider: memrl` to the
rendered config and keeps one run-scoped SQLite store at
`<run>/memrl/memrl.db`. The Hermes container still takes no mounts, so the store
is handed from task to task by copying:

- At start, the current store is staged into `HERMES_HOME` with the rest of the
  private runtime archive.
- After the one-shot process exits, the store is copied out, checked to be an
  SQLite file, retained as `harness/memrl/memrl.db`, and atomically replaces the
  run copy.
- A canceled or timed-out run is not copied back, because the process may still
  be writing.

Because the hand-off is sequential, the profile must set
`execution.concurrency` to 1, unless it sets `batch_size` or `frozen_store`
(below).

MemRL learns from the benchmark's verdict, which only exists after evaluation,
when the task's container is gone. So the harness starts the provider with
`MEMRL_REWARD_SOURCE=external` and the task execution ID, and the provider parks
the finished session. After each occurrence, the command wiring's
`RecordTaskOutcome` hook writes the verdict to `<run>/memrl/rewards.json`: +1
for full reward, −1 otherwise, and null when evaluation reached no verdict. The
next task's provider applies those rewards before its first recall. The harness
never sees an evaluation result during its own task, so evaluation stays
independent of it.

`harness.memrl.frozen_store` evaluates a trained store without learning. Every
task starts from that directory's `memrl.db`, the provider runs with
`MEMRL_FROZEN=1`, and `RecordTaskOutcome` records no verdicts. Nothing is
written back, so concurrency is allowed.

`harness.memrl.batch_size` trains in mini-batches. The run loop admits the
task list in batches of that size (up to `execution.concurrency` at once) and
calls the command wiring's `FinishTaskBatch` hook between batches. Each task
starts from the run's store and parks its session instead of handing the store
on. The harness queues the parked session under `<run>/memrl/pending`. The hook
runs `FinalizeMemRLBatch`, a short-lived container of the same image that
receives the model key as a staged file, like a task container. It applies the
batch's rewards and replaces the run's store, and the applied sessions move to
`<run>/memrl/batches/NNN`. A failed batch update stops the run's admissions
and leaves the queue in place.

## Customization & Contribution Guide

Add a harness only when it can implement the existing `AgentHarness` lifecycle
without taking ownership from the other roles. Keep the implementation in a
concrete harness package with an explicit constructor, add its explicit switch
to the command wiring, and provide focused tests for start, run, cancellation,
idempotent stop, positive absence, credential handling, and artifacts. Update
the supported reference and this guide with evidenced behavior; do not add a
registration, discovery, factory, reflection, DI, or generic plugin layer.
