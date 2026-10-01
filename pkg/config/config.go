package config

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/hyscale-lab/aries/pkg/containerimage"
	"github.com/hyscale-lab/aries/pkg/core"
)

const defaultOutputDir = "runs"

// Config is one explicit experiment profile. It has no inheritance, templates,
// merging, or secret values.
type Config struct {
	Name          string           `json:"name"`
	VersionsFile  string           `json:"versions_file"`
	OverridesFile string           `json:"overrides_file,omitempty"`
	Benchmark     BenchmarkConfig  `json:"benchmark"`
	Harness       HarnessConfig    `json:"harness"`
	Sandbox       SandboxConfig    `json:"sandbox"`
	Bridge        BridgeConfig     `json:"bridge"`
	Runtime       RuntimeConfig    `json:"runtime"`
	Model         ProfileModel     `json:"model"`
	Execution     ExecutionConfig  `json:"execution,omitempty"`
	OutputDir     string           `json:"output_dir"`
	Versions      Versions         `json:"-"`
	Overrides     RuntimeOverrides `json:"-"`
}

// ExecutionConfig controls bounded occurrence scheduling above the Runner.
type ExecutionConfig struct {
	Concurrency  int           `json:"concurrency"`
	LoopDuration string        `json:"loop_duration,omitempty"`
	Loop         time.Duration `json:"-"`
	// ArrivalsFile and ArrivalRatePerMin turn the run into an open loop: each
	// task starts at the offset the trace gives it, scaled from the trace's
	// base rate to ArrivalRatePerMin, instead of as soon as a worker is free.
	// Concurrency still caps the tasks in flight; set it at or above the task
	// count so the schedule, not the pool, decides when a task starts.
	ArrivalsFile      string  `json:"arrivals_file,omitempty"`
	ArrivalRatePerMin float64 `json:"arrival_rate_per_min,omitempty"`
}

// RuntimeConfig selects the model service. Mode is the ownership distinction:
// "external" is an endpoint ARIES validates but never starts, configures, or
// stops; "managed" is a host process ARIES owns for the run. Backend names the
// kind of service behind the endpoint, not a runtime ARIES prepares: it
// selects the preflight and the provider each harness renders. "deepseek" is
// the official DeepSeek endpoint with its own preflight; "openai" is any other
// OpenAI-compatible server (vLLM, llama.cpp, a gateway) with generic /v1/models
// discovery; both are external only. "sglang" shares that discovery, may be
// external or managed; only managed mode requires a native YAML launch file.
type RuntimeConfig struct {
	Backend string              `json:"backend"`
	Mode    string              `json:"mode"`
	Config  RuntimeConfigValues `json:"config,omitempty"`
}

type RuntimeConfigValues struct {
	File               string        `json:"file,omitempty"`
	Executable         string        `json:"executable,omitempty"`
	StartupTimeoutText string        `json:"startup_timeout,omitempty"`
	StopTimeoutText    string        `json:"stop_timeout,omitempty"`
	StartupTimeout     time.Duration `json:"-"`
	StopTimeout        time.Duration `json:"-"`
	ResolvedFile       string        `json:"-"`
	GPUIndices         []int         `json:"gpu_indices,omitempty"`
}

// RuntimeOverrides contains sparse, explicitly present runtime changes.
type RuntimeOverrides struct {
	HarnessResources      ResourceOverrides `json:"harness_resources,omitempty"`
	AgentSandboxResources ResourceOverrides `json:"agent_sandbox_resources,omitempty"`
	AgentTimeoutSeconds   *float64          `json:"agent_timeout_seconds,omitempty"`
	AgentTimeout          *time.Duration    `json:"-"`
	// VerifierTimeoutFloorSeconds raises a Terminal-Bench task's verifier
	// budget to at least this value. It never lowers a declared budget.
	VerifierTimeoutFloorSeconds *float64       `json:"verifier_timeout_floor_seconds,omitempty"`
	VerifierTimeoutFloor        *time.Duration `json:"-"`
}

// ResourceOverrides changes only the named container resource dimensions.
type ResourceOverrides struct {
	CPU      *float64 `json:"cpu,omitempty"`
	MemoryMB *int     `json:"memory_mb,omitempty"`
}

type ProfileModel struct {
	ID        string `json:"id"`
	BaseURL   string `json:"base_url"`
	APIKeyEnv string `json:"api_key_env"`
	// MaxOutputTokens bounds the harness model's completion length; see
	// core.ModelConfig.MaxOutputTokens. Zero (the default) leaves the
	// provider's own default in place.
	MaxOutputTokens int `json:"max_output_tokens,omitempty"`
	// ContextWindowTokens overrides the harness's belief about its context
	// window size; see core.ModelConfig.ContextWindowTokens. Zero (the
	// default) leaves the harness's own default in place.
	ContextWindowTokens int `json:"context_window_tokens,omitempty"`
	// CompactionTimeoutMs overrides how long the harness's auto-compaction
	// call may run before being aborted as hung; see
	// core.ModelConfig.CompactionTimeoutMs. Zero (the default) leaves the
	// harness's own default (180000ms) in place.
	CompactionTimeoutMs int `json:"compaction_timeout_ms,omitempty"`
	// ContextLength, MaxTokens, and Temperature are optional. They reach the
	// harness through core.ModelConfig. Only Hermes renders them, so a
	// profile that sets one under another harness is rejected.
	ContextLength int      `json:"context_length,omitempty"`
	MaxTokens     int      `json:"max_tokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
}

// validateGeneration checks the optional generation settings against the
// selected harness. Unset fields are always valid.
func (m ProfileModel) validateGeneration(harnessType string) error {
	set := m.ContextLength != 0 || m.MaxTokens != 0 || m.Temperature != nil
	if !set {
		return nil
	}
	if harnessType != "hermes" {
		return errors.New("model.context_length, model.max_tokens, and model.temperature require Hermes")
	}
	if m.ContextLength < 0 {
		return errors.New("model.context_length must be positive")
	}
	if m.MaxTokens < 0 {
		return errors.New("model.max_tokens must be positive")
	}
	if m.ContextLength > 0 && m.MaxTokens > 0 && m.MaxTokens >= m.ContextLength {
		return errors.New("model.max_tokens must be smaller than model.context_length")
	}
	if m.Temperature != nil {
		t := *m.Temperature
		if math.IsNaN(t) || math.IsInf(t, 0) || t < 0 || t > 2 {
			return errors.New("model.temperature must be between 0 and 2")
		}
	}
	return nil
}

type BenchmarkConfig struct {
	Type        string                `json:"type"`
	Root        string                `json:"root"`
	Tasks       []string              `json:"tasks"`
	Environment *BenchmarkEnvironment `json:"environment,omitempty"`
	Judge       *JudgeConfig          `json:"judge,omitempty"`
	Fact        *FactConfig           `json:"fact,omitempty"`

	// PlanOnly and PlansetDir configure Deep Research Bench's plan-generation
	// pass (see pkg/benchmark/deepresearchbench.Options.PlanOnly): each task
	// is wrapped in a minimal decomposition prompt instead of the full
	// research prompt, and Evaluate captures the resulting plan into
	// PlansetDir instead of running RACE/FACT. Mutually exclusive with
	// StructuredSubtasks (see validateBenchmarkType).
	PlanOnly   bool   `json:"plan_only,omitempty"`
	PlansetDir string `json:"planset_dir,omitempty"`

	// StructuredSubtasks configures Deep Research Bench's structured
	// execution pass (see
	// pkg/benchmark/deepresearchbench.Options.StructuredSubtasks): nil (the
	// default) keeps today's single-turn behavior. Mutually exclusive with
	// PlanOnly.
	StructuredSubtasks *StructuredSubtasksConfig `json:"structured_subtasks,omitempty"`
}

// StructuredSubtasksConfig mirrors
// pkg/benchmark/deepresearchbench.StructuredSubtasksOptions. PlansetDir
// reuses BenchmarkConfig.PlansetDir's own json tag ("planset_dir") deliberately:
// both name the same "directory of per-task plan files" concept, just for
// different passes over the same task set, so there is no reason to give
// them distinct field names.
type StructuredSubtasksConfig struct {
	PlansetDir string `json:"planset_dir"`
	Order      string `json:"order"`
	Seed       int64  `json:"seed,omitempty"`
}

// BenchmarkEnvironment describes the task sandbox for benchmarks (currently
// only Deep Research Bench) that have no per-task environment source of
// their own, unlike Terminal-Bench 2's task.toml and SWE-bench Pro's dataset
// rows. AllowNetwork is
// deliberately not configurable here: Deep Research Bench forces it on
// unconditionally because its tasks are open-ended web research.
type BenchmarkEnvironment struct {
	Image     string            `json:"image"`
	Workdir   string            `json:"workdir,omitempty"`
	CPU       float64           `json:"cpu,omitempty"`
	MemoryMB  int               `json:"memory_mb,omitempty"`
	StorageMB int               `json:"storage_mb,omitempty"`
	Env       map[string]string `json:"env,omitempty"`
}

// JudgeConfig identifies the LLM used to grade Deep Research Bench reports
// or SWE-Atlas QA answers (see BenchmarkConfig.Judge). It is intentionally a
// distinct type from ProfileModel so judge-specific fields can be added
// later without colliding with the harness model's shape.
//
// Enabled is a master switch for all LLM-based grading. For
// deepresearchbench, setting it to false also disables FACT (see
// BenchmarkConfig.Fact), even if a fact block is separately configured — a
// still-present fact block is silently skipped rather than rejected in that
// case (see deepresearchbench.New's FactSkipReason). For sweatlasqa, setting
// it to false skips rubric grading entirely (see sweatlas.New's
// JudgeDisabled). It is a pointer so "unset" (defaults to enabled, matching
// today's always-on grading behavior) is distinguishable from an explicit
// "false", mirroring HarnessSubagentsConfig.Enabled. The other fields must
// be left empty when Enabled is false, since they would otherwise be
// meaningless.
type JudgeConfig struct {
	Enabled   *bool  `json:"enabled,omitempty"`
	Provider  string `json:"provider"`
	BaseURL   string `json:"base_url"`
	ID        string `json:"model"`
	APIKeyEnv string `json:"api_key_env"`
}

func (j JudgeConfig) CoreModel() core.ModelConfig {
	return core.ModelConfig{Provider: j.Provider, BaseURL: j.BaseURL, Model: j.ID, APIKeyEnv: j.APIKeyEnv}
}

// FactConfig identifies the LLM used for the FACT citation-extraction/
// deduplication/validation pipeline, plus the Jina AI Reader API key used to
// scrape cited URLs (see BenchmarkConfig.Fact). The model fields
// (Provider/BaseURL/ID/APIKeyEnv) are optional as a group and default to the
// profile's main model config when all left empty; JinaAPIKeyEnv is always
// required to enable FACT at all.
type FactConfig struct {
	Provider      string `json:"provider"`
	BaseURL       string `json:"base_url"`
	ID            string `json:"model"`
	APIKeyEnv     string `json:"api_key_env"`
	JinaAPIKeyEnv string `json:"jina_api_key_env"`
}

func (f FactConfig) CoreModel() core.ModelConfig {
	return core.ModelConfig{Provider: f.Provider, BaseURL: f.BaseURL, Model: f.ID, APIKeyEnv: f.APIKeyEnv}
}

type HarnessConfig struct {
	Type            string                       `json:"type"`
	Mode            string                       `json:"mode,omitempty"`
	Realtime        HarnessRealtimeConfig        `json:"realtime,omitempty"`
	VoiceTranscribe HarnessVoiceTranscribeConfig `json:"voice_transcribe,omitempty"`
	WebSearch       HarnessWebSearchConfig       `json:"web_search,omitempty"`
	Subagents       HarnessSubagentsConfig       `json:"subagents,omitempty"`
	// MCPServers configures external or in-harness Model Context Protocol (MCP) servers
	// for Hermes and OpenClaw harnesses.
	MCPServers []core.MCPServerConfig `json:"mcp_servers,omitempty"`
	// Compaction is rendered only by Hermes today (see
	// (*HarnessConfig).validateHermesBlocks) but names a general harness
	// capability, so it lives on the shared struct and is gated by an
	// explicit type check, same as WebSearch above. An absent block keeps
	// Hermes's own defaults.
	Compaction *HarnessCompactionConfig `json:"compaction,omitempty"`
	// Hermes holds settings that exist only because of how Hermes is
	// configured and mean nothing to another harness. Such escape hatches go
	// under this type-specific block rather than onto the shared fields.
	Hermes       *HarnessHermesConfig      `json:"hermes,omitempty"`
	AMEM         HarnessAMEMConfig         `json:"amem,omitempty"`
	LosslessClaw HarnessLosslessClawConfig `json:"lossless_claw,omitempty"`
	Mem0         HarnessMem0Config         `json:"mem0,omitempty"`
	Memory       HarnessMemoryConfig       `json:"memory,omitempty"`
}

// HarnessMemoryConfig selects a Hermes memory manager: a memory provider
// plugin baked into the Hermes image the versions file pins, such as one
// built from the separate AgentMemory repository. ARIES renders
// memory.provider into config.yaml, passes Env to the container verbatim
// (provider tuning, never secrets), and carries one opaque state directory
// from task to task under <run>/memory, so changing the memory system needs
// only a new image, not a new ARIES binary. That hand-off is sequential by
// construction, hence execution.concurrency must be 1 unless FrozenState is
// set.
//
// FrozenState is a directory every task starts from, such as a copy of a
// previous run's <run>/memory; no task writes it back, so frozen tasks may
// run concurrently. Like output_dir, a relative path is resolved from the
// working directory. It is read when each task starts, so a profile can name
// a state that another run has yet to produce.
type HarnessMemoryConfig struct {
	Provider    string            `json:"provider,omitempty"`
	Env         map[string]string `json:"env,omitempty"`
	FrozenState string            `json:"frozen_state,omitempty"`
}

// HarnessAMEMConfig enables the amem memory plugin (https://amem.owo.lc, npm
// "openclaw-amem") as an OpenClaw plugin (see pkg/harness/openclaw/config.go's
// amemPluginConfig). OpenClaw-only, and requires a pinned OpenClaw image with
// the plugin pre-installed (docker/openclaw-amem) plus a pinned Qdrant image
// (Versions.AMEMQdrant) — see that image's Dockerfile comment for why.
//
// By default amem's own internal LLM calls (note-metadata extraction, merge/
// contradiction adjudication) reuse the profile's primary task model and API
// key — see amemPluginConfig's doc comment. The LLM* fields below override
// that with a separate model/endpoint instead, e.g. because the primary
// model doesn't reliably produce the plain JSON amem asks for (observed with
// at least one sglang-served Qwen deployment: it prefaces answers with prose
// instead of raw JSON, and amem's own lenient parser still fails on that,
// silently defaulting to empty note metadata and zero merges/links). All
// three LLM* fields must be set together or not at all — see (*HarnessConfig)
// validate.
type HarnessAMEMConfig struct {
	Enabled      bool   `json:"enabled,omitempty"`
	LLMBaseURL   string `json:"llm_base_url,omitempty"`
	LLMModel     string `json:"llm_model,omitempty"`
	LLMAPIKeyEnv string `json:"llm_api_key_env,omitempty"`
	// Scope selects how amem's Qdrant memory store is shared across task
	// occurrences in a run: "" or "task" (default) gives every task
	// occurrence its own private store, torn down at the end of that task —
	// this is the only behavior that existed before this field was added.
	// "repo" shares one store across every task occurrence in the run that
	// targets the same repository at the same commit (currently only
	// pkg/benchmark/sweatlas tasks carry that metadata; tasks without it
	// silently fall back to task scope), so a later task's agent can see
	// notes an earlier one already stored about that codebase instead of
	// re-discovering it from scratch. "global" shares one store across every
	// task occurrence in the run regardless of repository or commit, to test
	// whether memory transfers across different codebases (as opposed to
	// "repo" scope, which by construction can never share anything across a
	// repository boundary). See pkg/harness/openclaw/amem_qdrant.go and
	// amem_pool.go.
	Scope string `json:"scope,omitempty"`
	// RerankModel/RerankThreshold/RerankDevice configure the fork's post-BFS
	// cross-encoder reranker (third_party/amem-fork/packages/amem-core/src/
	// reranker.ts). Unlike LLM*/Scope above, these are not part of the
	// plugin's JSON config schema — the fork reads them as plain process
	// environment variables (AMEM_RERANK_MODEL/AMEM_RERANK_THRESHOLD/
	// AMEM_RERANK_DEVICE), so ARIES threads them into the container's
	// environment directly (see pkg/harness/openclaw/harness.go's
	// containerConfig.Env) rather than into amemPluginConfig. Each requires
	// only AMEM.Enabled, independent of the others and of LLM*/Scope.
	RerankModel string `json:"rerank_model,omitempty"`
	// RerankThreshold is a pointer so an explicit 0 (admit every BFS-expanded
	// candidate) is distinguishable from unset (accept the fork's own
	// default, DEFAULT_RERANK_THRESHOLD).
	RerankThreshold *float64 `json:"rerank_threshold,omitempty"`
	RerankDevice    string   `json:"rerank_device,omitempty"`
	// DisableTaskTraceFallback opts out of the fork's Method 2 (previous-turn
	// goal-trace fallback for memory_search's current_task): unlike the
	// rerank knobs above, this IS a real plugin JSON config field
	// (AmemPluginConfig.disableTaskTraceFallback in the fork's storage.ts),
	// rendered into amemPluginConfig (pkg/harness/openclaw/config.go) rather
	// than threaded as an env var.
	DisableTaskTraceFallback bool `json:"disable_task_trace_fallback,omitempty"`
	// DisableGoalEmbedding forces the fork's Method 1 (goal-aware embedding)
	// off, ignoring any current_task the agent supplies on a memory_search
	// call, regardless of memory_search's own current_task field or the
	// Method 2 trace-fallback cache. Threaded as AMEM_DISABLE_GOAL_EMBED
	// (env var, like the rerank knobs above) rather than plugin JSON config,
	// since current_task itself is an agent-discretionary tool field with no
	// other way to force it off — this exists specifically so an ablation
	// study's control arm can have current_task genuinely have no effect,
	// not just rely on the agent never choosing to pass it.
	DisableGoalEmbedding bool `json:"disable_goal_embedding,omitempty"`
	// DisableRerank skips the post-BFS cross-encoder reranker pass entirely
	// (AMEM_RERANK_ENABLED=false), unlike RerankThreshold above which only
	// tunes what the reranker filters — this is the true on/off switch an
	// ablation study's control arm needs, so it pays neither the reranker's
	// latency nor its reordering, not just none of its filtering.
	DisableRerank bool `json:"disable_rerank,omitempty"`
	// EpisodicBootstrap appends a second, additive mandate (amemEpisodicBootstrapInstruction
	// in pkg/benchmark/sweatlas) requiring the agent to log failed attempts —
	// a command that errored, a hypothesis that turned out wrong, a dead-end
	// search — via amem's memory_add_episodic tool, the fork's cheap raw-event
	// write path (third_party/amem-fork/packages/openclaw-amem/src/index.ts),
	// distinct from memory_add's LLM-distilled fact/conclusion path that
	// AMEMBootstrap alone (via Enabled) already mandates. Meaningless without
	// Enabled also set; requires no new plugin config, tool wiring, or Qdrant
	// pool changes, since memory_add_episodic is already allow-listed
	// (amemToolNames) and shares whatever Scope the arm already selects.
	EpisodicBootstrap bool `json:"episodic_bootstrap,omitempty"`
}

// HarnessLosslessClawConfig enables the lossless-claw context-management
// plugin (github.com/Martian-Engineering/lossless-claw, npm
// "@martian-engineering/lossless-claw") as an OpenClaw plugin claiming the
// "contextEngine" slot (see pkg/harness/openclaw/config.go's
// losslessClawPluginConfig) — mutually exclusive with AMEM (see
// (*HarnessConfig).validate), which claims the separate "memory" slot.
// Unlike amem, lossless-claw needs no sidecar container: it persists to a
// SQLite file inside the OpenClaw container's own state directory, exported
// as an artifact at harness Close() instead of scraped from a vector DB.
//
// By default lossless-claw's own summarization/expansion calls reuse the
// profile's primary task model (OpenClaw's own default resolution — see
// losslessClawPluginConfig's doc comment). The LLM* fields below override
// that with a separate model/endpoint instead, registered as an additional
// OpenClaw model provider. All three LLM* fields must be set together or not
// at all — see (*HarnessConfig).validate.
type HarnessLosslessClawConfig struct {
	Enabled      bool   `json:"enabled,omitempty"`
	LLMBaseURL   string `json:"llm_base_url,omitempty"`
	LLMModel     string `json:"llm_model,omitempty"`
	LLMAPIKeyEnv string `json:"llm_api_key_env,omitempty"`
}

// HarnessMem0Config enables mem0's first-party OpenClaw plugin
// (github.com/mem0ai/mem0, integrations/openclaw, npm "@mem0/openclaw-mem0")
// as an OpenClaw plugin claiming the "memory" slot (see
// pkg/harness/openclaw/config.go's mem0PluginConfig) — mutually exclusive
// with AMEM (see (*HarnessConfig).validate), which claims the same slot.
//
// Mode selects which of the plugin's two backends is used (confirmed
// against the plugin's shipped openclaw.plugin.json configSchema, whose
// "mode" enum is exactly ["platform", "open-source"]): "" or "open-source"
// (default) is the self-hosted path, needing no sidecar container since its
// default vector store is a local file inside the plugin's own state
// directory. "platform" instead routes through mem0's cloud API
// (https://api.mem0.ai by default) and needs an account API key — see
// APIKeyEnv/BaseURL below. The two modes are mutually exclusive: LLM*
// (open-source-mode only) and APIKeyEnv/BaseURL (platform-mode only) cannot
// mix — see (*HarnessConfig).validate.
//
// By default open-source mode's own internal LLM/embedder calls (fact
// extraction, embeddings) reuse the profile's primary task model and API
// key — see mem0PluginConfig's doc comment. The LLM* fields below override
// that with a separate model/endpoint instead. Note that mem0's embedder
// needs a real embeddings-capable endpoint (mem0's own OSS default is
// OpenAI's text-embedding-3-small): most chat-completions-only inference
// servers (e.g. an sglang deployment serving a chat model) do not also
// serve embeddings, so profiles whose primary model can't do that will
// likely need LLM* set to an endpoint that can. All three LLM* fields must
// be set together or not at all.
//
// In platform mode, APIKeyEnv names the host environment variable holding
// the mem0.ai account API key (from https://app.mem0.ai/dashboard/api-keys)
// — never placed in profile JSON. It is staged the same way LLMAPIKeyEnv is
// (see mem0PlatformKeyPath's doc comment in pkg/harness/openclaw), read from
// the named host env var and exported into the container under the fixed
// name the plugin's config schema expects (MEM0_API_KEY), referenced from
// rendered config as "${MEM0_API_KEY}". BaseURL optionally overrides the
// plugin's default https://api.mem0.ai, e.g. for a self-hosted mem0
// platform-API-compatible endpoint.
type HarnessMem0Config struct {
	Enabled      bool   `json:"enabled,omitempty"`
	Mode         string `json:"mode,omitempty"`
	LLMBaseURL   string `json:"llm_base_url,omitempty"`
	LLMModel     string `json:"llm_model,omitempty"`
	LLMAPIKeyEnv string `json:"llm_api_key_env,omitempty"`
	APIKeyEnv    string `json:"api_key_env,omitempty"`
	BaseURL      string `json:"base_url,omitempty"`
}

// HarnessHermesConfig is the harness.hermes block. It is valid only with
// harness.type "hermes" and must set at least one field.
type HarnessHermesConfig struct {
	// ExtraBody is an opaque JSON object that Hermes merges into every chat
	// request (custom_providers[].extra_body in config.yaml). It is kept as
	// raw bytes: ARIES validates its shape, its ${NAME} references, and the
	// absence of credential-bearing fields, but never interprets its keys.
	ExtraBody json.RawMessage `json:"extra_body,omitempty"`
}

// HarnessCompactionConfig controls Hermes's context compaction. Hermes
// compacts when the prompt reaches max(context_length * threshold, 64K),
// raised to 75% of windows under 512K. ThresholdTokens is an absolute cap
// that Hermes applies after those floors (compression.threshold_tokens in
// config.yaml), so it is the one knob that sets an exact trigger on a
// large-window model. Enabled false turns compaction off.
type HarnessCompactionConfig struct {
	Enabled         *bool `json:"enabled,omitempty"`
	ThresholdTokens int   `json:"threshold_tokens,omitempty"`
}

// extraBodyPlaceholders are the only ${NAME} references
// harness.hermes.extra_body may carry. Hermes expands every ${NAME} in its
// configuration from the process environment, and the Hermes harness exports
// exactly these two names into the container, so any other reference would
// either stay literal or pull a value, such as the credential, into request
// bodies.
var extraBodyPlaceholders = map[string]bool{"ARIES_RUN_ID": true, "ARIES_TASK_ID": true}

var placeholderPattern = regexp.MustCompile(`\$\{([^}]*)\}`)

// credentialFieldNames are field names that carry a credential in common
// request and gateway schemas, compared after lowercasing and dropping "_",
// "-", and ".". A profile is rejected when harness.hermes.extra_body contains
// one at any depth: the object is written into the retained config.yaml and
// sent with every request, and model keys stay out of JSON profiles. Exact
// names rather than substrings, so max_tokens and threshold_tokens pass.
var credentialFieldNames = map[string]bool{
	"accesstoken": true, "apikey": true, "apitoken": true, "auth": true, "authorization": true,
	"authtoken": true, "bearer": true, "bearertoken": true, "clientsecret": true, "credential": true,
	"credentials": true, "key": true, "passwd": true, "password": true, "privatekey": true,
	"refreshtoken": true, "secret": true, "sessiontoken": true, "token": true, "xapikey": true,
}

// findCredentialField walks a decoded JSON value and returns the dotted path
// of the first credential-bearing field name, or "" when there is none. Keys
// are visited in sorted order so the reported path is stable.
func findCredentialField(value any, path string) string {
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range slices.Sorted(maps.Keys(typed)) {
			child := path + "." + key
			normalized := strings.NewReplacer("_", "", "-", "", ".", "").Replace(strings.ToLower(key))
			if credentialFieldNames[normalized] {
				return child
			}
			if found := findCredentialField(typed[key], child); found != "" {
				return found
			}
		}
	case []any:
		for index, element := range typed {
			if found := findCredentialField(element, path+"["+strconv.Itoa(index)+"]"); found != "" {
				return found
			}
		}
	}
	return ""
}

// HarnessWebSearchConfig is an OpenClaw/Hermes-only concept (see
// (*HarnessConfig).validate), same as Realtime above: it lives on the shared
// struct rather than a type-specific sub-block, gated by an explicit type
// check instead of a nested namespace.
type HarnessWebSearchConfig struct {
	Enabled          bool   `json:"enabled,omitempty"`
	ExtractAPIKeyEnv string `json:"extract_api_key_env,omitempty"`
	// Provider selects the web_search tool's backend. Empty (the default)
	// keeps today's behavior (OpenClaw's built-in SearXNG instance); the other
	// accepted values are "firecrawl" and "tavily", which replace SearXNG
	// entirely as the active tools.web.search.provider (see
	// FirecrawlAPIKeyEnv/TavilyAPIKeyEnv) rather than layering alongside it
	// the way ExtractAPIKeyEnv's Tavily-extract capability does. OpenClaw-only
	// for now (see (*HarnessConfig).validate) — Hermes has its own analogous
	// wiring that would need mirroring separately.
	//
	// Firecrawl and Tavily differ in one important way verified against the
	// actual installed OpenClaw plugin manifests: Firecrawl's manifest
	// declares a webFetchProviders contract, so once it's installed+
	// credentialed OpenClaw auto-selects it as the web_fetch fallback too —
	// confirmed empirically, no config override prevents this. Tavily's
	// manifest declares no such contract (only webSearchProviders), so
	// selecting it here leaves web_fetch on OpenClaw's native fetcher,
	// avoiding Firecrawl's incremental fetch cost.
	Provider string `json:"provider,omitempty"`
	// FirecrawlAPIKeyEnv names the host environment variable holding the
	// Firecrawl API key, required when Provider is "firecrawl". Unlike
	// ExtractAPIKeyEnv (a bonus capability that degrades to disabled if the
	// key is missing at runtime), a missing key here fails the harness start
	// outright: the profile explicitly requested Firecrawl as its only search
	// provider, so silently falling back to SearXNG would mask the
	// misconfiguration for an entire run.
	FirecrawlAPIKeyEnv string `json:"firecrawl_api_key_env,omitempty"`
	// TavilyAPIKeyEnv names the host environment variable holding the Tavily
	// API key, required when Provider is "tavily". Same fail-fast rationale
	// as FirecrawlAPIKeyEnv. Deliberately a separate field from
	// ExtractAPIKeyEnv even though both ultimately credential the same
	// "tavily" plugin: collapsing them would force the tavily_extract tool on
	// whenever Tavily search is requested, which isn't what a profile wanting
	// search-only (zero fetch-tool change) is asking for.
	TavilyAPIKeyEnv string `json:"tavily_api_key_env,omitempty"`
}

// HarnessSubagentsConfig is an OpenClaw/Hermes-only concept (see
// (*HarnessConfig).validate), same as WebSearch above. Under OpenClaw,
// disabling denies its sessions_spawn/sessions_yield tools, because ARIES's
// harness protocol has no continuation mechanism to relay async sub-agent
// completions back to a single-turn request. Under Hermes, disabling removes
// its delegate_task tool via disabled_toolsets; Hermes's delegation blocks
// and returns within the same tool call, so it has no equivalent protocol
// hazard — disabling it there is purely a cost/determinism control.
// Enabled is a pointer so "unset" (defaults to enabled, matching both
// harnesses' own defaults) is distinguishable from an explicit "false" (e.g.
// Deep Research Bench, which opts out since subagent spawning isn't useful
// for its single-report task shape and adds uncontrolled cost).
//
// MaxConcurrent bounds how many subagents may run at once without disabling
// subagents outright: OpenClaw's agents.defaults.subagents.maxConcurrent, or
// Hermes's delegation.max_concurrent_children. Zero leaves each harness's own
// built-in default in place.
type HarnessSubagentsConfig struct {
	Enabled       *bool `json:"enabled,omitempty"`
	MaxConcurrent int   `json:"max_concurrent,omitempty"`
}

type HarnessRealtimeConfig struct {
	AgentQuestionTemplate string            `json:"agent_question_template,omitempty"`
	TTS                   RealtimeTTSConfig `json:"tts,omitempty"`
	ChunkDurationText     string            `json:"chunk_duration,omitempty"`
	ListenDurationText    string            `json:"listen_duration,omitempty"`
	QuietDurationText     string            `json:"quiet_duration,omitempty"`
	AgentWaitDurationText string            `json:"agent_wait_duration,omitempty"`
	ToolCallTimeoutText   string            `json:"tool_call_timeout,omitempty"`
	TrailingSilenceMillis int               `json:"trailing_silence_ms,omitempty"`
	Provider              string            `json:"provider,omitempty"`
	Model                 string            `json:"model,omitempty"`
	Voice                 string            `json:"voice,omitempty"`
	ReasoningEffort       string            `json:"reasoning_effort,omitempty"`
	IncludeEvents         bool              `json:"include_events,omitempty"`
	ChunkDuration         time.Duration     `json:"-"`
	ListenDuration        time.Duration     `json:"-"`
	QuietDuration         time.Duration     `json:"-"`
	AgentWaitDuration     time.Duration     `json:"-"`
	ToolCallTimeout       time.Duration     `json:"-"`
}

type RealtimeTTSConfig struct {
	Provider     string        `json:"provider,omitempty"`
	BaseURL      string        `json:"base_url,omitempty"`
	APIKeyEnv    string        `json:"api_key_env,omitempty"`
	Model        string        `json:"model,omitempty"`
	Voice        string        `json:"voice,omitempty"`
	Instructions string        `json:"instructions,omitempty"`
	Speed        *float64      `json:"speed,omitempty"`
	TimeoutText  string        `json:"timeout,omitempty"`
	Timeout      time.Duration `json:"-"`
}

type HarnessVoiceTranscribeConfig struct {
	HarnessRealtimeConfig
	STT VoiceSTTConfig `json:"stt,omitempty"`
}

type VoiceSTTConfig struct {
	Provider    string        `json:"provider,omitempty"`
	Model       string        `json:"model,omitempty"`
	Language    string        `json:"language,omitempty"`
	TimeoutText string        `json:"timeout,omitempty"`
	Timeout     time.Duration `json:"-"`
}

type SandboxConfig struct {
	Type string `json:"type"`
}

type BridgeConfig struct {
	Type string `json:"type"`
	// RetainRawLog keeps the bridge's byte-level ssh_raw.log. It defaults to
	// false because that log is the largest artifact a run writes and captures
	// raw wire commands, request payloads, and binary stdin; profiles that need
	// that forensic record must opt in by setting it true.
	RetainRawLog *bool `json:"retain_raw_log,omitempty"`
}

// RetainBridgeRawLog reports whether ssh_raw.log should be written, defaulting
// to false when a profile does not mention it.
func (c BridgeConfig) RetainBridgeRawLog() bool {
	return c.RetainRawLog != nil && *c.RetainRawLog
}

func (c Config) CoreModel() core.ModelConfig {
	return core.ModelConfig{
		Provider: c.Runtime.Backend, BaseURL: c.Model.BaseURL, Model: c.Model.ID, APIKeyEnv: c.Model.APIKeyEnv,
		ContextLength: c.Model.ContextLength, MaxTokens: c.Model.MaxTokens, Temperature: c.Model.Temperature,
		MaxOutputTokens: c.Model.MaxOutputTokens, ContextWindowTokens: c.Model.ContextWindowTokens,
		CompactionTimeoutMs: c.Model.CompactionTimeoutMs,
	}
}

// Versions contains the upstream version selections shared by profiles.
type Versions struct {
	TerminalBench2    TerminalBench2Versions    `json:"terminalbench2"`
	DeepResearchBench DeepResearchBenchVersions `json:"deepresearchbench"`
	SWEAtlas          SWEAtlasVersions          `json:"sweatlasqa"`
	SWEbenchPro       SWEbenchProVersions       `json:"swebenchpro"`
	OpenClaw          OpenClawVersions          `json:"openclaw"`
	Hermes            HermesVersions            `json:"hermes"`
	AMEMQdrant        AMEMQdrantVersions        `json:"amem_qdrant"`
}

type TerminalBench2Versions struct {
	RepositoryURL string `json:"repository_url"`
	Revision      string `json:"revision"`
}

type DeepResearchBenchVersions struct {
	RepositoryURL string `json:"repository_url"`
	Revision      string `json:"revision"`
}

type SWEAtlasVersions struct {
	RepositoryURL string `json:"repository_url"`
	Revision      string `json:"revision"`
}

type SWEbenchProVersions struct {
	DatasetRepositoryURL   string `json:"dataset_repository_url"`
	DatasetRevision        string `json:"dataset_revision"`
	EvaluatorRepositoryURL string `json:"evaluator_repository_url"`
	EvaluatorRevision      string `json:"evaluator_revision"`
}

type OpenClawVersions struct {
	Image string `json:"image"`
}

type HermesVersions struct {
	Image string `json:"image"`
}

// AMEMQdrantVersions pins the Qdrant image the OpenClaw harness starts as a
// run-scoped sidecar when harness.amem.enabled — required only in that case
// (see (Versions).validate and (*HarnessConfig).validate), since amem's
// plugin hardcodes its Qdrant client to reach it via a fixed alias.
type AMEMQdrantVersions struct {
	Image string `json:"image"`
}

// Load reads and strictly validates one JSON experiment file.
func Load(path string) (Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return Config{}, fmt.Errorf("open experiment config: %w", err)
	}
	defer f.Close()

	cfg, err := Decode(f)
	if err != nil {
		return Config{}, fmt.Errorf("decode experiment config %q: %w", path, err)
	}
	versionsPath := cfg.VersionsFile
	if !filepath.IsAbs(versionsPath) {
		versionsPath = filepath.Join(filepath.Dir(path), versionsPath)
	}
	versions, err := LoadVersions(versionsPath)
	if err != nil {
		return Config{}, fmt.Errorf("load version pins: %w", err)
	}
	cfg.Versions = versions
	if cfg.OverridesFile != "" {
		overridesPath := cfg.OverridesFile
		if !filepath.IsAbs(overridesPath) {
			overridesPath = filepath.Join(filepath.Dir(path), overridesPath)
		}
		overrides, err := LoadRuntimeOverrides(overridesPath)
		if err != nil {
			return Config{}, fmt.Errorf("load runtime overrides: %w", err)
		}
		cfg.Overrides = overrides
	}
	if cfg.Runtime.Config.File != "" {
		resolved := cfg.Runtime.Config.File
		if !filepath.IsAbs(resolved) {
			resolved = filepath.Join(filepath.Dir(path), resolved)
		}
		resolved, err = filepath.Abs(resolved)
		if err != nil {
			return Config{}, fmt.Errorf("resolve runtime config file: %w", err)
		}
		cfg.Runtime.Config.ResolvedFile = resolved
	}
	return cfg, nil
}

// LoadRuntimeOverrides reads one dedicated strict runtime override document.
func LoadRuntimeOverrides(path string) (RuntimeOverrides, error) {
	f, err := os.Open(path)
	if err != nil {
		return RuntimeOverrides{}, fmt.Errorf("open runtime overrides %q: %w", path, err)
	}
	defer f.Close()
	var overrides RuntimeOverrides
	if err := decodeStrictJSON(f, &overrides, "runtime overrides"); err != nil {
		return RuntimeOverrides{}, fmt.Errorf("decode runtime overrides %q: %w", path, err)
	}
	if err := overrides.validate(); err != nil {
		return RuntimeOverrides{}, fmt.Errorf("validate runtime overrides %q: %w", path, err)
	}
	return overrides, nil
}

func (o *RuntimeOverrides) validate() error {
	if err := validateResources("harness_resources", o.HarnessResources); err != nil {
		return err
	}
	if err := validateResources("agent_sandbox_resources", o.AgentSandboxResources); err != nil {
		return err
	}
	var err error
	if o.AgentTimeout, err = secondsDuration("agent_timeout_seconds", o.AgentTimeoutSeconds); err != nil {
		return err
	}
	if o.VerifierTimeoutFloor, err = secondsDuration("verifier_timeout_floor_seconds", o.VerifierTimeoutFloorSeconds); err != nil {
		return err
	}
	return nil
}

// secondsDuration converts an optional positive seconds value into a duration.
func secondsDuration(name string, seconds *float64) (*time.Duration, error) {
	if seconds == nil {
		return nil, nil
	}
	scaled := *seconds * float64(time.Second)
	if *seconds <= 0 || math.IsNaN(*seconds) || math.IsInf(*seconds, 0) || scaled >= math.Exp2(63) {
		return nil, fmt.Errorf("%s must be finite, positive, and convert to nanoseconds below 2^63", name)
	}
	duration := time.Duration(scaled)
	if duration <= 0 {
		return nil, fmt.Errorf("%s must be finite, positive, and convert to a positive duration below 2^63 nanoseconds", name)
	}
	return &duration, nil
}

func validateResources(name string, resources ResourceOverrides) error {
	if resources.CPU != nil {
		scaled := *resources.CPU * 1e9
		if *resources.CPU <= 0 || math.IsNaN(*resources.CPU) || math.IsInf(*resources.CPU, 0) || scaled >= math.Exp2(63) {
			return fmt.Errorf("%s.cpu must be finite, positive, and convert to NanoCPUs below 2^63", name)
		}
	}
	if resources.MemoryMB != nil && (*resources.MemoryMB <= 0 || int64(*resources.MemoryMB) > math.MaxInt64>>20) {
		return fmt.Errorf("%s.memory_mb must be positive and no greater than %d", name, int64(math.MaxInt64)>>20)
	}
	return nil
}

// Decode rejects unknown fields and trailing JSON values.
func Decode(r io.Reader) (Config, error) {
	cfg := Config{Execution: ExecutionConfig{Concurrency: 1}}
	if err := decodeStrictJSON(r, &cfg, "experiment config"); err != nil {
		return Config{}, err
	}

	if cfg.OutputDir == "" {
		cfg.OutputDir = defaultOutputDir
	}
	if err := cfg.validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// LoadVersions reads one strict version pin catalog.
func LoadVersions(path string) (Versions, error) {
	f, err := os.Open(path)
	if err != nil {
		return Versions{}, fmt.Errorf("open version pins %q: %w", path, err)
	}
	defer f.Close()
	versions, err := DecodeVersions(f)
	if err != nil {
		return Versions{}, fmt.Errorf("decode version pins %q: %w", path, err)
	}
	return versions, nil
}

// DecodeVersions rejects unknown fields and invalid version selections.
func DecodeVersions(r io.Reader) (Versions, error) {
	var versions Versions
	if err := decodeStrictJSON(r, &versions, "version pins"); err != nil {
		return Versions{}, err
	}
	if err := versions.validate(); err != nil {
		return Versions{}, err
	}
	return versions, nil
}

func decodeStrictJSON(r io.Reader, destination any, name string) error {
	decoder := json.NewDecoder(r)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("%s contains multiple JSON values", name)
		}
		return fmt.Errorf("read trailing JSON: %w", err)
	}
	return nil
}

func (c *Config) validate() error {
	if c.Execution.Concurrency <= 0 {
		return errors.New("execution.concurrency must be positive")
	}
	if c.Execution.LoopDuration != "" {
		loop, err := time.ParseDuration(c.Execution.LoopDuration)
		if err != nil || loop <= 0 {
			return errors.New("execution.loop_duration must be a positive Go duration")
		}
		c.Execution.Loop = loop
	}
	if (c.Execution.ArrivalsFile != "") != (c.Execution.ArrivalRatePerMin != 0) {
		return errors.New("execution.arrivals_file and execution.arrival_rate_per_min must be set together")
	}
	if c.Execution.ArrivalsFile != "" {
		if !(c.Execution.ArrivalRatePerMin > 0) || math.IsInf(c.Execution.ArrivalRatePerMin, 0) {
			return errors.New("execution.arrival_rate_per_min must be finite and positive")
		}
		if c.Execution.LoopDuration != "" {
			return errors.New("execution.arrivals_file cannot be combined with execution.loop_duration")
		}
	}
	checks := []struct {
		name  string
		value string
	}{
		{"name", c.Name},
		{"versions_file", c.VersionsFile},
		{"benchmark.type", c.Benchmark.Type},
		{"benchmark.root", c.Benchmark.Root},
		{"harness.type", c.Harness.Type},
		{"sandbox.type", c.Sandbox.Type},
		{"bridge.type", c.Bridge.Type},
		{"runtime.backend", c.Runtime.Backend},
		{"runtime.mode", c.Runtime.Mode},
		{"model.base_url", c.Model.BaseURL},
		{"model.id", c.Model.ID},
		{"model.api_key_env", c.Model.APIKeyEnv},
		{"output_dir", c.OutputDir},
	}
	for _, check := range checks {
		if strings.TrimSpace(check.value) == "" {
			return fmt.Errorf("%s is required", check.name)
		}
	}
	if c.Runtime.Backend != "deepseek" && c.Runtime.Backend != "sglang" && c.Runtime.Backend != "openai" {
		return errors.New("runtime.backend must be deepseek, sglang, or openai")
	}
	if c.Harness.Mode == "" {
		c.Harness.Mode = "agent"
	}
	if memory := c.Harness.Memory; memory.Provider != "" && memory.FrozenState == "" && c.Execution.Concurrency != 1 {
		return errors.New("harness.memory requires execution.concurrency 1 unless frozen_state is set: its run-scoped state is handed from task to task")
	}
	for key := range c.Harness.Memory.Env {
		if key == c.Model.APIKeyEnv || c.Benchmark.Judge != nil && key == c.Benchmark.Judge.APIKeyEnv {
			return fmt.Errorf("harness.memory.env must not set the credential variable %q", key)
		}
	}
	if err := c.Harness.validate(); err != nil {
		return err
	}
	if err := c.Model.validateGeneration(c.Harness.Type); err != nil {
		return err
	}
	if c.Harness.Compaction != nil && c.Model.ContextLength > 0 && c.Harness.Compaction.ThresholdTokens >= c.Model.ContextLength {
		return errors.New("harness.compaction.threshold_tokens must be smaller than model.context_length")
	}
	// Hermes merges a custom_providers extra_body only for its "custom"
	// provider, which is how the OpenAI-compatible backends render; under
	// DeepSeek the block would be silently ignored.
	if c.Harness.Hermes != nil && c.Runtime.Backend != "sglang" && c.Runtime.Backend != "openai" {
		return errors.New("harness.hermes.extra_body requires runtime.backend sglang or openai")
	}
	if c.Model.Temperature != nil {
		if c.Runtime.Backend != "sglang" && c.Runtime.Backend != "openai" {
			return errors.New("model.temperature requires runtime.backend sglang or openai for Hermes")
		}
		if c.Harness.Hermes != nil {
			var object map[string]json.RawMessage
			if err := json.Unmarshal(c.Harness.Hermes.ExtraBody, &object); err != nil {
				return fmt.Errorf("harness.hermes.extra_body: %w", err)
			}
			if _, exists := object["temperature"]; exists {
				return errors.New("model.temperature conflicts with harness.hermes.extra_body.temperature")
			}
		}
	}
	if err := c.Runtime.validate(); err != nil {
		return err
	}
	if err := validateExperimentName(c.Name); err != nil {
		return err
	}
	if len(c.Benchmark.Tasks) == 0 {
		return errors.New("benchmark.tasks must contain at least one task ID")
	}
	for i, task := range c.Benchmark.Tasks {
		if strings.TrimSpace(task) == "" {
			return fmt.Errorf("benchmark.tasks[%d] must not be empty", i)
		}
	}

	if c.Runtime.Backend == "sglang" || c.Runtime.Backend == "openai" {
		normalized, err := normalizeV1BaseURL(c.Model.BaseURL)
		if err != nil {
			return fmt.Errorf("model.base_url for %s: %w", c.Runtime.Backend, err)
		}
		c.Model.BaseURL = normalized
	} else if err := validateHTTPBaseURL("model.base_url", c.Model.BaseURL); err != nil {
		return err
	}
	if !validEnvName(c.Model.APIKeyEnv) {
		return errors.New("model.api_key_env must be an environment variable name")
	}
	if c.Model.MaxOutputTokens < 0 {
		return errors.New("model.max_output_tokens must not be negative")
	}
	if c.Model.ContextWindowTokens < 0 {
		return errors.New("model.context_window_tokens must not be negative")
	}
	if c.Model.CompactionTimeoutMs < 0 {
		return errors.New("model.compaction_timeout_ms must not be negative")
	}
	if err := c.validateBenchmarkType(); err != nil {
		return err
	}
	return nil
}

// validateHTTPBaseURL rejects anything but a credential-free, query-free,
// fragment-free absolute HTTP(S) URL.
func validateHTTPBaseURL(name, value string) error {
	parsed, err := url.Parse(value)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return fmt.Errorf("%s must be an absolute HTTP(S) URL", name)
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return fmt.Errorf("%s must not contain credentials, a query, or a fragment", name)
	}
	return nil
}

// validateBenchmarkType enforces which optional profile blocks a given
// benchmark type requires versus forbids. Fields that don't apply to the
// active benchmark type are rejected outright rather than silently ignored.
// An unrecognized benchmark type is intentionally not rejected here: that is
// the wiring layer's ValidateComponents responsibility, so unknown types
// still decode into a Config for that check to inspect and reject.
func (c *Config) validateBenchmarkType() error {
	switch c.Benchmark.Type {
	case "deepresearchbench":
		if c.Benchmark.Environment == nil || strings.TrimSpace(c.Benchmark.Environment.Image) == "" {
			return errors.New("benchmark.environment.image is required for deepresearchbench")
		}
		if c.Benchmark.PlanOnly && c.Benchmark.StructuredSubtasks != nil {
			return errors.New("benchmark.plan_only and benchmark.structured_subtasks are mutually exclusive")
		}
		if c.Benchmark.PlanOnly {
			if judge := c.Benchmark.Judge; judge == nil || judge.Enabled == nil || *judge.Enabled {
				return errors.New("benchmark.plan_only requires judge.enabled to be explicitly false")
			}
			if strings.TrimSpace(c.Benchmark.PlansetDir) == "" {
				return errors.New("benchmark.planset_dir is required when benchmark.plan_only is true")
			}
		}
		if structured := c.Benchmark.StructuredSubtasks; structured != nil {
			if strings.TrimSpace(structured.PlansetDir) == "" {
				return errors.New("benchmark.structured_subtasks.planset_dir is required")
			}
			switch structured.Order {
			case "sequential", "shuffled", "adversarial":
			default:
				return errors.New(`benchmark.structured_subtasks.order must be "sequential", "shuffled", or "adversarial"`)
			}
			if structured.Order == "shuffled" && structured.Seed == 0 {
				return errors.New("benchmark.structured_subtasks.seed is required (non-zero) when order is shuffled")
			}
		}
		if judge := c.Benchmark.Judge; judge != nil {
			if judge.Enabled != nil && !*judge.Enabled {
				if judge.Provider != "" || judge.BaseURL != "" || judge.ID != "" || judge.APIKeyEnv != "" {
					return errors.New("judge model fields must not be set when judge.enabled is false")
				}
			} else {
				if strings.TrimSpace(judge.Provider) == "" {
					return errors.New("judge.provider is required")
				}
				if err := validateHTTPBaseURL("judge.base_url", judge.BaseURL); err != nil {
					return err
				}
				if strings.TrimSpace(judge.ID) == "" {
					return errors.New("judge.model is required")
				}
				if !validEnvName(judge.APIKeyEnv) {
					return errors.New("judge.api_key_env must be an environment variable name")
				}
			}
		}
		if fact := c.Benchmark.Fact; fact != nil {
			factModelSpecified := strings.TrimSpace(fact.Provider) != "" || strings.TrimSpace(fact.BaseURL) != "" ||
				strings.TrimSpace(fact.ID) != "" || strings.TrimSpace(fact.APIKeyEnv) != ""
			if factModelSpecified {
				if strings.TrimSpace(fact.Provider) == "" {
					return errors.New("fact.provider is required")
				}
				if err := validateHTTPBaseURL("fact.base_url", fact.BaseURL); err != nil {
					return err
				}
				if strings.TrimSpace(fact.ID) == "" {
					return errors.New("fact.model is required")
				}
				if !validEnvName(fact.APIKeyEnv) {
					return errors.New("fact.api_key_env must be an environment variable name")
				}
			}
			if !validEnvName(fact.JinaAPIKeyEnv) {
				return errors.New("fact.jina_api_key_env must be an environment variable name")
			}
		}
		return nil
	case "terminalbench2":
		if c.Benchmark.Environment != nil {
			return errors.New("benchmark.environment must not be set for terminalbench2")
		}
		if c.Benchmark.Judge != nil {
			return errors.New("judge must not be set for terminalbench2")
		}
		if c.Benchmark.Fact != nil {
			return errors.New("fact must not be set for terminalbench2")
		}
		if c.Benchmark.PlanOnly || c.Benchmark.PlansetDir != "" {
			return errors.New("benchmark.plan_only and benchmark.planset_dir must not be set for terminalbench2")
		}
		if c.Benchmark.StructuredSubtasks != nil {
			return errors.New("benchmark.structured_subtasks must not be set for terminalbench2")
		}
		return nil
	case "sweatlasqa":
		judge := c.Benchmark.Judge
		if judge == nil {
			return errors.New("benchmark.judge is required for sweatlasqa")
		}
		if judge.Enabled != nil && !*judge.Enabled {
			if judge.Provider != "" || judge.BaseURL != "" || judge.ID != "" || judge.APIKeyEnv != "" {
				return errors.New("judge fields must not be set when judge.enabled is false for sweatlasqa")
			}
		} else {
			if strings.TrimSpace(judge.Provider) == "" {
				return errors.New("judge.provider is required for sweatlasqa")
			}
			if err := validateHTTPBaseURL("judge.base_url", judge.BaseURL); err != nil {
				return err
			}
			if strings.TrimSpace(judge.ID) == "" {
				return errors.New("judge.model is required for sweatlasqa")
			}
			if !validEnvName(judge.APIKeyEnv) {
				return errors.New("judge.api_key_env must be an environment variable name")
			}
		}
		if c.Benchmark.Environment != nil {
			return errors.New("benchmark.environment must not be set for sweatlasqa")
		}
		if c.Benchmark.Fact != nil {
			return errors.New("fact must not be set for sweatlasqa")
		}
		return nil
	case "swebenchpro":
		if c.Benchmark.Environment != nil {
			return errors.New("benchmark.environment must not be set for swebenchpro")
		}
		if c.Benchmark.Judge != nil {
			return errors.New("judge must not be set for swebenchpro")
		}
		if c.Benchmark.Fact != nil {
			return errors.New("fact must not be set for swebenchpro")
		}
		if c.Benchmark.PlanOnly || c.Benchmark.PlansetDir != "" {
			return errors.New("benchmark.plan_only and benchmark.planset_dir must not be set for swebenchpro")
		}
		if c.Benchmark.StructuredSubtasks != nil {
			return errors.New("benchmark.structured_subtasks must not be set for swebenchpro")
		}
		return nil
	default:
		return nil
	}
}

// validateHermesBlocks rejects the Hermes-only blocks under another harness
// and checks their values. Empty blocks are rejected rather than ignored so a
// profile never carries a block that changes nothing.
func (h *HarnessConfig) validateHermesBlocks() error {
	if h.Compaction != nil {
		if h.Type != "hermes" {
			return errors.New("harness.compaction requires Hermes")
		}
		if h.Compaction.Enabled == nil && h.Compaction.ThresholdTokens == 0 {
			return errors.New("harness.compaction must set enabled or threshold_tokens")
		}
		if h.Compaction.ThresholdTokens < 0 {
			return errors.New("harness.compaction.threshold_tokens must be positive")
		}
	}
	if h.Hermes != nil {
		if h.Type != "hermes" {
			return errors.New("harness.hermes requires Hermes")
		}
		if len(h.Hermes.ExtraBody) == 0 {
			return errors.New("harness.hermes must set extra_body")
		}
		var object map[string]any
		if err := json.Unmarshal(h.Hermes.ExtraBody, &object); err != nil || len(object) == 0 {
			return errors.New("harness.hermes.extra_body must be a non-empty JSON object")
		}
		if field := findCredentialField(object, "harness.hermes.extra_body"); field != "" {
			return fmt.Errorf("%s is named like a credential; model keys stay out of JSON profiles", field)
		}
		for _, match := range placeholderPattern.FindAllSubmatch(h.Hermes.ExtraBody, -1) {
			if !extraBodyPlaceholders[string(match[1])] {
				return fmt.Errorf("harness.hermes.extra_body may reference only ${ARIES_RUN_ID} and ${ARIES_TASK_ID}, not ${%s}", match[1])
			}
		}
	}
	return nil
}

func (h *HarnessConfig) validate() error {
	if h.Mode == "" {
		h.Mode = "agent"
	}
	if err := h.validateHermesBlocks(); err != nil {
		return err
	}
	if h.WebSearch.Enabled && h.Type != "openclaw" && h.Type != "hermes" {
		return errors.New("harness.web_search requires OpenClaw or Hermes")
	}
	if h.WebSearch.ExtractAPIKeyEnv != "" {
		if h.Type != "openclaw" && h.Type != "hermes" {
			return errors.New("harness.web_search.extract_api_key_env requires OpenClaw or Hermes")
		}
		if !h.WebSearch.Enabled {
			return errors.New("harness.web_search.extract_api_key_env requires harness.web_search.enabled")
		}
		if !validEnvName(h.WebSearch.ExtractAPIKeyEnv) {
			return errors.New("harness.web_search.extract_api_key_env must be an environment variable name")
		}
	}
	if h.WebSearch.Provider != "" {
		if h.WebSearch.Provider != "firecrawl" && h.WebSearch.Provider != "tavily" {
			return errors.New("harness.web_search.provider must be \"firecrawl\" or \"tavily\" if set")
		}
		if h.Type != "openclaw" {
			return errors.New("harness.web_search.provider requires OpenClaw")
		}
		if !h.WebSearch.Enabled {
			return errors.New("harness.web_search.provider requires harness.web_search.enabled")
		}
		if h.WebSearch.Provider == "firecrawl" {
			if !validEnvName(h.WebSearch.FirecrawlAPIKeyEnv) {
				return errors.New("harness.web_search.provider \"firecrawl\" requires harness.web_search.firecrawl_api_key_env")
			}
		} else {
			if !validEnvName(h.WebSearch.TavilyAPIKeyEnv) {
				return errors.New("harness.web_search.provider \"tavily\" requires harness.web_search.tavily_api_key_env")
			}
		}
	} else if h.WebSearch.FirecrawlAPIKeyEnv != "" {
		return errors.New("harness.web_search.firecrawl_api_key_env requires harness.web_search.provider \"firecrawl\"")
	} else if h.WebSearch.TavilyAPIKeyEnv != "" {
		return errors.New("harness.web_search.tavily_api_key_env requires harness.web_search.provider \"tavily\"")
	}
	if h.Subagents.Enabled != nil && h.Type != "openclaw" && h.Type != "hermes" {
		return errors.New("harness.subagents requires OpenClaw or Hermes")
	}
	if h.Subagents.MaxConcurrent != 0 {
		if h.Type != "openclaw" && h.Type != "hermes" {
			return errors.New("harness.subagents.max_concurrent requires OpenClaw or Hermes")
		}
		if h.Subagents.MaxConcurrent < 0 {
			return errors.New("harness.subagents.max_concurrent must be positive")
		}
	}
	if h.Subagents.Enabled == nil && (h.Type == "openclaw" || h.Type == "hermes") {
		enabled := true
		h.Subagents.Enabled = &enabled
	}
	if h.AMEM.Enabled && h.Type != "openclaw" {
		return errors.New("harness.amem requires OpenClaw")
	}
	if h.AMEM.Scope != "" && h.AMEM.Scope != "task" && h.AMEM.Scope != "repo" && h.AMEM.Scope != "global" {
		return errors.New(`harness.amem.scope must be "task", "repo", or "global"`)
	}
	if h.AMEM.Scope != "" && !h.AMEM.Enabled {
		return errors.New("harness.amem.scope requires harness.amem.enabled")
	}
	if h.AMEM.RerankModel != "" && !h.AMEM.Enabled {
		return errors.New("harness.amem.rerank_model requires harness.amem.enabled")
	}
	if h.AMEM.RerankThreshold != nil && !h.AMEM.Enabled {
		return errors.New("harness.amem.rerank_threshold requires harness.amem.enabled")
	}
	if h.AMEM.RerankDevice != "" && !h.AMEM.Enabled {
		return errors.New("harness.amem.rerank_device requires harness.amem.enabled")
	}
	if h.AMEM.DisableTaskTraceFallback && !h.AMEM.Enabled {
		return errors.New("harness.amem.disable_task_trace_fallback requires harness.amem.enabled")
	}
	if h.AMEM.DisableGoalEmbedding && !h.AMEM.Enabled {
		return errors.New("harness.amem.disable_goal_embedding requires harness.amem.enabled")
	}
	if h.AMEM.DisableRerank && !h.AMEM.Enabled {
		return errors.New("harness.amem.disable_rerank requires harness.amem.enabled")
	}
	if h.AMEM.EpisodicBootstrap && !h.AMEM.Enabled {
		return errors.New("harness.amem.episodic_bootstrap requires harness.amem.enabled")
	}
	if h.AMEM.LLMBaseURL != "" || h.AMEM.LLMModel != "" || h.AMEM.LLMAPIKeyEnv != "" {
		if !h.AMEM.Enabled {
			return errors.New("harness.amem.llm_base_url/llm_model/llm_api_key_env require harness.amem.enabled")
		}
		if h.AMEM.LLMBaseURL == "" || h.AMEM.LLMModel == "" || h.AMEM.LLMAPIKeyEnv == "" {
			return errors.New("harness.amem.llm_base_url, llm_model, and llm_api_key_env must be set together")
		}
		if err := validateHTTPBaseURL("harness.amem.llm_base_url", h.AMEM.LLMBaseURL); err != nil {
			return err
		}
		if !validEnvName(h.AMEM.LLMAPIKeyEnv) {
			return errors.New("harness.amem.llm_api_key_env must be an environment variable name")
		}
	}
	if h.LosslessClaw.Enabled && h.Type != "openclaw" {
		return errors.New("harness.lossless_claw requires OpenClaw")
	}
	if h.AMEM.Enabled && h.LosslessClaw.Enabled {
		return errors.New("harness.amem and harness.lossless_claw are mutually exclusive")
	}
	if h.LosslessClaw.LLMBaseURL != "" || h.LosslessClaw.LLMModel != "" || h.LosslessClaw.LLMAPIKeyEnv != "" {
		if !h.LosslessClaw.Enabled {
			return errors.New("harness.lossless_claw.llm_base_url/llm_model/llm_api_key_env require harness.lossless_claw.enabled")
		}
		if h.LosslessClaw.LLMBaseURL == "" || h.LosslessClaw.LLMModel == "" || h.LosslessClaw.LLMAPIKeyEnv == "" {
			return errors.New("harness.lossless_claw.llm_base_url, llm_model, and llm_api_key_env must be set together")
		}
		if err := validateHTTPBaseURL("harness.lossless_claw.llm_base_url", h.LosslessClaw.LLMBaseURL); err != nil {
			return err
		}
		if !validEnvName(h.LosslessClaw.LLMAPIKeyEnv) {
			return errors.New("harness.lossless_claw.llm_api_key_env must be an environment variable name")
		}
	}
	if h.Mem0.Enabled && h.Type != "openclaw" {
		return errors.New("harness.mem0 requires OpenClaw")
	}
	if err := h.Memory.validate(h.Type); err != nil {
		return err
	}
	if h.AMEM.Enabled && h.Mem0.Enabled {
		return errors.New("harness.amem and harness.mem0 are mutually exclusive")
	}
	if h.Mem0.Mode != "" && h.Mem0.Mode != "open-source" && h.Mem0.Mode != "platform" {
		return errors.New(`harness.mem0.mode must be "open-source" or "platform"`)
	}
	if h.Mem0.Mode != "" && !h.Mem0.Enabled {
		return errors.New("harness.mem0.mode requires harness.mem0.enabled")
	}
	if h.Mem0.Mode == "platform" {
		if h.Mem0.LLMBaseURL != "" || h.Mem0.LLMModel != "" || h.Mem0.LLMAPIKeyEnv != "" {
			return errors.New(`harness.mem0.llm_base_url/llm_model/llm_api_key_env are open-source-mode only, not valid with mode "platform"`)
		}
		if h.Mem0.APIKeyEnv == "" {
			return errors.New(`harness.mem0.mode "platform" requires harness.mem0.api_key_env`)
		}
		if !validEnvName(h.Mem0.APIKeyEnv) {
			return errors.New("harness.mem0.api_key_env must be an environment variable name")
		}
		if h.Mem0.BaseURL != "" {
			if err := validateHTTPBaseURL("harness.mem0.base_url", h.Mem0.BaseURL); err != nil {
				return err
			}
		}
	} else if h.Mem0.APIKeyEnv != "" || h.Mem0.BaseURL != "" {
		return errors.New(`harness.mem0.api_key_env/base_url require harness.mem0.mode "platform"`)
	}
	if h.Mem0.LLMBaseURL != "" || h.Mem0.LLMModel != "" || h.Mem0.LLMAPIKeyEnv != "" {
		if !h.Mem0.Enabled {
			return errors.New("harness.mem0.llm_base_url/llm_model/llm_api_key_env require harness.mem0.enabled")
		}
		if h.Mem0.LLMBaseURL == "" || h.Mem0.LLMModel == "" || h.Mem0.LLMAPIKeyEnv == "" {
			return errors.New("harness.mem0.llm_base_url, llm_model, and llm_api_key_env must be set together")
		}
		if err := validateHTTPBaseURL("harness.mem0.llm_base_url", h.Mem0.LLMBaseURL); err != nil {
			return err
		}
		if !validEnvName(h.Mem0.LLMAPIKeyEnv) {
			return errors.New("harness.mem0.llm_api_key_env must be an environment variable name")
		}
	}
	if len(h.MCPServers) > 0 {
		if h.Type != "openclaw" && h.Type != "hermes" {
			return errors.New("harness.mcp_servers requires OpenClaw or Hermes")
		}
		seenMCPServers := make(map[string]bool, len(h.MCPServers))
		for _, server := range h.MCPServers {
			if seenMCPServers[server.Name] {
				return fmt.Errorf("duplicate MCP server name %q", server.Name)
			}
			seenMCPServers[server.Name] = true
			if err := core.ValidateMCPServer(server); err != nil {
				return fmt.Errorf("harness.mcp_servers: %w", err)
			}
		}
	}
	switch h.Mode {
	case "agent":
		if h.Realtime != (HarnessRealtimeConfig{}) {
			return errors.New("harness.realtime must be empty unless harness.mode is realtime")
		}
		if h.VoiceTranscribe != (HarnessVoiceTranscribeConfig{}) {
			return errors.New("harness.voice_transcribe must be empty unless harness.mode is voice-transcribe")
		}
		return nil
	case "realtime":
		if h.Type != "openclaw" {
			return errors.New("harness.mode realtime requires OpenClaw")
		}
		if h.VoiceTranscribe != (HarnessVoiceTranscribeConfig{}) {
			return errors.New("harness.voice_transcribe must be empty unless harness.mode is voice-transcribe")
		}
		return h.Realtime.validate()
	case "voice-transcribe":
		switch h.Type {
		case "openclaw":
			if h.Realtime != (HarnessRealtimeConfig{}) {
				return errors.New("harness.realtime must be empty unless harness.mode is realtime")
			}
			return h.VoiceTranscribe.validateOpenClaw()
		case "hermes":
			if h.Realtime != (HarnessRealtimeConfig{}) {
				return errors.New("harness.realtime must be empty unless harness.mode is realtime")
			}
			if err := h.VoiceTranscribe.TTS.validateNamed("harness.voice_transcribe.tts"); err != nil {
				return err
			}
			if err := h.VoiceTranscribe.STT.validate(); err != nil {
				return err
			}
			return nil
		default:
			return errors.New("harness.mode voice-transcribe requires OpenClaw or Hermes")
		}
	default:
		return errors.New("harness.mode must be agent, realtime, or voice-transcribe")
	}
}

func (realtime *HarnessRealtimeConfig) validate() error {
	return realtime.validateNamed("harness.realtime")
}

func (realtime *HarnessRealtimeConfig) validateNamed(name string) error {
	if err := realtime.TTS.validateNamed(name + ".tts"); err != nil {
		return err
	}
	if realtime.TrailingSilenceMillis < 0 {
		return fmt.Errorf("%s.trailing_silence_ms must not be negative", name)
	}
	var err error
	if realtime.ChunkDuration, err = parseOptionalPositiveDuration(name+".chunk_duration", realtime.ChunkDurationText); err != nil {
		return err
	}
	if realtime.ListenDuration, err = parseOptionalPositiveDuration(name+".listen_duration", realtime.ListenDurationText); err != nil {
		return err
	}
	if realtime.QuietDuration, err = parseOptionalPositiveDuration(name+".quiet_duration", realtime.QuietDurationText); err != nil {
		return err
	}
	if realtime.AgentWaitDuration, err = parseOptionalPositiveDuration(name+".agent_wait_duration", realtime.AgentWaitDurationText); err != nil {
		return err
	}
	if realtime.ToolCallTimeout, err = parseOptionalPositiveDuration(name+".tool_call_timeout", realtime.ToolCallTimeoutText); err != nil {
		return err
	}
	return nil
}

func (tts *RealtimeTTSConfig) validate() error {
	return tts.validateNamed("harness.realtime.tts")
}

func (voice *HarnessVoiceTranscribeConfig) validateOpenClaw() error {
	if voice.STT != (VoiceSTTConfig{}) {
		return errors.New("harness.voice_transcribe.stt must be empty for OpenClaw voice-transcribe")
	}
	return voice.HarnessRealtimeConfig.validateNamed("harness.voice_transcribe")
}

func (tts *RealtimeTTSConfig) validateNamed(name string) error {
	if tts.Provider == "" {
		tts.Provider = "openai"
	}
	if tts.Provider != "openai" {
		return fmt.Errorf("%s.provider must be openai", name)
	}
	if tts.APIKeyEnv == "" {
		tts.APIKeyEnv = "OPENAI_API_KEY"
	}
	if !validEnvName(tts.APIKeyEnv) {
		return fmt.Errorf("%s.api_key_env must be an environment variable name", name)
	}
	if strings.ContainsRune(tts.BaseURL, 0) || strings.ContainsRune(tts.Model, 0) || strings.ContainsRune(tts.Voice, 0) || strings.ContainsRune(tts.Instructions, 0) {
		return fmt.Errorf("%s contains an invalid NUL byte", name)
	}
	if tts.Speed != nil && (*tts.Speed < 0.25 || *tts.Speed > 4 || math.IsNaN(*tts.Speed) || math.IsInf(*tts.Speed, 0)) {
		return fmt.Errorf("%s.speed must be between 0.25 and 4", name)
	}
	var err error
	tts.Timeout, err = parseOptionalPositiveDuration(name+".timeout", tts.TimeoutText)
	return err
}

func parseOptionalPositiveDuration(name, value string) (time.Duration, error) {
	if strings.TrimSpace(value) == "" {
		return 0, nil
	}
	duration, err := time.ParseDuration(value)
	if err != nil || duration <= 0 {
		return 0, fmt.Errorf("%s must be a positive Go duration", name)
	}
	return duration, nil
}

func (c *RuntimeConfig) validate() error {
	switch c.Backend {
	case "deepseek", "openai":
		if c.Mode != "external" {
			return fmt.Errorf("runtime.backend %s requires external mode", c.Backend)
		}
		if c.Config.File != "" || c.Config.Executable != "" || c.Config.StartupTimeoutText != "" || c.Config.StopTimeoutText != "" || len(c.Config.GPUIndices) != 0 {
			return fmt.Errorf("external %s runtime.config must be empty", c.Backend)
		}
		return nil
	case "sglang":
	default:
		return errors.New("runtime.backend must be deepseek, sglang, or openai")
	}
	switch c.Mode {
	case "external":
		if c.Config.Executable != "" || c.Config.StartupTimeoutText != "" || c.Config.StopTimeoutText != "" || len(c.Config.GPUIndices) != 0 {
			return errors.New("external SGLang runtime.config must not set executable, timeouts, or gpu_indices")
		}
		return nil
	case "managed":
		if strings.TrimSpace(c.Config.File) == "" {
			return errors.New("managed SGLang runtime.config.file is required")
		}
		if strings.TrimSpace(c.Config.Executable) == "" || strings.ContainsRune(c.Config.Executable, 0) {
			return errors.New("managed SGLang runtime.config.executable is required")
		}
		seenGPU := make(map[int]struct{}, len(c.Config.GPUIndices))
		for _, index := range c.Config.GPUIndices {
			if index < 0 {
				return errors.New("managed SGLang runtime.config.gpu_indices must contain only non-negative indices")
			}
			if _, exists := seenGPU[index]; exists {
				return errors.New("managed SGLang runtime.config.gpu_indices must not contain duplicates")
			}
			seenGPU[index] = struct{}{}
		}
		var err error
		c.Config.StartupTimeout, err = time.ParseDuration(c.Config.StartupTimeoutText)
		if err != nil || c.Config.StartupTimeout <= 0 {
			return errors.New("managed SGLang runtime.config.startup_timeout must be a positive Go duration")
		}
		c.Config.StopTimeout, err = time.ParseDuration(c.Config.StopTimeoutText)
		if err != nil || c.Config.StopTimeout <= 0 {
			return errors.New("managed SGLang runtime.config.stop_timeout must be a positive Go duration")
		}
		return nil
	default:
		return errors.New("runtime.mode must be external or managed")
	}
}

// normalizeV1BaseURL accepts the base URL of an OpenAI-compatible server: an
// absolute HTTP(S) URL whose path is exactly the versioned /v1 prefix.
func normalizeV1BaseURL(baseURL string) (string, error) {
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Scheme != "http" && parsed.Scheme != "https" || parsed.Host == "" || parsed.Opaque != "" || parsed.User != nil || parsed.RawPath != "" || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" || strings.Contains(baseURL, "#") {
		return "", errors.New("must be an absolute HTTP(S) URL without credentials, escaped path, query, or fragment")
	}
	if parsed.Path != "/v1" && parsed.Path != "/v1/" {
		return "", errors.New("path must be exactly /v1")
	}
	parsed.Path = "/v1"
	return parsed.String(), nil
}

func validateExperimentName(name string) error {
	if len(name) > 80 {
		return errors.New("name must not exceed 80 bytes")
	}
	for index, character := range name {
		allowed := character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' || character == '-' || character == '_' || character == '.'
		if !allowed || index == 0 && (character == '-' || character == '.') {
			return errors.New("name must contain only ASCII letters, digits, dashes, underscores, or dots and must not begin with a dash or dot")
		}
	}
	return nil
}

func (stt *VoiceSTTConfig) validate() error {
	if stt.Provider == "" {
		stt.Provider = "openai"
	}
	if stt.Provider != "openai" && stt.Provider != "local" {
		return errors.New("harness.voice_transcribe.stt.provider must be openai or local")
	}
	if stt.Provider == "openai" && stt.Model == "" {
		stt.Model = "gpt-4o-mini-transcribe"
	}
	if stt.Provider == "local" && stt.Model == "" {
		stt.Model = "base"
	}
	if strings.ContainsRune(stt.Provider, 0) || strings.ContainsRune(stt.Model, 0) || strings.ContainsRune(stt.Language, 0) {
		return errors.New("harness.voice_transcribe.stt contains an invalid NUL byte")
	}
	var err error
	stt.Timeout, err = parseOptionalPositiveDuration("harness.voice_transcribe.stt.timeout", stt.TimeoutText)
	return err
}

func (c Versions) validate() error {
	if strings.TrimSpace(c.OpenClaw.Image) == "" {
		return errors.New("openclaw.image is required")
	}
	if err := validateRepositoryPin("terminalbench2", c.TerminalBench2.RepositoryURL, c.TerminalBench2.Revision); err != nil {
		return err
	}
	if err := validateRepositoryPin("deepresearchbench", c.DeepResearchBench.RepositoryURL, c.DeepResearchBench.Revision); err != nil {
		return err
	}
	if err := validateRepositoryPin("sweatlasqa", c.SWEAtlas.RepositoryURL, c.SWEAtlas.Revision); err != nil {
		return err
	}
	if err := validateRepositoryPin("swebenchpro.dataset", c.SWEbenchPro.DatasetRepositoryURL, c.SWEbenchPro.DatasetRevision); err != nil {
		return err
	}
	if err := validateRepositoryPin("swebenchpro.evaluator", c.SWEbenchPro.EvaluatorRepositoryURL, c.SWEbenchPro.EvaluatorRevision); err != nil {
		return err
	}
	if err := containerimage.ValidatePinnedTagOnly(c.OpenClaw.Image); err != nil {
		return fmt.Errorf("openclaw.image: %w", err)
	}
	// A catalog predating a given harness stays valid: only the image the
	// selected harness actually needs is required, and that is enforced by
	// HarnessImage once the profile's harness type is known. Any image that is
	// present must still be pinned.
	if strings.TrimSpace(c.Hermes.Image) != "" {
		if err := containerimage.ValidatePinnedTagOnly(c.Hermes.Image); err != nil {
			return fmt.Errorf("hermes.image: %w", err)
		}
	}
	if strings.TrimSpace(c.AMEMQdrant.Image) != "" {
		if err := containerimage.ValidatePinnedTagOnly(c.AMEMQdrant.Image); err != nil {
			return fmt.Errorf("amem_qdrant.image: %w", err)
		}
	}
	return nil
}

// HarnessImage returns the pinned image the named harness runs from. The set of
// harnesses is enumerated rather than discovered, so an unknown type is an
// error instead of an empty result.
func (c Versions) HarnessImage(harnessType string) (string, error) {
	var image, field string
	switch harnessType {
	case "openclaw":
		image, field = c.OpenClaw.Image, "openclaw.image"
	case "hermes":
		image, field = c.Hermes.Image, "hermes.image"
	default:
		return "", fmt.Errorf("unsupported harness type %q", harnessType)
	}
	if strings.TrimSpace(image) == "" {
		return "", fmt.Errorf("%s is required for harness type %q", field, harnessType)
	}
	return image, nil
}

func validateRepositoryPin(name, repositoryURL, revision string) error {
	if strings.TrimSpace(repositoryURL) == "" {
		return fmt.Errorf("%s.repository_url is required", name)
	}
	if strings.TrimSpace(revision) == "" {
		return fmt.Errorf("%s.revision is required", name)
	}
	repository, err := url.Parse(repositoryURL)
	if err != nil || repository.Scheme != "https" || repository.Host == "" {
		return fmt.Errorf("%s.repository_url must be an absolute HTTPS URL", name)
	}
	if repository.User != nil || repository.RawQuery != "" || repository.Fragment != "" {
		return fmt.Errorf("%s.repository_url must not contain credentials, a query, or a fragment", name)
	}
	if !isHex(revision, 40) {
		return fmt.Errorf("%s.revision must be a 40-character Git revision", name)
	}
	return nil
}

func isHex(value string, characters int) bool {
	if len(value) != characters {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func (m HarnessMemoryConfig) validate(harnessType string) error {
	if m.Provider == "" {
		if len(m.Env) != 0 || m.FrozenState != "" {
			return errors.New("harness.memory.env and frozen_state require harness.memory.provider")
		}
		return nil
	}
	if harnessType != "hermes" {
		return errors.New("harness.memory requires Hermes")
	}
	if !memoryProviderPattern.MatchString(m.Provider) || len(m.Provider) > 64 {
		return errors.New("harness.memory.provider must be a lowercase plugin name of letters, digits, '-' and '_'")
	}
	for key := range m.Env {
		if !memoryEnvKeyPattern.MatchString(key) {
			return fmt.Errorf("harness.memory.env name %q must be uppercase letters, digits and '_'", key)
		}
		if strings.HasPrefix(key, "ARIES_") || strings.HasPrefix(key, "HERMES_") || strings.HasPrefix(key, "TERMINAL_") {
			return fmt.Errorf("harness.memory.env name %q is reserved for ARIES and Hermes", key)
		}
	}
	return nil
}

var (
	memoryProviderPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)
	memoryEnvKeyPattern   = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
)

func validEnvName(value string) bool {
	for i, r := range value {
		if r == '_' || r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' || i > 0 && r >= '0' && r <= '9' {
			continue
		}
		return false
	}
	return value != ""
}
