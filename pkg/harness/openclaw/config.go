package openclaw

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/hyscale-lab/aries/pkg/core"
)

const (
	configContainerPath = "/run/aries/openclaw.json"
	modelKeyPath        = "/run/aries/model.key"
	realtimeKeyPath     = "/run/aries/realtime.key"
	gatewayKeyPath      = "/run/aries/gateway.key"
	gatewayTokenEnv     = "OPENCLAW_GATEWAY_TOKEN"
	launcherPath        = "/run/aries/launch"
	agentWrapperPath    = "/run/aries/run-agent"
	stateContainerPath  = "/home/node/.openclaw"
	workspaceRoot       = "/aries/openclaw"
	// amemPluginID is the npm package / OpenClaw plugin ID for amem
	// (https://amem.owo.lc, github.com/amemhq/amem) — a real OpenClaw plugin
	// (installed via `openclaw plugins install`, not an MCP stdio server),
	// pre-baked into this profile's pinned image (docker/openclaw-amem).
	amemPluginID = "openclaw-amem"
	// amemLLMAPIKeyEnv is the container-side env var amem's own source reads
	// for its LLM calls (note construction/linking/evolution), confirmed
	// against the plugin's shipped openclaw.plugin.json manifest. By default
	// launcherScript exports it from the already-staged primary model key; if
	// a profile configures a distinct amem LLM (harness.amem.llm_*), it's
	// exported from amemLLMKeyPath instead — see launcherScript.
	amemLLMAPIKeyEnv = "AMEM_LLM_API_KEY"
	// amemLLMKeyPath stages a distinct API key for amem's own LLM calls when
	// a profile sets harness.amem.llm_api_key_env — separate from
	// modelKeyPath so the primary task model's key and amem's own key can
	// differ (e.g. an sglang-served primary model alongside a DeepSeek key
	// for amem's internal calls).
	amemLLMKeyPath = "/run/aries/amem-llm.key"
	// losslessClawPluginID is the OpenClaw plugin ID for lossless-claw
	// (github.com/Martian-Engineering/lossless-claw, npm
	// "@martian-engineering/lossless-claw") — a native OpenClaw plugin
	// (installed via `openclaw plugins install`, not an MCP stdio server),
	// pre-baked into this profile's pinned image (docker/openclaw-lossless-claw).
	// Unlike amem it claims the "contextEngine" slot, not "memory" — see
	// (*HarnessConfig).validate's mutual-exclusion check with amem.
	losslessClawPluginID = "lossless-claw"
	// losslessClawLLMAPIKeyEnv is the container-side env var exported for the
	// override OpenClaw model provider lossless-claw's summaryModel/
	// expansionModel point at when a profile configures
	// harness.lossless_claw.llm_*. Unlike amemLLMAPIKeyEnv, this is only ever
	// exported/referenced when an override is configured — with no override,
	// lossless-claw's config omits summaryModel/expansionModel entirely and
	// falls back to OpenClaw's own default model resolution, so no extra
	// provider entry (and thus no extra key) is needed at all.
	losslessClawLLMAPIKeyEnv = "LOSSLESS_CLAW_LLM_API_KEY"
	// losslessClawLLMKeyPath stages the API key for the override provider
	// above, separate from modelKeyPath so the primary task model's key and
	// lossless-claw's own override key can differ.
	losslessClawLLMKeyPath = "/run/aries/lossless-claw-llm.key"
	// losslessClawProviderID names the extra models.providers entry registered
	// only when harness.lossless_claw.llm_* is configured — lossless-claw
	// references its summarization/expansion model as an OpenClaw model ID
	// string ("provider/model"), not a standalone endpoint the way amem's
	// llmProvider/llmBaseURL/llmModel does, so an override means adding a
	// second provider entry rather than passing a raw base URL into the
	// plugin's own config.
	losslessClawProviderID = "lossless-claw-llm"
	// searxngBaseURL matches the fixed network alias
	// (pkg/sandbox/docker/docker.go's `networkAlias = "task-sandbox"`) and
	// port (images/deep-research-bench/Dockerfile) that the DRB task
	// sandbox's built-in SearXNG instance is always reachable at from the
	// OpenClaw harness container, which joins the same per-task Docker
	// network. Not profile-configurable: it's an internal wiring detail, not
	// something a user should need to know or vary.
	searxngBaseURL  = "http://task-sandbox:8888"
	tavilyKeyPath   = "/run/aries/tavily.key"
	tavilyAPIKeyEnv = "TAVILY_API_KEY"
	// firecrawlKeyPath/firecrawlAPIKeyEnv mirror tavilyKeyPath/tavilyAPIKeyEnv:
	// firecrawlAPIKeyEnv is the fixed *container-side* exported variable name
	// the Firecrawl plugin auto-detects its key from, independent of whatever
	// host environment variable a profile's firecrawl_api_key_env names.
	firecrawlKeyPath   = "/run/aries/firecrawl.key"
	firecrawlAPIKeyEnv = "FIRECRAWL_API_KEY"
)

type openClawConfig struct {
	Gateway gatewayConfig  `json:"gateway"`
	Models  modelsConfig   `json:"models"`
	Agents  agentsConfig   `json:"agents"`
	Tools   toolPolicy     `json:"tools"`
	Plugins *pluginsConfig `json:"plugins,omitempty"`
}

type toolPolicy struct {
	Deny    []string          `json:"deny"`
	Web     *webToolsConfig   `json:"web,omitempty"`
	Sandbox *sandboxToolsGate `json:"sandbox,omitempty"`
}

// sandboxToolsGate is a second, independent tool-visibility gate OpenClaw
// applies specifically to sandboxed sessions (agents.defaults.sandbox.mode
// "all", which ARIES always sets): plugin-owned tools like web_search are
// invisible to the agent even when tools.web/plugins configure them, unless
// also explicitly allowed here. `alsoAllow` (not `allow`) is required:
// `allow` replaces the sandbox's default tool set entirely, which would drop
// `exec` — the only way the agent can act at all under ARIES's harness
// protocol.
type sandboxToolsGate struct {
	Tools sandboxToolsAllowList `json:"tools"`
}

type sandboxToolsAllowList struct {
	AlsoAllow []string `json:"alsoAllow,omitempty"`
}

type webToolsConfig struct {
	Search *webSearchToolConfig `json:"search,omitempty"`
}

type webSearchToolConfig struct {
	Provider string `json:"provider"`
}

// pluginsConfig's Allow lists externally-installed (non-bundled) plugins
// OpenClaw's gateway is allowed to actually expose tools from. This is a
// separate trust gate from toolPolicy.Sandbox's alsoAllow: alsoAllow makes a
// tool visible to the sandboxed agent session, but OpenClaw additionally
// refuses to auto-load any non-bundled plugin's tools at all — regardless of
// alsoAllow — unless it's also named here. Without it, `openclaw doctor
// --fix` still installs and enables the plugin (so gateway.log shows it
// selected as the search provider), but web_search is never actually
// registered as a callable tool, and the agent silently falls back to
// guessing URLs for web_fetch instead.
type pluginsConfig struct {
	Entries map[string]pluginEntry `json:"entries"`
	Allow   []string               `json:"allow,omitempty"`
	// Slots resolves an OpenClaw capability slot (e.g. "memory") to the
	// externally-installed plugin that should own it — required for amem:
	// without plugins.slots.memory pointing at amemPluginID, OpenClaw's
	// bundled "memory-core" plugin keeps the memory slot and amem's own
	// tools never activate (confirmed via `openclaw plugins inspect`
	// against a live install: "Error: memory slot set to memory-core").
	Slots map[string]string `json:"slots,omitempty"`
}

type pluginEntry struct {
	Enabled bool `json:"enabled,omitempty"`
	// Config is deliberately untyped: each plugin's config schema shape
	// differs (searxng/tavily/firecrawl nest under a "webSearch" key via
	// pluginConfigBlock below; amem's schema, confirmed against its shipped
	// openclaw.plugin.json, is a flat object — see amemPluginConfig).
	Config any               `json:"config,omitempty"`
	Hooks  *pluginHooksBlock `json:"hooks,omitempty"`
	// LLM/Subagent are lossless-claw-specific host trust policies (confirmed
	// against its shipped README/docs/configuration.md — no equivalent exists
	// for any other plugin wired here): summaryModel/expansionModel overrides
	// are rejected by OpenClaw's runtime unless explicitly trusted here, even
	// though the requested model is already present in models.providers. LLM
	// gates summaryModel/largeFileSummaryModel (routed through OpenClaw's
	// api.runtime.llm.complete capability); Subagent separately gates
	// expansionModel (routed through OpenClaw's sub-agent delegation layer
	// instead) — see modelOverridePolicy's doc comment.
	LLM      *modelOverridePolicy `json:"llm,omitempty"`
	Subagent *modelOverridePolicy `json:"subagent,omitempty"`
}

// modelOverridePolicy is lossless-claw's trust-policy shape for both the
// "llm" and "subagent" pluginEntry blocks (confirmed identical in its README
// examples): AllowModelOverride must be true for OpenClaw to honor the
// plugin's requested model at all, and AllowedModels allowlists which
// "provider/model" strings are trusted (the README explicitly notes "*"
// trusts any target, which renderConfig never uses — it always allowlists
// exactly the one override model a profile configured).
type modelOverridePolicy struct {
	AllowModelOverride bool     `json:"allowModelOverride"`
	AllowedModels      []string `json:"allowedModels,omitempty"`
}

type pluginConfigBlock struct {
	WebSearch webSearchPluginConfig `json:"webSearch"`
}

type webSearchPluginConfig struct {
	BaseURL string `json:"baseUrl"`
}

// pluginHooksBlock gates a plugin's access to elevated capabilities amem
// needs: allowConversationAccess is amem's own name (confirmed in its
// bundled dist/index.js) for permission to read conversation content in its
// agent_end hook, which is how it captures memories at all — without it the
// plugin registers but never actually writes anything back.
type pluginHooksBlock struct {
	AllowConversationAccess bool `json:"allowConversationAccess,omitempty"`
}

// amemPluginConfig is amem's plugins.entries.openclaw-amem.config shape
// (confirmed against its shipped openclaw.plugin.json configSchema).
// llmProvider/llmBaseURL/llmModel route amem's own note-construction/
// linking/evolution LLM calls through an OpenAI-compatible endpoint — by
// default the same one ARIES already configured for the primary task model,
// or a distinct override (see renderConfig's amemLLMBaseURL/amemLLMModel
// params, sourced from harness.amem.llm_base_url/llm_model) when a profile
// needs amem's internal calls on a different model. "openai" always applies
// for LLMProvider regardless: amem's enum only allows "anthropic"/"openai",
// and every provider ARIES lets a profile point amem at (deepseek, sglang,
// or an explicit amem LLM override) speaks the OpenAI-compatible dialect.
// The API key is deliberately not set here (config fields land in the
// world-readable rendered JSON); see amemLLMAPIKeyEnv, exported by
// launcherScript instead. Every other field (agentId, topK, collection, ...)
// is left unset to accept amem's own defaults — notably agentId defaults to
// "main", which is what makes memory persist and link across every harness
// turn within one task occurrence (see amem_qdrant.go's ensureAMEMQdrant:
// the Qdrant volume itself is scoped per task occurrence, not shared across
// separate tasks in the same run).
type amemPluginConfig struct {
	LLMProvider string `json:"llmProvider"`
	LLMBaseURL  string `json:"llmBaseURL"`
	LLMModel    string `json:"llmModel"`
}

// losslessClawPluginConfig is lossless-claw's
// plugins.entries.lossless-claw.config shape (confirmed against the plugin's
// README example: freshTailCount/leafChunkTokens/contextThreshold/
// summaryModel/expansionModel). freshTailCount/leafChunkTokens/contextThreshold
// are left unset (accepting the plugin's own defaults) — only
// SummaryModel/ExpansionModel are ever populated, and only when
// harness.lossless_claw.llm_* configures an override; both point at the same
// "provider/model" string referencing losslessClawProviderID (see
// renderConfig). Left unset, lossless-claw's own default resolves to
// OpenClaw's configured primary agent model — the "reuse primary by default"
// behavior, achieved here without a second endpoint the way amem needs one.
type losslessClawPluginConfig struct {
	SummaryModel   string `json:"summaryModel,omitempty"`
	ExpansionModel string `json:"expansionModel,omitempty"`
}

type gatewayConfig struct {
	Mode   string        `json:"mode"`
	Auth   gatewayAuth   `json:"auth"`
	Remote gatewayRemote `json:"remote"`
}

type gatewayAuth struct {
	Mode  string `json:"mode"`
	Token string `json:"token"`
}

type gatewayRemote struct {
	Token string `json:"token"`
}

type modelsConfig struct {
	Mode      string                    `json:"mode"`
	Providers map[string]providerConfig `json:"providers"`
}

type providerConfig struct {
	BaseURL string        `json:"baseUrl"`
	APIKey  string        `json:"apiKey"`
	API     string        `json:"api"`
	Models  []modelRecord `json:"models"`
}

type modelRecord struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	// MaxTokens is a best-effort attempt to bound the model's completion
	// length (see core.ModelConfig.MaxOutputTokens); OpenClaw's provider/
	// model-catalog schema is not vendored in this repo, so this field name
	// is unverified against OpenClaw's actual config parser. Omitted
	// entirely when unset, matching today's behavior.
	MaxTokens int `json:"maxTokens,omitempty"`
	// ContextWindow is the model's native context window size in tokens
	// (see core.ModelConfig.ContextWindow), confirmed against OpenClaw's
	// docs (models.providers.*.models.*.contextWindow). Without it, a
	// custom provider entry defaults to OpenClaw's own built-in assumption,
	// which has no relation to whatever context length the serving engine
	// (e.g. sglang's --context-length) actually enforces. Omitted entirely
	// when unset, matching today's behavior.
	ContextWindow int `json:"contextWindow,omitempty"`
}

type agentsConfig struct {
	Defaults agentDefaults `json:"defaults"`
}

type agentDefaults struct {
	Model        primaryModel        `json:"model"`
	Sandbox      sandboxConfig       `json:"sandbox"`
	Subagents    *subagentsConfig    `json:"subagents,omitempty"`
	MemorySearch *memorySearchConfig `json:"memorySearch,omitempty"`
}

// memorySearchConfig disables OpenClaw's unrelated, bundled "memory-core"
// semantic-recall feature when amem is enabled — confirmed via a live
// `openclaw doctor` run that it otherwise warns/requires its own
// OPENAI_API_KEY independent of anything amem needs. amem owns the "memory"
// plugin slot instead (see pluginsConfig.Slots); this just keeps the
// unrelated bundled feature out of the way.
type memorySearchConfig struct {
	Enabled bool `json:"enabled"`
}

// subagentsConfig bounds sessions_spawn concurrency. Omitted entirely when no
// limit is configured, leaving OpenClaw's own built-in default in place.
type subagentsConfig struct {
	MaxConcurrent int `json:"maxConcurrent,omitempty"`
}

type primaryModel struct {
	Primary string `json:"primary"`
}

type sandboxConfig struct {
	Mode            string    `json:"mode"`
	Scope           string    `json:"scope"`
	Backend         string    `json:"backend"`
	WorkspaceAccess string    `json:"workspaceAccess"`
	SSH             sshConfig `json:"ssh"`
}

type sshConfig struct {
	Target                string `json:"target"`
	Command               string `json:"command"`
	WorkspaceRoot         string `json:"workspaceRoot"`
	StrictHostKeyChecking bool   `json:"strictHostKeyChecking"`
	UpdateHostKeys        bool   `json:"updateHostKeys"`
	IdentityFile          string `json:"identityFile"`
	KnownHostsFile        string `json:"knownHostsFile"`
}

// amemToolNames is openclaw-amem@2.1.1's full tool surface, with no plugin-ID
// prefix — confirmed against a live `openclaw gateway run` boot log ("[plugins]
// openclaw-amem: memory_search, memory_add, memory_list, memory_consolidate,
// memory_quality_scan tools registered"). A version bump could add/rename
// tools, in which case this list needs re-capturing from a live run rather
// than guessed, exactly like the old MCP-based amem's tool list before it.
var amemToolNames = []string{
	"memory_search", "memory_add", "memory_list", "memory_consolidate", "memory_quality_scan",
}

// losslessClawToolNames is lossless-claw's agent-accessible tool surface per
// its README ("Agent Tools" section: lcm_grep, lcm_describe,
// lcm_expand_query). Like amemToolNames, a version bump could add/rename
// tools, requiring re-verification against a live gateway boot log.
var losslessClawToolNames = []string{
	"lcm_grep", "lcm_describe", "lcm_expand_query",
}

func renderConfig(model core.ModelConfig, endpoint core.ToolEndpoint, webSearchEnabled bool, searchProvider string, extractEnabled, subagentsEnabled, amemEnabled bool, maxConcurrentSubagents int, amemLLMBaseURL, amemLLMModel string, losslessClawEnabled bool, losslessClawLLMBaseURL, losslessClawLLMModel string) ([]byte, error) {
	if err := validateModel(model); err != nil {
		return nil, err
	}
	if model.Provider == "sglang" {
		model.BaseURL, _ = normalizeSGLangBaseURL(model.BaseURL)
	}
	if err := validateEndpoint(endpoint); err != nil {
		return nil, err
	}
	providerID := model.Provider
	if providerID == "deepseek" {
		providerID = "aries"
	}
	configuration := openClawConfig{
		Gateway: gatewayConfig{
			Mode:   "local",
			Auth:   gatewayAuth{Mode: "token", Token: "${" + gatewayTokenEnv + "}"},
			Remote: gatewayRemote{Token: "${" + gatewayTokenEnv + "}"},
		},
		Models: modelsConfig{
			Mode: "merge",
			Providers: map[string]providerConfig{
				providerID: {
					BaseURL: model.BaseURL,
					APIKey:  "${" + model.APIKeyEnv + "}",
					API:     "openai-completions",
					Models:  []modelRecord{{ID: model.Model, Name: model.Model, MaxTokens: model.MaxOutputTokens, ContextWindow: model.ContextWindow}},
				},
			},
		},
		Agents: agentsConfig{Defaults: agentDefaults{
			Model: primaryModel{Primary: providerID + "/" + model.Model},
			Sandbox: sandboxConfig{
				Mode: "all", Scope: "shared", Backend: "ssh", WorkspaceAccess: "rw",
				SSH: sshConfig{
					Target: endpoint.Username + "@" + endpoint.Address, Command: endpoint.ClientCommand,
					WorkspaceRoot: workspaceRoot, StrictHostKeyChecking: true, UpdateHostKeys: false,
					IdentityFile: endpoint.IdentityFile, KnownHostsFile: endpoint.KnownHostsFile,
				},
			},
		}},
		Tools: toolPolicy{Deny: denyToolList(subagentsEnabled)},
	}
	if subagentsEnabled && maxConcurrentSubagents > 0 {
		configuration.Agents.Defaults.Subagents = &subagentsConfig{MaxConcurrent: maxConcurrentSubagents}
	}
	var alsoAllow []string
	pluginEntries := map[string]pluginEntry{}
	var pluginAllow []string
	var pluginSlots map[string]string
	if webSearchEnabled {
		alsoAllow = append(alsoAllow, "web_search", "web_fetch")
		var entries map[string]pluginEntry
		var allow []string
		switch searchProvider {
		case "firecrawl":
			// Firecrawl fully replaces SearXNG as the active search provider
			// (not layered alongside it the way Tavily's extract-only entry
			// is below): no searxng entry is registered at all. The plugin
			// entry carries no Config/apiKey — same "secret never touches
			// JSON" approach as tavily's entry — it auto-detects its key from
			// the ambient FIRECRAWL_API_KEY env var the launcher exports.
			//
			// Firecrawl's plugin manifest declares a webFetchProviders
			// contract (verified against the actual installed OpenClaw
			// package), so enabling it here also makes OpenClaw auto-select
			// it as the web_fetch fallback — confirmed empirically, no known
			// config override prevents this. Profiles that don't want that
			// incremental fetch cost should use "tavily" instead (below),
			// whose manifest declares no webFetchProviders contract.
			configuration.Tools.Web = &webToolsConfig{Search: &webSearchToolConfig{Provider: "firecrawl"}}
			entries = map[string]pluginEntry{"firecrawl": {Enabled: true}}
			allow = []string{"firecrawl"}
		case "tavily":
			// Same "no Config/apiKey in JSON" approach as firecrawl/tavily-
			// extract above. Unlike Firecrawl, Tavily's plugin manifest
			// declares only a webSearchProviders contract (no
			// webFetchProviders), so web_fetch stays on OpenClaw's native
			// fetcher — verified against the actual installed plugin
			// manifest and a live gateway startup, not just documentation.
			configuration.Tools.Web = &webToolsConfig{Search: &webSearchToolConfig{Provider: "tavily"}}
			entries = map[string]pluginEntry{"tavily": {Enabled: true}}
			allow = []string{"tavily"}
		default:
			configuration.Tools.Web = &webToolsConfig{Search: &webSearchToolConfig{Provider: "searxng"}}
			entries = map[string]pluginEntry{
				"searxng": {Config: &pluginConfigBlock{WebSearch: webSearchPluginConfig{BaseURL: searxngBaseURL}}},
			}
			allow = []string{"searxng"}
		}
		if extractEnabled {
			// tavily_extract only: the entry deliberately carries no apiKey
			// (the launcher exports it as TAVILY_API_KEY instead, so it
			// never enters this JSON), and alsoAllow omits tavily_search so
			// Tavily backs extraction only — search stays whatever
			// searchProvider selected. The entry/allow-list entry may already
			// exist if searchProvider is "tavily" itself (same plugin backing
			// both capabilities); only add it if it isn't there yet, so the
			// two capabilities compose without a duplicate "tavily" in allow.
			alsoAllow = append(alsoAllow, "tavily_extract")
			if _, exists := entries["tavily"]; !exists {
				entries["tavily"] = pluginEntry{Enabled: true}
				allow = append(allow, "tavily")
			}
		}
		for id, entry := range entries {
			pluginEntries[id] = entry
		}
		pluginAllow = append(pluginAllow, allow...)
	}
	if amemEnabled {
		// amem (https://amem.owo.lc, npm package "openclaw-amem", from
		// github.com/amemhq/amem) is a real OpenClaw plugin pre-installed into
		// this profile's pinned image (docker/openclaw-amem) via `openclaw
		// plugins install`, not an MCP stdio server. Its tools are gated by the
		// same sandbox tools.alsoAllow list as every other plugin here —
		// verified against a live gateway run — so amemToolNames (see its doc
		// comment) must be kept in sync with a version bump the same way the
		// old MCP-based amem's tool list had to be.
		//
		// llmProvider/llmBaseURL/llmModel route amem's own LLM calls through an
		// OpenAI-compatible endpoint (see amemPluginConfig's doc comment); its
		// API key is exported separately by launcherScript under
		// amemLLMAPIKeyEnv, never placed in this rendered JSON. Leaving
		// config.agentId unset accepts amem's own "main" default for every
		// task, which is what makes memory persist and link across all tasks
		// sharing the run's Qdrant volume (see amem_qdrant.go).
		//
		// By default this reuses the primary task model/endpoint (empty
		// amemLLMBaseURL/amemLLMModel). A profile can override both together
		// (harness.amem.llm_base_url/llm_model/llm_api_key_env) to point
		// amem's own calls at a different model — added after a live run
		// showed the primary sglang-served Qwen model wasn't reliably
		// producing the plain JSON amem asks for internally (it prefaced
		// answers with prose instead of raw JSON), silently defaulting note
		// metadata to empty and merge/link decisions to "no" every time.
		llmBaseURL, llmModel := model.BaseURL, model.Model
		if amemLLMBaseURL != "" {
			llmBaseURL = amemLLMBaseURL
		}
		if amemLLMModel != "" {
			llmModel = amemLLMModel
		}
		pluginEntries[amemPluginID] = pluginEntry{
			Enabled: true,
			Hooks:   &pluginHooksBlock{AllowConversationAccess: true},
			Config: &amemPluginConfig{
				LLMProvider: "openai",
				LLMBaseURL:  llmBaseURL,
				LLMModel:    llmModel,
			},
		}
		pluginAllow = append(pluginAllow, amemPluginID)
		pluginSlots = map[string]string{"memory": amemPluginID}
		configuration.Agents.Defaults.MemorySearch = &memorySearchConfig{Enabled: false}
		alsoAllow = append(alsoAllow, amemToolNames...)
	}
	if losslessClawEnabled {
		// lossless-claw (github.com/Martian-Engineering/lossless-claw, npm
		// "@martian-engineering/lossless-claw") is a real OpenClaw plugin
		// pre-installed into this profile's pinned image
		// (docker/openclaw-lossless-claw) via `openclaw plugins install`, not
		// an MCP stdio server. Its tools are gated by the same sandbox
		// tools.alsoAllow list as every other plugin here (losslessClawToolNames).
		//
		// Unlike amem, lossless-claw references its summarization/expansion
		// model as an OpenClaw model ID string resolved through
		// models.providers/agents.defaults.model, not a standalone endpoint —
		// so an override means registering a second provider entry and
		// pointing summaryModel/expansionModel at "<provider>/<model>", rather
		// than passing a raw base URL into the plugin's own config the way
		// amemPluginConfig does. Its API key is exported separately by
		// launcherScript under losslessClawLLMAPIKeyEnv, never placed in this
		// rendered JSON.
		//
		// By default (empty losslessClawLLMBaseURL/losslessClawLLMModel) no
		// extra provider is registered and summaryModel/expansionModel stay
		// unset, which lossless-claw's own default resolves to OpenClaw's
		// configured primary agent model.
		//
		// An override additionally requires explicit host trust policy
		// (confirmed against the plugin's shipped README/docs/configuration.md):
		// OpenClaw's runtime rejects a plugin-requested summaryModel unless
		// plugins.entries.lossless-claw.llm.allowModelOverride/allowedModels
		// trusts it, and separately rejects expansionModel (routed through
		// OpenClaw's sub-agent delegation layer, not api.runtime.llm.complete)
		// unless the sibling "subagent" block trusts it — merely having the
		// model in models.providers is not sufficient on its own.
		entry := pluginEntry{Enabled: true}
		cfg := losslessClawPluginConfig{}
		if losslessClawLLMBaseURL != "" && losslessClawLLMModel != "" {
			configuration.Models.Providers[losslessClawProviderID] = providerConfig{
				BaseURL: losslessClawLLMBaseURL,
				APIKey:  "${" + losslessClawLLMAPIKeyEnv + "}",
				API:     "openai-completions",
				Models:  []modelRecord{{ID: losslessClawLLMModel, Name: losslessClawLLMModel}},
			}
			modelRef := losslessClawProviderID + "/" + losslessClawLLMModel
			cfg.SummaryModel, cfg.ExpansionModel = modelRef, modelRef
			entry.LLM = &modelOverridePolicy{AllowModelOverride: true, AllowedModels: []string{modelRef}}
			entry.Subagent = &modelOverridePolicy{AllowModelOverride: true, AllowedModels: []string{modelRef}}
		}
		entry.Config = &cfg
		pluginEntries[losslessClawPluginID] = entry
		pluginAllow = append(pluginAllow, losslessClawPluginID)
		pluginSlots = map[string]string{"contextEngine": losslessClawPluginID}
		alsoAllow = append(alsoAllow, losslessClawToolNames...)
	}
	if len(pluginEntries) > 0 || len(pluginAllow) > 0 || len(pluginSlots) > 0 {
		configuration.Plugins = &pluginsConfig{Entries: pluginEntries, Allow: pluginAllow, Slots: pluginSlots}
	}
	if len(alsoAllow) > 0 {
		configuration.Tools.Sandbox = &sandboxToolsGate{Tools: sandboxToolsAllowList{AlsoAllow: alsoAllow}}
	}
	var output bytes.Buffer
	encoder := json.NewEncoder(&output)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(configuration); err != nil {
		return nil, fmt.Errorf("encode OpenClaw config: %w", err)
	}
	return output.Bytes(), nil
}

func denyToolList(subagentsEnabled bool) []string {
	// OpenClaw's native SSH filesystem helpers require python3 inside the
	// remote image. Terminal-Bench images do not promise it, while the exec
	// tool has the same sandbox access without changing task images.
	deny := []string{"read", "write", "edit", "apply_patch"}
	if !subagentsEnabled {
		// sessions_spawn/sessions_yield let the agent defer its final answer
		// to async sub-agents and pause for their completion events, but
		// ARIES's harness protocol sends exactly one agent request and treats
		// the first terminal response as final (docs/design/harness.md) — it
		// has no continuation mechanism to relay those events back. A yield
		// ends the run immediately, stranding any spawned sub-agents
		// mid-task. Denying both (the default) forces the agent to complete
		// its work within the single turn ARIES can actually observe.
		// subagentsEnabled opts back into both tools for callers who accept
		// that a spawned sub-agent's completion may go unobserved.
		deny = append(deny, "sessions_spawn", "sessions_yield")
	}
	return deny
}

func validateModel(model core.ModelConfig) error {
	if model.Provider != "deepseek" && model.Provider != "sglang" {
		return errors.New("OpenClaw model provider must be deepseek or sglang")
	}
	if model.Provider == "sglang" {
		if _, err := normalizeSGLangBaseURL(model.BaseURL); err != nil {
			return fmt.Errorf("OpenClaw SGLang base URL: %w", err)
		}
	} else {
		parsed, err := url.Parse(model.BaseURL)
		if err != nil || parsed.Host == "" || parsed.Scheme != "http" && parsed.Scheme != "https" {
			return errors.New("OpenClaw model base URL must be absolute HTTP(S)")
		}
		if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
			return errors.New("OpenClaw model base URL must not contain credentials, query, or fragment")
		}
	}
	if strings.TrimSpace(model.Model) == "" || strings.ContainsAny(model.Model, "\x00\r\n") {
		return errors.New("OpenClaw model ID is invalid")
	}
	if !validEnvironmentName(model.APIKeyEnv) {
		return errors.New("OpenClaw API-key environment name is invalid")
	}
	return nil
}

func normalizeSGLangBaseURL(baseURL string) (string, error) {
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

func validateEndpoint(endpoint core.ToolEndpoint) error {
	if endpoint.Protocol != "ssh" || endpoint.Username != "aries" || strings.TrimSpace(endpoint.Network) == "" {
		return errors.New("OpenClaw requires a task-local SSH endpoint")
	}
	host, port, err := net.SplitHostPort(endpoint.Address)
	if err != nil || net.ParseIP(host) == nil || port == "" {
		return errors.New("OpenClaw SSH address must be an IP host and port")
	}
	paths := map[string]string{
		"client command": endpoint.ClientCommand, "client source": endpoint.ClientSourceFile,
		"identity": endpoint.IdentityFile, "identity source": endpoint.IdentitySourceFile,
		"known-hosts": endpoint.KnownHostsFile, "known-hosts source": endpoint.KnownHostsSourceFile,
	}
	for name, path := range paths {
		if path == "" || strings.ContainsRune(path, 0) || !filepath.IsAbs(path) || filepath.Clean(path) != path {
			return fmt.Errorf("OpenClaw %s path must be absolute and clean", name)
		}
	}
	if endpoint.ClientCommand != "/opt/aries/bin/aries-ssh" || endpoint.IdentityFile != "/run/aries/ssh/id_ed25519" || endpoint.KnownHostsFile != "/run/aries/ssh/known_hosts" {
		return errors.New("OpenClaw endpoint paths do not match the pinned bridge contract")
	}
	return nil
}

func validEnvironmentName(value string) bool {
	for index, r := range value {
		if r == '_' || r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' || index > 0 && r >= '0' && r <= '9' {
			continue
		}
		return false
	}
	return value != ""
}

func launcherScript(apiKeyEnv, realtimeAPIKeyEnv string, extractEnabled, firecrawlEnabled, tavilySearchEnabled, amemEnabled, amemLLMOverride, losslessClawLLMOverride bool) []byte {
	script := "#!/bin/sh\nset -eu\nmodel_key=$(cat " + modelKeyPath + ")\ngateway_key=$(cat " + gatewayKeyPath + ")\nexport " + apiKeyEnv + "=\"$model_key\"\nexport " + gatewayTokenEnv + "=\"$gateway_key\"\n"
	if realtimeAPIKeyEnv != "" {
		script += "realtime_key=$(cat " + realtimeKeyPath + ")\nexport " + realtimeAPIKeyEnv + "=\"$realtime_key\"\nunset realtime_key\n"
	}
	if extractEnabled || tavilySearchEnabled {
		script += "tavily_key=$(cat " + tavilyKeyPath + ")\nexport " + tavilyAPIKeyEnv + "=\"$tavily_key\"\nunset tavily_key\n"
	}
	if firecrawlEnabled {
		script += "firecrawl_key=$(cat " + firecrawlKeyPath + ")\nexport " + firecrawlAPIKeyEnv + "=\"$firecrawl_key\"\nunset firecrawl_key\n"
	}
	if amemLLMOverride {
		// A profile configured a distinct LLM for amem's own internal calls
		// (harness.amem.llm_*, see Options.AMEMLLMAPIKeyEnv) — its key is
		// staged separately, never derived from the primary model key.
		script += "amem_llm_key=$(cat " + amemLLMKeyPath + ")\nexport " + amemLLMAPIKeyEnv + "=\"$amem_llm_key\"\nunset amem_llm_key\n"
	} else if amemEnabled {
		// Default: reuses the already-loaded primary model key rather than
		// staging a separate secret file — amem's own LLM calls are
		// configured (see amemPluginConfig) to route through the same
		// OpenAI-compatible endpoint as the primary task model.
		script += "export " + amemLLMAPIKeyEnv + "=\"$model_key\"\n"
	}
	if losslessClawLLMOverride {
		// A profile configured a distinct model for lossless-claw's own
		// summarization/expansion calls (harness.lossless_claw.llm_*, see
		// Options.LosslessClawLLMAPIKeyEnv) — its key is staged separately.
		// Unlike amem, there is no "else" default-export branch: without an
		// override, renderConfig never registers the extra provider entry
		// that would reference this env var at all.
		script += "lcm_llm_key=$(cat " + losslessClawLLMKeyPath + ")\nexport " + losslessClawLLMAPIKeyEnv + "=\"$lcm_llm_key\"\nunset lcm_llm_key\n"
	}
	script += "unset model_key gateway_key\nexec \"$@\"\n"
	return []byte(script)
}
