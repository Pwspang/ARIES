package openclaw

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/hyscale-lab/aries/pkg/core"
)

func testEndpoint() core.ToolEndpoint {
	return core.ToolEndpoint{
		Protocol: "ssh", Address: "172.22.0.1:39425", Username: "aries", Network: "aries-net-test",
		ClientCommand: "/opt/aries/bin/aries-ssh", ClientSourceFile: "/host/aries-ssh",
		IdentityFile: "/run/aries/ssh/id_ed25519", IdentitySourceFile: "/host/id_ed25519",
		KnownHostsFile: "/run/aries/ssh/known_hosts", KnownHostsSourceFile: "/host/known_hosts",
	}
}

func testModel() core.ModelConfig {
	return core.ModelConfig{Provider: "deepseek", BaseURL: "http://fake-model:8080/v1", Model: "deterministic-model", APIKeyEnv: "ARIES_FAKE_API_KEY"}
}

func TestRenderConfigLocksProviderSharedSSHAndPlaceholder(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), false, "", false, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(content, []byte("dummy-secret")) || !bytes.Contains(content, []byte(`"apiKey": "${ARIES_FAKE_API_KEY}"`)) {
		t.Fatalf("API key placeholder = %s", content)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	provider := configuration.Models.Providers["aries"]
	sandbox := configuration.Agents.Defaults.Sandbox
	if configuration.Gateway.Mode != "local" || configuration.Models.Mode != "merge" || provider.API != "openai-completions" || provider.BaseURL != testModel().BaseURL {
		t.Fatalf("provider config = %#v", configuration)
	}
	if configuration.Gateway.Auth.Mode != "token" || configuration.Gateway.Auth.Token != "${OPENCLAW_GATEWAY_TOKEN}" || configuration.Gateway.Remote.Token != "${OPENCLAW_GATEWAY_TOKEN}" {
		t.Fatalf("gateway config = %#v", configuration.Gateway)
	}
	if configuration.Agents.Defaults.Model.Primary != "aries/deterministic-model" || sandbox.Mode != "all" || sandbox.Scope != "shared" || sandbox.Backend != "ssh" || sandbox.WorkspaceAccess != "rw" {
		t.Fatalf("agent config = %#v", configuration.Agents.Defaults)
	}
	if sandbox.SSH.Target != "aries@172.22.0.1:39425" || sandbox.SSH.Command != "/opt/aries/bin/aries-ssh" || sandbox.SSH.WorkspaceRoot != workspaceRoot || !sandbox.SSH.StrictHostKeyChecking || sandbox.SSH.UpdateHostKeys {
		t.Fatalf("SSH config = %#v", sandbox.SSH)
	}
	if got := strings.Join(configuration.Tools.Deny, ","); got != "read,write,edit,apply_patch,sessions_spawn,sessions_yield" {
		t.Fatalf("tool deny list = %q", got)
	}
}

func TestRenderConfigOmitsMaxTokensWhenUnset(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), false, "", false, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(content, []byte("maxTokens")) {
		t.Fatalf("expected no maxTokens field when MaxOutputTokens is unset, got %s", content)
	}
}

func TestRenderConfigSetsMaxTokensWhenConfigured(t *testing.T) {
	model := testModel()
	model.MaxOutputTokens = 32000
	content, err := renderConfig(model, testEndpoint(), false, "", false, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	provider := configuration.Models.Providers["aries"]
	if len(provider.Models) != 1 || provider.Models[0].MaxTokens != 32000 {
		t.Fatalf("provider models = %#v, want MaxTokens=32000", provider.Models)
	}
}

func TestRenderConfigSelectsSGLangProviderWithoutSerializingKey(t *testing.T) {
	model := testModel()
	model.Provider = "sglang"
	model.APIKeyEnv = "SGLANG_API_KEY"
	content, err := renderConfig(model, testEndpoint(), false, "", false, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	provider, ok := configuration.Models.Providers["sglang"]
	if !ok || len(configuration.Models.Providers) != 1 || provider.API != "openai-completions" || provider.APIKey != "${SGLANG_API_KEY}" || configuration.Agents.Defaults.Model.Primary != "sglang/deterministic-model" {
		t.Fatalf("configuration = %#v", configuration)
	}
	if bytes.Contains(content, []byte("dummy-local-key")) {
		t.Fatalf("serialized key: %s", content)
	}
}

func TestRenderConfigNormalizesAndStrictlyValidatesSGLangBaseURL(t *testing.T) {
	model := testModel()
	model.Provider = "sglang"
	model.BaseURL += "/"
	content, err := renderConfig(model, testEndpoint(), false, "", false, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if got := configuration.Models.Providers["sglang"].BaseURL; got != testModel().BaseURL {
		t.Fatalf("normalized base URL = %q", got)
	}
	for _, invalid := range []string{"http://host/v1/v1", "http://host/v1?", "http://host/v%31"} {
		model.BaseURL = invalid
		if _, err := renderConfig(model, testEndpoint(), false, "", false, false, false, 0, "", "", false, "", ""); err == nil {
			t.Fatalf("accepted SGLang base URL %q", invalid)
		}
	}
}

func TestRenderConfigRejectsInvalidInputs(t *testing.T) {
	for name, mutate := range map[string]func(*core.ModelConfig, *core.ToolEndpoint){
		"provider": func(model *core.ModelConfig, _ *core.ToolEndpoint) { model.Provider = "other" },
		"base URL": func(model *core.ModelConfig, _ *core.ToolEndpoint) { model.BaseURL = "file:///tmp/model" },
		"model":    func(model *core.ModelConfig, _ *core.ToolEndpoint) { model.Model = "\n" },
		"key env":  func(model *core.ModelConfig, _ *core.ToolEndpoint) { model.APIKeyEnv = "bad-name" },
		"protocol": func(_ *core.ModelConfig, endpoint *core.ToolEndpoint) { endpoint.Protocol = "http" },
		"network":  func(_ *core.ModelConfig, endpoint *core.ToolEndpoint) { endpoint.Network = "" },
		"address":  func(_ *core.ModelConfig, endpoint *core.ToolEndpoint) { endpoint.Address = "task-sandbox:2222" },
		"identity": func(_ *core.ModelConfig, endpoint *core.ToolEndpoint) { endpoint.IdentityFile = "/wrong" },
	} {
		t.Run(name, func(t *testing.T) {
			model, endpoint := testModel(), testEndpoint()
			mutate(&model, &endpoint)
			if _, err := renderConfig(model, endpoint, false, "", false, false, false, 0, "", "", false, "", ""); err == nil {
				t.Fatal("invalid input was accepted")
			}
		})
	}
}

func TestRenderConfigAcceptsLowercaseEnvironmentName(t *testing.T) {
	model := testModel()
	model.APIKeyEnv = "aries_fake_api_key"
	if _, err := renderConfig(model, testEndpoint(), false, "", false, false, false, 0, "", "", false, "", ""); err != nil {
		t.Fatalf("renderConfig() rejected a valid environment name: %v", err)
	}
}

func TestRenderConfigOmitsWebSearchWhenDisabled(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), false, "", false, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	// Note: "sandbox" is not checked here as a raw substring — it's also the
	// key for the unrelated, always-present agents.defaults.sandbox (SSH)
	// block. tools.Sandbox is checked below via the parsed struct instead.
	if bytes.Contains(content, []byte(`"web"`)) || bytes.Contains(content, []byte(`"plugins"`)) {
		t.Fatalf("disabled web search leaked into rendered config: %s", content)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Tools.Web != nil || configuration.Plugins != nil || configuration.Tools.Sandbox != nil {
		t.Fatalf("configuration = %#v", configuration)
	}
}

func TestRenderConfigEnablesSearXNGWebSearch(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), true, "", false, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Tools.Web == nil || configuration.Tools.Web.Search == nil || configuration.Tools.Web.Search.Provider != "searxng" {
		t.Fatalf("tools.web.search = %#v", configuration.Tools.Web)
	}
	if configuration.Plugins == nil {
		t.Fatal("plugins block missing")
	}
	entry, ok := configuration.Plugins.Entries["searxng"]
	block, blockOK := entry.Config.(map[string]any)
	webSearch, webSearchOK := block["webSearch"].(map[string]any)
	if !ok || !blockOK || !webSearchOK || webSearch["baseUrl"] != searxngBaseURL {
		t.Fatalf("plugins.entries.searxng = %#v", configuration.Plugins.Entries)
	}
	if len(configuration.Plugins.Allow) != 1 || configuration.Plugins.Allow[0] != "searxng" {
		t.Fatalf("plugins.allow = %#v, want [\"searxng\"]: without it OpenClaw never actually registers web_search as a callable tool", configuration.Plugins.Allow)
	}
	if configuration.Tools.Sandbox == nil {
		t.Fatal("tools.sandbox gate missing: web_search/web_fetch would be invisible to a sandboxed session")
	}
	alsoAllow := configuration.Tools.Sandbox.Tools.AlsoAllow
	if len(alsoAllow) != 2 || alsoAllow[0] != "web_search" || alsoAllow[1] != "web_fetch" {
		t.Fatalf("tools.sandbox.tools.alsoAllow = %#v", alsoAllow)
	}
}

func TestRenderConfigOmitsAMEMPluginWhenDisabled(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), false, "", false, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(content, []byte(amemPluginID)) {
		t.Fatalf("disabled amem leaked its plugin ID into rendered config: %s", content)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Plugins != nil {
		t.Fatalf("configuration.Plugins = %#v, want nil", configuration.Plugins)
	}
	if configuration.Agents.Defaults.MemorySearch != nil {
		t.Fatalf("configuration.Agents.Defaults.MemorySearch = %#v, want nil", configuration.Agents.Defaults.MemorySearch)
	}
}

func TestRenderConfigEnablesAMEMPlugin(t *testing.T) {
	model := testModel()
	content, err := renderConfig(model, testEndpoint(), false, "", false, false, true, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Plugins == nil {
		t.Fatal("plugins block missing")
	}
	entry, ok := configuration.Plugins.Entries[amemPluginID]
	if !ok || !entry.Enabled {
		t.Fatalf("plugins.entries[%q] = %#v, want enabled", amemPluginID, entry)
	}
	if entry.Hooks == nil || !entry.Hooks.AllowConversationAccess {
		t.Fatalf("plugins.entries[%q].hooks = %#v, want allowConversationAccess", amemPluginID, entry.Hooks)
	}
	block, ok := entry.Config.(map[string]any)
	if !ok || block["llmProvider"] != "openai" || block["llmBaseURL"] != model.BaseURL || block["llmModel"] != model.Model {
		t.Fatalf("plugins.entries[%q].config = %#v", amemPluginID, entry.Config)
	}
	if len(configuration.Plugins.Allow) != 1 || configuration.Plugins.Allow[0] != amemPluginID {
		t.Fatalf("plugins.allow = %#v, want [%q]", configuration.Plugins.Allow, amemPluginID)
	}
	if configuration.Plugins.Slots["memory"] != amemPluginID {
		t.Fatalf("plugins.slots.memory = %q, want %q: without it the bundled memory-core plugin keeps the slot", configuration.Plugins.Slots["memory"], amemPluginID)
	}
	if configuration.Agents.Defaults.MemorySearch == nil || configuration.Agents.Defaults.MemorySearch.Enabled {
		t.Fatalf("agents.defaults.memorySearch = %#v, want disabled", configuration.Agents.Defaults.MemorySearch)
	}
	if configuration.Tools.Sandbox == nil {
		t.Fatal("tools.sandbox gate missing: amem's tools would be stripped from the sandboxed session")
	}
	alsoAllow := configuration.Tools.Sandbox.Tools.AlsoAllow
	if len(alsoAllow) != len(amemToolNames) {
		t.Fatalf("tools.sandbox.tools.alsoAllow = %#v, want amem's %d tool names", alsoAllow, len(amemToolNames))
	}
	for i, name := range amemToolNames {
		if alsoAllow[i] != name {
			t.Fatalf("tools.sandbox.tools.alsoAllow[%d] = %q, want %q", i, alsoAllow[i], name)
		}
	}
}

func TestRenderConfigAMEMPluginOverridesLLM(t *testing.T) {
	model := testModel()
	content, err := renderConfig(model, testEndpoint(), false, "", false, false, true, 0, "https://api.deepseek.com", "deepseek-v4-flash", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	entry, ok := configuration.Plugins.Entries[amemPluginID]
	block, blockOK := entry.Config.(map[string]any)
	if !ok || !blockOK || block["llmProvider"] != "openai" || block["llmBaseURL"] != "https://api.deepseek.com" || block["llmModel"] != "deepseek-v4-flash" {
		t.Fatalf("plugins.entries[%q].config = %#v, want the overridden LLM, not the primary model (%s/%s)", amemPluginID, entry.Config, model.BaseURL, model.Model)
	}
}

func TestRenderConfigOmitsLosslessClawPluginWhenDisabled(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), false, "", false, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(content, []byte(losslessClawPluginID)) {
		t.Fatalf("disabled lossless-claw leaked its plugin ID into rendered config: %s", content)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Plugins != nil {
		t.Fatalf("configuration.Plugins = %#v, want nil", configuration.Plugins)
	}
}

func TestRenderConfigEnablesLosslessClawPluginReusingPrimaryModel(t *testing.T) {
	model := testModel()
	content, err := renderConfig(model, testEndpoint(), false, "", false, false, false, 0, "", "", true, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Plugins == nil {
		t.Fatal("plugins block missing")
	}
	entry, ok := configuration.Plugins.Entries[losslessClawPluginID]
	if !ok || !entry.Enabled {
		t.Fatalf("plugins.entries[%q] = %#v, want enabled", losslessClawPluginID, entry)
	}
	if entry.Hooks != nil {
		t.Fatalf("plugins.entries[%q].hooks = %#v, want nil (no allowConversationAccess needed)", losslessClawPluginID, entry.Hooks)
	}
	if entry.LLM != nil || entry.Subagent != nil {
		t.Fatalf("plugins.entries[%q].llm/subagent = %#v/%#v, want nil: no override was requested", losslessClawPluginID, entry.LLM, entry.Subagent)
	}
	block, ok := entry.Config.(map[string]any)
	if !ok {
		t.Fatalf("plugins.entries[%q].config = %#v", losslessClawPluginID, entry.Config)
	}
	if _, present := block["summaryModel"]; present {
		t.Fatalf("plugins.entries[%q].config = %#v, want summaryModel omitted so lossless-claw resolves OpenClaw's own default (the primary model)", losslessClawPluginID, entry.Config)
	}
	if _, present := configuration.Models.Providers[losslessClawProviderID]; present {
		t.Fatalf("no override configured, but an extra %q provider was registered", losslessClawProviderID)
	}
	if len(configuration.Plugins.Allow) != 1 || configuration.Plugins.Allow[0] != losslessClawPluginID {
		t.Fatalf("plugins.allow = %#v, want [%q]", configuration.Plugins.Allow, losslessClawPluginID)
	}
	if configuration.Plugins.Slots["contextEngine"] != losslessClawPluginID {
		t.Fatalf("plugins.slots.contextEngine = %q, want %q", configuration.Plugins.Slots["contextEngine"], losslessClawPluginID)
	}
	if configuration.Agents.Defaults.MemorySearch != nil {
		t.Fatalf("agents.defaults.memorySearch = %#v, want untouched: lossless-claw claims contextEngine, not memory", configuration.Agents.Defaults.MemorySearch)
	}
	alsoAllow := configuration.Tools.Sandbox.Tools.AlsoAllow
	if len(alsoAllow) != len(losslessClawToolNames) {
		t.Fatalf("tools.sandbox.tools.alsoAllow = %#v, want lossless-claw's %d tool names", alsoAllow, len(losslessClawToolNames))
	}
	for i, name := range losslessClawToolNames {
		if alsoAllow[i] != name {
			t.Fatalf("tools.sandbox.tools.alsoAllow[%d] = %q, want %q", i, alsoAllow[i], name)
		}
	}
}

func TestRenderConfigLosslessClawPluginOverridesLLM(t *testing.T) {
	model := testModel()
	content, err := renderConfig(model, testEndpoint(), false, "", false, false, false, 0, "", "", true, "https://api.deepseek.com", "deepseek-v4-flash")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	provider, ok := configuration.Models.Providers[losslessClawProviderID]
	if !ok || provider.BaseURL != "https://api.deepseek.com" || provider.APIKey != "${"+losslessClawLLMAPIKeyEnv+"}" || len(provider.Models) != 1 || provider.Models[0].ID != "deepseek-v4-flash" {
		t.Fatalf("models.providers[%q] = %#v, want the override provider", losslessClawProviderID, provider)
	}
	entry := configuration.Plugins.Entries[losslessClawPluginID]
	block, ok := entry.Config.(map[string]any)
	wantRef := losslessClawProviderID + "/deepseek-v4-flash"
	if !ok || block["summaryModel"] != wantRef || block["expansionModel"] != wantRef {
		t.Fatalf("plugins.entries[%q].config = %#v, want summaryModel/expansionModel = %q", losslessClawPluginID, entry.Config, wantRef)
	}
	// An override is silently ignored by OpenClaw's host policy layer unless
	// explicitly trusted here (confirmed against the plugin's shipped docs) —
	// merely appearing in models.providers/config.summaryModel is not enough.
	if entry.LLM == nil || !entry.LLM.AllowModelOverride || len(entry.LLM.AllowedModels) != 1 || entry.LLM.AllowedModels[0] != wantRef {
		t.Fatalf("plugins.entries[%q].llm = %#v, want allowModelOverride trusting %q", losslessClawPluginID, entry.LLM, wantRef)
	}
	if entry.Subagent == nil || !entry.Subagent.AllowModelOverride || len(entry.Subagent.AllowedModels) != 1 || entry.Subagent.AllowedModels[0] != wantRef {
		t.Fatalf("plugins.entries[%q].subagent = %#v, want allowModelOverride trusting %q", losslessClawPluginID, entry.Subagent, wantRef)
	}
	if bytes.Contains(content, []byte(losslessClawLLMAPIKeyEnv)) && bytes.Contains(content, []byte("dummy")) {
		t.Fatalf("unexpectedly serialized a literal key value: %s", content)
	}
}

func TestRenderConfigRejectsAMEMAndLosslessClawTogether(t *testing.T) {
	// renderConfig itself doesn't enforce mutual exclusion (that's
	// (*config.HarnessConfig).validate's job), but both claiming pluginSlots
	// wholesale means the second assignment would silently clobber the
	// first's slot entry if both were ever passed enabled=true — document
	// that hazard by asserting today's actual (last-wins) behavior so a
	// future refactor doesn't accidentally think it's safe to call both.
	model := testModel()
	content, err := renderConfig(model, testEndpoint(), false, "", false, false, true, 0, "", "", true, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if _, hasMemory := configuration.Plugins.Slots["memory"]; hasMemory {
		t.Fatalf("plugins.slots = %#v, want amem's \"memory\" slot clobbered by lossless-claw's wholesale assignment", configuration.Plugins.Slots)
	}
	if configuration.Plugins.Slots["contextEngine"] != losslessClawPluginID {
		t.Fatalf("plugins.slots.contextEngine = %q, want %q", configuration.Plugins.Slots["contextEngine"], losslessClawPluginID)
	}
}

func TestRenderConfigEnablesTavilyExtractAlongsideSearXNGSearch(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), true, "", true, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	// Search stays SearXNG: extract must not change tools.web.search.provider.
	if configuration.Tools.Web == nil || configuration.Tools.Web.Search == nil || configuration.Tools.Web.Search.Provider != "searxng" {
		t.Fatalf("tools.web.search = %#v", configuration.Tools.Web)
	}
	searxngEntry, ok := configuration.Plugins.Entries["searxng"]
	searxngBlock, searxngBlockOK := searxngEntry.Config.(map[string]any)
	searxngWebSearch, searxngWebSearchOK := searxngBlock["webSearch"].(map[string]any)
	if !ok || !searxngBlockOK || !searxngWebSearchOK || searxngWebSearch["baseUrl"] != searxngBaseURL {
		t.Fatalf("plugins.entries.searxng = %#v", configuration.Plugins.Entries)
	}
	tavilyEntry, ok := configuration.Plugins.Entries["tavily"]
	if !ok || !tavilyEntry.Enabled || tavilyEntry.Config != nil {
		t.Fatalf("plugins.entries.tavily = %#v", configuration.Plugins.Entries)
	}
	if len(configuration.Plugins.Allow) != 2 || configuration.Plugins.Allow[0] != "searxng" || configuration.Plugins.Allow[1] != "tavily" {
		t.Fatalf("plugins.allow = %#v, want [\"searxng\", \"tavily\"]", configuration.Plugins.Allow)
	}
	alsoAllow := configuration.Tools.Sandbox.Tools.AlsoAllow
	if len(alsoAllow) != 3 || alsoAllow[2] != "tavily_extract" {
		t.Fatalf("tools.sandbox.tools.alsoAllow = %#v, want tavily_extract appended", alsoAllow)
	}
	for _, disallowed := range alsoAllow {
		if disallowed == "tavily_search" {
			t.Fatalf("tavily_search must stay invisible: alsoAllow = %#v", alsoAllow)
		}
	}
}

func TestRenderConfigUsesFirecrawlAsSearchProvider(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), true, "firecrawl", false, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Tools.Web == nil || configuration.Tools.Web.Search == nil || configuration.Tools.Web.Search.Provider != "firecrawl" {
		t.Fatalf("tools.web.search = %#v, want provider firecrawl", configuration.Tools.Web)
	}
	if _, ok := configuration.Plugins.Entries["searxng"]; ok {
		t.Fatalf("plugins.entries = %#v, want no searxng entry when firecrawl is the provider", configuration.Plugins.Entries)
	}
	firecrawlEntry, ok := configuration.Plugins.Entries["firecrawl"]
	if !ok || !firecrawlEntry.Enabled || firecrawlEntry.Config != nil {
		t.Fatalf("plugins.entries.firecrawl = %#v, want enabled with no config/apiKey serialized", configuration.Plugins.Entries)
	}
	if len(configuration.Plugins.Allow) != 1 || configuration.Plugins.Allow[0] != "firecrawl" {
		t.Fatalf("plugins.allow = %#v, want [\"firecrawl\"]", configuration.Plugins.Allow)
	}
	alsoAllow := configuration.Tools.Sandbox.Tools.AlsoAllow
	if len(alsoAllow) != 2 || alsoAllow[0] != "web_search" || alsoAllow[1] != "web_fetch" {
		t.Fatalf("tools.sandbox.tools.alsoAllow = %#v, want unchanged (agent still calls generic web_search)", alsoAllow)
	}
}

func TestRenderConfigCombinesFirecrawlSearchWithTavilyExtract(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), true, "firecrawl", true, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Tools.Web.Search.Provider != "firecrawl" {
		t.Fatalf("tools.web.search.provider = %q, want firecrawl even with extract enabled", configuration.Tools.Web.Search.Provider)
	}
	if len(configuration.Plugins.Allow) != 2 || configuration.Plugins.Allow[0] != "firecrawl" || configuration.Plugins.Allow[1] != "tavily" {
		t.Fatalf("plugins.allow = %#v, want [\"firecrawl\", \"tavily\"]", configuration.Plugins.Allow)
	}
	alsoAllow := configuration.Tools.Sandbox.Tools.AlsoAllow
	if len(alsoAllow) != 3 || alsoAllow[2] != "tavily_extract" {
		t.Fatalf("tools.sandbox.tools.alsoAllow = %#v, want tavily_extract appended", alsoAllow)
	}
}

func TestRenderConfigUsesTavilyAsSearchProvider(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), true, "tavily", false, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Tools.Web == nil || configuration.Tools.Web.Search == nil || configuration.Tools.Web.Search.Provider != "tavily" {
		t.Fatalf("tools.web.search = %#v, want provider tavily", configuration.Tools.Web)
	}
	if _, ok := configuration.Plugins.Entries["searxng"]; ok {
		t.Fatalf("plugins.entries = %#v, want no searxng entry when tavily is the provider", configuration.Plugins.Entries)
	}
	tavilyEntry, ok := configuration.Plugins.Entries["tavily"]
	if !ok || !tavilyEntry.Enabled || tavilyEntry.Config != nil {
		t.Fatalf("plugins.entries.tavily = %#v, want enabled with no config/apiKey serialized", configuration.Plugins.Entries)
	}
	if len(configuration.Plugins.Allow) != 1 || configuration.Plugins.Allow[0] != "tavily" {
		t.Fatalf("plugins.allow = %#v, want [\"tavily\"]", configuration.Plugins.Allow)
	}
	alsoAllow := configuration.Tools.Sandbox.Tools.AlsoAllow
	if len(alsoAllow) != 2 || alsoAllow[0] != "web_search" || alsoAllow[1] != "web_fetch" {
		t.Fatalf("tools.sandbox.tools.alsoAllow = %#v, want unchanged (no tavily_extract without extract enabled)", alsoAllow)
	}
}

func TestRenderConfigTavilySearchAndExtractShareOnePluginEntry(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), true, "tavily", true, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Tools.Web.Search.Provider != "tavily" {
		t.Fatalf("tools.web.search.provider = %q, want tavily", configuration.Tools.Web.Search.Provider)
	}
	// The same "tavily" plugin backs both capabilities here: allow must list
	// it exactly once, not once per capability.
	if len(configuration.Plugins.Allow) != 1 || configuration.Plugins.Allow[0] != "tavily" {
		t.Fatalf("plugins.allow = %#v, want [\"tavily\"] (deduplicated)", configuration.Plugins.Allow)
	}
	alsoAllow := configuration.Tools.Sandbox.Tools.AlsoAllow
	if len(alsoAllow) != 3 || alsoAllow[2] != "tavily_extract" {
		t.Fatalf("tools.sandbox.tools.alsoAllow = %#v, want tavily_extract appended", alsoAllow)
	}
}

func TestRenderConfigIgnoresExtractWhenWebSearchDisabled(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), false, "", true, false, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(content, []byte(`"web"`)) || bytes.Contains(content, []byte(`"plugins"`)) {
		t.Fatalf("extract leaked into rendered config despite web search being disabled: %s", content)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Tools.Web != nil || configuration.Plugins != nil || configuration.Tools.Sandbox != nil {
		t.Fatalf("configuration = %#v", configuration)
	}
}

func TestRenderConfigAllowsSubagentsWhenEnabled(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), false, "", false, true, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(configuration.Tools.Deny, ","); got != "read,write,edit,apply_patch" {
		t.Fatalf("tool deny list = %q, want sessions_spawn/sessions_yield omitted", got)
	}
}

func TestRenderConfigSetsMaxConcurrentSubagentsWhenEnabled(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), false, "", false, true, false, 2, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Agents.Defaults.Subagents == nil || configuration.Agents.Defaults.Subagents.MaxConcurrent != 2 {
		t.Fatalf("agents.defaults.subagents = %#v, want maxConcurrent 2", configuration.Agents.Defaults.Subagents)
	}
}

func TestRenderConfigOmitsSubagentsBlockWhenNoLimitSet(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), false, "", false, true, false, 0, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Agents.Defaults.Subagents != nil {
		t.Fatalf("agents.defaults.subagents = %#v, want omitted", configuration.Agents.Defaults.Subagents)
	}
}

func TestRenderConfigIgnoresMaxConcurrentWhenSubagentsDisabled(t *testing.T) {
	content, err := renderConfig(testModel(), testEndpoint(), false, "", false, false, false, 2, "", "", false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	var configuration openClawConfig
	if err := json.Unmarshal(content, &configuration); err != nil {
		t.Fatal(err)
	}
	if configuration.Agents.Defaults.Subagents != nil {
		t.Fatalf("agents.defaults.subagents = %#v, want omitted when subagents disabled", configuration.Agents.Defaults.Subagents)
	}
}

func TestLauncherUsesFileSecretAndDirectExec(t *testing.T) {
	script := string(launcherScript("ARIES_FAKE_API_KEY", "OPENAI_API_KEY", false, false, false, false, false, false))
	for _, required := range []string{"model_key=$(cat /run/aries/model.key)", "gateway_key=$(cat /run/aries/gateway.key)", "export ARIES_FAKE_API_KEY=\"$model_key\"", "export OPENCLAW_GATEWAY_TOKEN=\"$gateway_key\"", "exec \"$@\""} {
		if !strings.Contains(script, required) {
			t.Fatalf("launcher missing %q: %s", required, script)
		}
	}
	for _, required := range []string{"realtime_key=$(cat /run/aries/realtime.key)", "export OPENAI_API_KEY=\"$realtime_key\"", "unset realtime_key"} {
		if !strings.Contains(script, required) {
			t.Fatalf("launcher missing realtime export %q: %s", required, script)
		}
	}
	agentScript := string(launcherScript("ARIES_FAKE_API_KEY", "", false, false, false, false, false, false))
	if strings.Contains(agentScript, "realtime.key") || strings.Contains(agentScript, "OPENAI_API_KEY") {
		t.Fatalf("agent launcher exports realtime key: %s", agentScript)
	}
}

func TestLauncherExportsTavilyKeyWhenExtractEnabled(t *testing.T) {
	script := string(launcherScript("ARIES_FAKE_API_KEY", "", true, false, false, false, false, false))
	for _, required := range []string{"tavily_key=$(cat /run/aries/tavily.key)", "export TAVILY_API_KEY=\"$tavily_key\"", "unset tavily_key"} {
		if !strings.Contains(script, required) {
			t.Fatalf("launcher missing tavily export %q: %s", required, script)
		}
	}
	disabledScript := string(launcherScript("ARIES_FAKE_API_KEY", "", false, false, false, false, false, false))
	if strings.Contains(disabledScript, "tavily.key") || strings.Contains(disabledScript, "TAVILY_API_KEY") {
		t.Fatalf("launcher exports tavily key when extract is disabled: %s", disabledScript)
	}
}

func TestLauncherExportsFirecrawlKeyWhenEnabled(t *testing.T) {
	script := string(launcherScript("ARIES_FAKE_API_KEY", "", false, true, false, false, false, false))
	for _, required := range []string{"firecrawl_key=$(cat /run/aries/firecrawl.key)", "export FIRECRAWL_API_KEY=\"$firecrawl_key\"", "unset firecrawl_key"} {
		if !strings.Contains(script, required) {
			t.Fatalf("launcher missing firecrawl export %q: %s", required, script)
		}
	}
	disabledScript := string(launcherScript("ARIES_FAKE_API_KEY", "", false, false, false, false, false, false))
	if strings.Contains(disabledScript, "firecrawl.key") || strings.Contains(disabledScript, "FIRECRAWL_API_KEY") {
		t.Fatalf("launcher exports firecrawl key when disabled: %s", disabledScript)
	}
}

func TestLauncherExportsTavilyKeyWhenSearchProviderEnabled(t *testing.T) {
	script := string(launcherScript("ARIES_FAKE_API_KEY", "", false, false, true, false, false, false))
	for _, required := range []string{"tavily_key=$(cat /run/aries/tavily.key)", "export TAVILY_API_KEY=\"$tavily_key\"", "unset tavily_key"} {
		if !strings.Contains(script, required) {
			t.Fatalf("launcher missing tavily export %q: %s", required, script)
		}
	}
	// The export must fire exactly once even if both extract and the tavily
	// search provider are enabled together (same underlying credential/env var).
	combinedScript := string(launcherScript("ARIES_FAKE_API_KEY", "", true, false, true, false, false, false))
	if strings.Count(combinedScript, "export TAVILY_API_KEY=") != 1 {
		t.Fatalf("launcher exported TAVILY_API_KEY more than once: %s", combinedScript)
	}
	disabledScript := string(launcherScript("ARIES_FAKE_API_KEY", "", false, false, false, false, false, false))
	if strings.Contains(disabledScript, "tavily.key") || strings.Contains(disabledScript, "TAVILY_API_KEY") {
		t.Fatalf("launcher exports tavily key when disabled: %s", disabledScript)
	}
}

func TestLauncherExportsLosslessClawLLMKeyOnlyWhenOverridden(t *testing.T) {
	script := string(launcherScript("ARIES_FAKE_API_KEY", "", false, false, false, false, false, true))
	for _, required := range []string{"lcm_llm_key=$(cat /run/aries/lossless-claw-llm.key)", "export LOSSLESS_CLAW_LLM_API_KEY=\"$lcm_llm_key\"", "unset lcm_llm_key"} {
		if !strings.Contains(script, required) {
			t.Fatalf("launcher missing lossless-claw LLM export %q: %s", required, script)
		}
	}
	// Unlike amem, lossless-claw's default (unset override) path never
	// exports the env var at all: renderConfig never registers the extra
	// provider entry that would reference it.
	defaultScript := string(launcherScript("ARIES_FAKE_API_KEY", "", false, false, false, false, false, false))
	if strings.Contains(defaultScript, "lossless-claw-llm.key") || strings.Contains(defaultScript, "LOSSLESS_CLAW_LLM_API_KEY") {
		t.Fatalf("launcher exports lossless-claw LLM key without an override: %s", defaultScript)
	}
}
