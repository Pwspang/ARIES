"""Thin client for an OpenAI-compatible /chat/completions endpoint (sglang)."""
import json
import os
import urllib.error
import urllib.request


class ModelClient:
    def __init__(self, base_url, model, api_key_env=None, temperature=0, timeout=300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system_prompt, messages, tools):
        """Returns {"tool_calls": [{"name":..., "arguments": dict}, ...], "text": str} or None on error."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
        }
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        # Broad catch by design: this fires many hundreds of calls per run, often against a
        # server under heavy concurrent load (multi-minute requests, several in flight at once).
        # A dropped connection, a malformed response, or a timeout are all just as "this
        # candidate produced no usable action" as an HTTP error status -- any of them should
        # degrade to {"error": ...} so one flaky call can't crash the whole search.
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
            choice = body["choices"][0]["message"]
            tool_calls = []
            for tc in choice.get("tool_calls") or []:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = tc["function"]["arguments"]
                tool_calls.append({"name": tc["function"]["name"], "arguments": args})
            return {"tool_calls": tool_calls, "text": choice.get("content") or ""}
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}
