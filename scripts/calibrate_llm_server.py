#!/usr/bin/env python3
"""Calibrate the model server's prefill and decode cost for critical-path profiling.

Sends single-stream (concurrency 1) completions against an idle server and
records time-to-first-token and inter-token time as a function of prompt
length, plus a repeated-prefix probe that shows whether prefix caching is on.
Output is a JSON calibration consumed by profile_critical_path.py.

Usage:
  scripts/calibrate_llm_server.py --out scripts/out_profile/calibration.json
"""
import argparse, json, random, time, urllib.request

def post(url, body, stream=False):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)

def metric(base, name):
    text = urllib.request.urlopen(base.rsplit("/v1", 1)[0] + "/metrics", timeout=10).read().decode()
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            return float(line.rsplit(" ", 1)[1])
    return None

def make_prompt(base, model, n_tokens, rng):
    words = [f"w{rng.randrange(10**6)}" for _ in range(n_tokens)]
    text = " ".join(words)
    # trim to target length using the server tokenizer
    toks = json.load(post(base.rsplit("/v1", 1)[0] + "/tokenize", {"model": model, "prompt": text}))["tokens"]
    detok = json.load(post(base.rsplit("/v1", 1)[0] + "/detokenize", {"model": model, "tokens": toks[:n_tokens]}))["prompt"]
    return detok

def run(base, model, prompt, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    t0 = time.monotonic(); first = None; stamps = []; usage = None
    with post(base + "/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("text"):
                now = time.monotonic(); stamps.append(now)
                first = first or now
    end = time.monotonic()
    tpot = (stamps[-1] - stamps[0]) / max(1, len(stamps) - 1) if len(stamps) > 1 else None
    return {"ttft_s": first - t0, "total_s": end - t0, "tpot_s": tpot, "chunks": len(stamps), "usage": usage}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://192.168.111.200:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--lengths", default="1000,4000,8000,16000,32000,48000,64000,96000,128000")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    running = metric(a.base_url, "vllm:num_requests_running")
    res = {"base_url": a.base_url, "model": a.model, "server_running_at_start": running,
           "cache_hits_before": metric(a.base_url, "vllm:prefix_cache_hits_total"), "points": [], "repeat": []}
    run(a.base_url, a.model, "warmup", 8)
    for n in [int(x) for x in a.lengths.split(",")]:
        p = make_prompt(a.base_url, a.model, n, rng)
        r = run(a.base_url, a.model, p, 128); r["prompt_tokens"] = n
        res["points"].append(r); print(json.dumps({k: r[k] for k in ("prompt_tokens", "ttft_s", "tpot_s")}), flush=True)
        if n in (16000, 64000):
            r2 = run(a.base_url, a.model, p + " tail", 16); r2["prompt_tokens"] = n
            res["repeat"].append(r2); print("repeat", n, round(r2["ttft_s"], 3), flush=True)
    res["cache_hits_after"] = metric(a.base_url, "vllm:prefix_cache_hits_total")
    res["cache_queries_after"] = metric(a.base_url, "vllm:prefix_cache_queries_total")
    json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
