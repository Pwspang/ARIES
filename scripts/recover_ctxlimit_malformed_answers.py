#!/usr/bin/env python3
"""Recover and (re-)judge the ctxlimit-amem-repo tasks whose agent produced
a real, complete answer in its own chat response but never actually wrote
it to /logs/agent/answer.txt -- so the harness saw "task succeeded" but the
evaluator saw no answer file and scored 0 without ever calling the judge
(see pkg/benchmark/sweatlas/evaluate.go's scoreNoAnswer path).

Two ways this showed up, both found by grepping run-result.json's
harness.final_response for these 6 no-eval-json ctxlimit-amem-repo tasks:
  - the model wrote its answer as narrated text describing a bash heredoc
    ("```bash\\ncat <<'ANSWER_EOF' > /logs/agent/answer.txt\\n<<FINAL_ANSWER>>...")
    instead of actually invoking a tool to run it
  - the model answered in plain chat prose and just never wrote the file at
    all, despite (sometimes) claiming "The answer file has been written"

One of the six (task-...a6-003 in the ctxlimit-shuffle... 101452 run) has no
recoverable text at all (its last assistant turn was a bare tool call, and
its gateway.log shows sustained critical memory pressure throughout) -- that
one is left as a genuine failure, not "fixed".

For the other five, this script:
  1. pulls the LAST non-empty assistant text turn out of the task's own
     telemetry transcript (not run-result.json's final_response, which can
     be truncated) as the recovered answer,
  2. strips the model's own `<<FINAL_ANSWER>>` tag and, for the
     heredoc-narrated case, the trailing heredoc terminator/fence junk,
  3. writes it to evaluation/answer.txt in the same run directory the other
     (real) answers live in,
  4. runs the task's own tests/evaluate_answer.py (unmodified upstream
     grading script -- same judge every other task in this study was
     scored by) against it, writing reward.txt/evaluation_results.json in
     place so this task now has real files like every other task.

Requires DEEPSEEK_API_KEY (loaded from .env if not already in the
environment) -- the same judge (deepseek-v4-flash @ api.deepseek.com) used
for the rest of this study (see profiles/*.json's "judge" block).
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (run_dir, task_dir_name) for the 6 no-eval-json ctxlimit-amem-repo tasks
# whose harness.final_response wasn't a "Context overflow" or a
# gateway-stall abort -- see analyze_ctxlimit_degradation.py's investigation.
CANDIDATES = [
    ("runs/ctxlimit/20260914T071929.732959741Z-openclaw-sweatlasqa-pilot30-ctxlimit-amem-repo-sglang",
     "task-6905333b74f22949d97ba9a8-005"),
    ("runs/ctxlimit/20260914T101452.147765009Z-openclaw-sweatlasqa-pilot30-ctxlimit-amem-repo-sglang",
     "task-6905333b74f22949d97ba9a6-003"),  # expected: no recoverable text, skipped
    ("runs/ctxlimit/20260914T101452.147765009Z-openclaw-sweatlasqa-pilot30-ctxlimit-amem-repo-sglang",
     "task-6905333b74f22949d97ba9a8-005"),
    ("runs/ctxlimit/20260914T110808.708508012Z-openclaw-sweatlasqa-pilot30-ctxlimit-shuffle1-amem-repo-sglang",
     "task-6905333b74f22949d97ba9d9-021"),
    ("runs/ctxlimit/20260914T110808.708508012Z-openclaw-sweatlasqa-pilot30-ctxlimit-shuffle1-amem-repo-sglang",
     "task-6905333b74f22949d97ba9d0-027"),
    ("runs/ctxlimit/20260914T110824.178452805Z-openclaw-sweatlasqa-pilot30-ctxlimit-shuffle2-amem-repo-sglang",
     "task-6905333b74f22949d97ba9d0-026"),
]

FINAL_ANSWER_TAG = "<<FINAL_ANSWER>>"
BARE_TASK_ID_RE = re.compile(r"^(task-[0-9a-f]+)-\d+$")


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def last_assistant_text(task_dir: Path) -> str | None:
    tel_dir = task_dir / "harness-turn-01" / "telemetry"
    candidates = [p for p in tel_dir.glob("*.jsonl") if ".trajectory." not in p.name]
    if not candidates:
        return None
    texts = []
    for line in candidates[0].read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("type") != "message" or r.get("message", {}).get("role") != "assistant":
            continue
        content = r["message"].get("content")
        if isinstance(content, list):
            for c in content:
                if c.get("type") == "text" and c.get("text", "").strip():
                    texts.append(c["text"])
    return texts[-1] if texts else None


def recover_answer(raw_text: str) -> str | None:
    """Extract the actual answer content, tolerating the heredoc-narration
    failure mode (model describes writing a heredoc instead of using a real
    tool call) as well as plain undelivered prose."""
    text = raw_text.strip()
    if FINAL_ANSWER_TAG in text:
        text = text.split(FINAL_ANSWER_TAG, 1)[1]
        # heredoc-narrated case: trim the terminator line and anything the
        # model appended after it (closing code fence, stray tags, ...).
        for terminator in ("\nANSWER_EOF", "\nEOF"):
            idx = text.find(terminator)
            if idx != -1:
                text = text[:idx]
    text = text.strip()
    return text or None


def find_tests_dir(task_id: str) -> Path:
    m = BARE_TASK_ID_RE.match(task_id)
    bare_id = m.group(1) if m else task_id
    return REPO_ROOT / ".cache" / "swe-atlas-qa" / "data" / "qa" / bare_id / "tests"


def load_evaluator_module(evaluate_answer_py: Path):
    spec = importlib.util.spec_from_file_location("recovered_evaluate_answer", evaluate_answer_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_evaluation(tests_dir: Path, answer_path: Path, out_dir: Path) -> dict:
    mod = load_evaluator_module(tests_dir / "evaluate_answer.py")
    mod.ANSWER_PATH = str(answer_path)
    mod.RUBRICS_PATH = str(tests_dir / "rubrics.json")
    mod.PROMPT_PATH = str(tests_dir / "prompt.txt")
    mod.SYSTEM_PROMPT_PATH = str(tests_dir / "system_prompt.txt")
    mod.USER_PROMPT_TEMPLATE_PATH = str(tests_dir / "user_prompt_template.txt")
    mod.REWARD_PATH = str(out_dir / "reward.txt")
    mod.RESULTS_PATH = str(out_dir / "evaluation_results.json")
    mod.main()
    return json.loads((out_dir / "evaluation_results.json").read_text())


def main():
    load_env_file(REPO_ROOT / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set (checked environment and .env)")
    os.environ.setdefault("EVAL_API_KEY", os.environ["DEEPSEEK_API_KEY"])
    os.environ.setdefault("EVAL_BASE_URL", "https://api.deepseek.com")
    os.environ.setdefault("EVAL_MODEL", "deepseek-v4-flash")

    for run_rel, task_id in CANDIDATES:
        run_dir = REPO_ROOT / run_rel
        task_dir = run_dir / task_id
        out_dir = task_dir / "evaluation"
        print(f"\n=== {run_dir.name} / {task_id} ===")

        raw = last_assistant_text(task_dir)
        if raw is None:
            print("  no assistant text found at all -- leaving as a genuine failure")
            continue
        recovered = recover_answer(raw)
        if not recovered:
            print("  nothing recoverable after stripping tags -- leaving as a genuine failure")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        answer_path = out_dir / "answer.txt"
        answer_path.write_text(f"{FINAL_ANSWER_TAG}\n{recovered}\n")
        print(f"  recovered {len(recovered)} chars -> {answer_path}")

        tests_dir = find_tests_dir(task_id)
        if not tests_dir.is_dir():
            print(f"  no tests dir at {tests_dir}, skipping evaluation")
            continue

        results = run_evaluation(tests_dir, answer_path, out_dir)
        print(f"  reward={results['reward']} agg_score={results['agg_score']:.3f} "
              f"pass={results['pass']} ({results['num_passed']}/{results['num_scored']} rubrics)")


if __name__ == "__main__":
    main()
