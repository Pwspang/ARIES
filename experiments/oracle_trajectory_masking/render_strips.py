"""Generate a browsable HTML page of redaction-strip diagrams for every stable turn's minimal
set -- the same visual language used in the reports, but exhaustive rather than hand-picked.

Usage:
  python render_strips.py \
    --source aries:/home/ws/ARIES/runs/20260812T041739.765784256Z-openclaw-drb-subset10-baseline-sglang:"results/run-final-consolidated/*/turns.jsonl" \
    --source fanout:/home/ws/ARIES/runs/eviction_fanout_deepresearch_bench_openclaw_subset20_baseline_20260729_175953:"results/eviction-fanout-run-*/*/turns.jsonl" \
    --out strips_atlas.html

Each --source is "<kind>:<base_dir>:<results_glob>", kind is "aries" or "fanout" (the two
telemetry directory layouts this repo's extractor supports). Repeat --source for multiple
datasets in one page; each gets its own section.
"""
import argparse
import glob
import html
import json
import os

from context_chunks import unit_ids, unit_label
from trajectory_extractor import extract_turns

PAGE_HEAD = """<title>{title}</title>
<style>
  :root {{
    --paper: #eef0ee; --paper-raised: #f7f8f6; --ink: #191b1d; --muted: #5c6360; --faint: #8b918d;
    --line: #d3d7d2; --redact: #16181a; --accent: #b9791f; --accent-ink: #6b4a14;
    --accent-soft: #f1e0bd; --accent-soft-line: #dcbb7c;
    --serif: "Source Serif 4", Georgia, serif; --sans: "IBM Plex Sans", system-ui, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --paper: #17191b; --paper-raised: #1e2123; --ink: #e9e7e2; --muted: #a3a9a5; --faint: #6b716d;
      --line: #33383a; --redact: #302f34; --accent: #e0a848; --accent-ink: #f4d999;
      --accent-soft: #3a3220; --accent-soft-line: #6b5628;
    }}
  }}
  :root[data-theme="dark"] {{
    --paper: #17191b; --paper-raised: #1e2123; --ink: #e9e7e2; --muted: #a3a9a5; --faint: #6b716d;
    --line: #33383a; --redact: #302f34; --accent: #e0a848; --accent-ink: #f4d999;
    --accent-soft: #3a3220; --accent-soft-line: #6b5628;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.5; }}
  .page {{ max-width: 1000px; margin: 0 auto; padding: 3rem 1.5rem 5rem; }}
  h1 {{ font-family: var(--serif); font-size: 2rem; margin: 0 0 0.5rem; }}
  .dek {{ color: var(--muted); max-width: 70ch; margin-bottom: 1.5rem; }}
  h2.dataset {{ font-family: var(--serif); font-size: 1.4rem; margin: 3rem 0 1.2rem; border-bottom: 1px solid var(--line); padding-bottom: 0.5rem; }}
  h3.task {{ font-family: var(--mono); font-size: 0.95rem; color: var(--accent-ink); margin: 2rem 0 0.8rem; }}
  .legend-row {{ display: flex; gap: 1.4rem; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; color: var(--muted); }}
  .swatch {{ width: 18px; height: 13px; border-radius: 2px; flex: none; }}
  .swatch.kept {{ background: var(--accent-soft); border: 1.5px solid var(--accent-soft-line); }}
  .swatch.masked {{ background: var(--redact); }}
  .strip-card {{ border: 1px solid var(--line); border-radius: 3px; background: var(--paper-raised); padding: 0.8rem 1rem; margin-bottom: 0.7rem; }}
  .strip-head {{ display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.78rem; color: var(--muted); margin-bottom: 0.5rem; }}
  .strip-head b {{ color: var(--ink); }}
  .strip {{ display: flex; gap: 2px; height: 30px; }}
  .unit {{ flex: 1; min-width: 4px; border-radius: 2px; display: flex; align-items: center; justify-content: center; font-family: var(--mono); font-size: 0.6rem; color: var(--accent-ink); overflow: hidden; white-space: nowrap; }}
  .unit.kept {{ background: var(--accent-soft); border: 1.5px solid var(--accent-soft-line); }}
  .unit.masked {{ background: var(--redact); }}
</style>
<div class="page">
  <h1>{title}</h1>
  <p class="dek">{dek}</p>
  <div class="legend-row">
    <div class="legend-item"><span class="swatch kept"></span>kept &mdash; in the minimal set</div>
    <div class="legend-item"><span class="swatch masked"></span>masked &mdash; dropped, decision unchanged</div>
  </div>
"""

PAGE_TAIL = "</div>"


def short_label(label):
    if label == "system_prompt":
        return "sys"
    if label.startswith("compaction@"):
        return "comp"
    if label.startswith("call@"):
        n = label.split("(")[1].split(" ")[0] if "(" in label else ""
        return f"call {n}" if n else "call"
    if label.startswith("result@"):
        n = label.split("(")[1].split(" ")[0] if "(" in label else ""
        return f"out {n}" if n else "out"
    if label.startswith("user@"):
        return "task"
    return label[:6]


def sessions_dir_aries(base, task_id):
    return os.path.join(base, task_id, "harness", "telemetry")


def sessions_dir_fanout(base, task_id, _cache={}):
    key = base
    if key not in _cache:
        mapping = {}
        for manifest_path in glob.glob(f"{base}/cells/*/runs/*/*/manifest.json"):
            m = json.load(open(manifest_path))
            mapping[m["instance_id"]] = os.path.join(os.path.dirname(manifest_path), "conversation", "sessions")
        _cache[key] = mapping
    return _cache[key].get(task_id)


RESOLVERS = {"aries": sessions_dir_aries, "fanout": sessions_dir_fanout}


def render_turn(task_id, r, turn):
    ids = unit_ids(turn)
    minimal = set(r["minimal_unit_labels"])
    cells = []
    for u in ids:
        label = unit_label(turn, u)
        kept = label in minimal
        cls = "kept" if kept else "masked"
        text = html.escape(short_label(label)) if kept else ""
        cells.append(f'<div class="unit {cls}">{text}</div>')
    ratio = r["minimal_units"] / r["total_units"] if r["total_units"] else 0
    return (
        f'<div class="strip-card">'
        f'<div class="strip-head"><span><b>turn {r["turn_index"]}</b></span>'
        f'<span>{r["minimal_units"]} / {r["total_units"]} kept ({ratio:.0%})</span></div>'
        f'<div class="strip">{"".join(cells)}</div>'
        f"</div>"
    )


def render_dataset(kind, base, results_glob):
    resolver = RESOLVERS[kind]
    parts = [f'<h2 class="dataset">{html.escape(kind)}</h2>']
    seen = set()
    n_turns = 0
    for f in sorted(glob.glob(results_glob)):
        task_id = f.split("/")[-2]
        if task_id in seen:
            continue
        recs = [json.loads(line) for line in open(f)]
        if not recs:
            continue
        seen.add(task_id)
        sess_dir = resolver(base, task_id)
        if not sess_dir:
            continue
        turns = extract_turns(sess_dir)
        turns_by_idx = {t.index: t for t in turns}
        stable = [r for r in recs if r.get("reproducible")]
        if not stable:
            continue
        stable.sort(key=lambda r: r["minimal_units"] / r["total_units"] if r["total_units"] else 0)
        parts.append(f'<h3 class="task">{html.escape(task_id)} &mdash; {len(stable)} stable turn(s)</h3>')
        for r in stable:
            t = turns_by_idx[r["turn_index"]]
            parts.append(render_turn(task_id, r, t))
            n_turns += 1
    return "".join(parts), len(seen), n_turns


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help='"<kind>:<base_dir>:<results_glob>", repeatable')
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Minimal Set Atlas")
    args = ap.parse_args()

    body_parts = []
    total_tasks = total_turns = 0
    for src in args.source:
        kind, base, results_glob = src.split(":", 2)
        section, n_tasks, n_turns = render_dataset(kind, base, results_glob)
        body_parts.append(section)
        total_tasks += n_tasks
        total_turns += n_turns

    dek = f"Every stable turn's minimal context set, rendered as a redaction strip in chronological order. {total_turns} turns across {total_tasks} tasks."
    page = PAGE_HEAD.format(title=args.title, dek=dek) + "".join(body_parts) + PAGE_TAIL
    with open(args.out, "w") as f:
        f.write(page)
    print(f"wrote {total_turns} turn diagrams across {total_tasks} tasks to {args.out}")
