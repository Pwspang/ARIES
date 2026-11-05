#!/usr/bin/env bash
# CodeRAG (claude-context) for SWE-Atlas QA: offline index build + read-only MCP serving.
# Guide: docs/benchmarks/swe-atlas-qa-coderag.md
#
#   coderag.sh repos                  group QA tasks by image into $STATE/repos.json
#   coderag.sh extract [slug...]      copy /app out of each pinned image into $STATE/repos/<slug>
#   coderag.sh up                     network, Milvus (auth on), RBAC users, image
#   coderag.sh build [slug...]        offline index build; accepted builds write $STATE/build/manifest-<slug>.json
#   coderag.sh serve [slug...]        start read-only SSE MCP servers (one per repo)
#   coderag.sh check [slug...]        tools, search_code, clear_index refused, rows unchanged (from another network)
#   coderag.sh profiles BASE.json     per-repo coderag + baseline ARIES profiles into $STATE/profiles
#   coderag.sh ports                  slug -> host port -> MCP URL
#   coderag.sh down                   stop MCP servers and Milvus (index kept in $STATE/milvus/data)
#
# Environment:
#   CODERAG_STATE          state directory (default: <repo>/.cache/coderag; gitignored)
#   CODERAG_DATASET        SWE-Atlas checkout (default: <repo>/.cache/swe-atlas-qa)
#   CODERAG_EMBEDDING_ENV  embedding endpoint env file shared by build and serve (default: $STATE/embedding.env)
#   CODERAG_HOST_ADDR      address harness containers use to reach the servers (default: first `hostname -I`)
#   CODERAG_TOOLS          space-separated tool allow-list for the served endpoint (default: all four tools)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
STATE="$(mkdir -p "${CODERAG_STATE:-$repo_root/.cache/coderag}" && cd "${CODERAG_STATE:-$repo_root/.cache/coderag}" && pwd)"
DATASET="${CODERAG_DATASET:-$repo_root/.cache/swe-atlas-qa}"
EMBED_ENV="${CODERAG_EMBEDDING_ENV:-$STATE/embedding.env}"
HOST_ADDR="${CODERAG_HOST_ADDR:-$(hostname -I | awk '{print $1}')}"
TOOLS="${CODERAG_TOOLS:-}"

IMAGE=aries-coderag:0.1.15
MILVUS_IMAGE=milvusdb/milvus:v2.6.4
NET=coderag
MILVUS=coderag-milvus
PORT_BASE=8201

repos_json() {
  [ -f "$STATE/repos.json" ] || { echo "$STATE/repos.json missing: run coderag.sh repos" >&2; exit 1; }
  echo "$STATE/repos.json"
}
slugs_all() { python3 -I -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))))' "$(repos_json)"; }
port_of() { echo $((PORT_BASE + $(slugs_all | grep -nx "$1" | cut -d: -f1) - 1)); }
pick() { if [ $# -eq 0 ]; then slugs_all; else printf '%s\n' "$@"; fi; }
image_of() { python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["image"])' "$(repos_json)" "$1"; }

secrets() {
  if [ ! -f "$STATE/secrets.env" ]; then
    (
      umask 077
      echo "MILVUS_ROOT_PASSWORD=$(openssl rand -hex 16)"
      echo "CODERAG_BUILD_PASSWORD=$(openssl rand -hex 16)"
      echo "CODERAG_SERVE_PASSWORD=$(openssl rand -hex 16)"
    ) > "$STATE/secrets.env"
    chmod 600 "$STATE/secrets.env"
  fi
  # shellcheck disable=SC1091
  set -a; . "$STATE/secrets.env"; set +a
}

need_embedding() {
  [ -f "$EMBED_ENV" ] || {
    echo "$EMBED_ENV missing: copy $here/embedding.env.example there and fill in your endpoint" >&2; exit 1; }
}

repos() {
  python3 -I - "$DATASET/data/qa" "$STATE/repos.json" <<'EOF'
import json, pathlib, re, sys, tomllib
root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
groups = {}
for task in sorted(root.glob("task-*/task.toml")):
    image = tomllib.loads(task.read_text())["environment"]["docker_image"]
    groups.setdefault(image, []).append(task.parent.name)
if not groups:
    sys.exit(f"no QA tasks under {root}: run ./bin/aries setup on a SWE-Atlas QA profile first")
repos = {}
for image, tasks in groups.items():
    m = re.search(r"swe_atlas_QnA_(.+?)_(?:[0-9a-f]{32}_qna_)?1\.0\d?$", image)
    slug = re.sub(r"[^a-z0-9]+", "_", (m.group(1) if m else image).lower()).strip("_")
    repos[slug] = {"image": image, "tasks": tasks}
out.write_text(json.dumps(repos, indent=2) + "\n")
for slug, entry in repos.items():
    print(f"{slug:30} {len(entry['tasks']):3} {entry['image']}")
EOF
}

extract() {
  mkdir -p "$STATE/repos"
  for slug in $(pick "$@"); do
    dest="$STATE/repos/$slug"
    if [ -f "$dest.digest" ]; then echo "skip $slug (already extracted)"; continue; fi
    image="$(image_of "$slug")"
    echo "== $slug <- $image"
    docker image inspect "$image" >/dev/null 2>&1 || docker pull -q "$image"
    digest="$(docker image inspect --format '{{index .RepoDigests 0}}' "$image")"
    ctr="$(docker create "$image" true)"
    rm -rf "$dest.tmp" && mkdir -p "$dest.tmp"
    if ! docker cp "$ctr:/app/." "$dest.tmp/"; then docker rm -f "$ctr" >/dev/null; exit 1; fi
    docker rm -f "$ctr" >/dev/null
    rm -rf "$dest" && mv "$dest.tmp" "$dest"
    printf '%s\n' "$digest" > "$dest.digest"
    echo "   $(du -sh "$dest" | cut -f1)  $digest"
  done
}

milvus_up() {
  docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
  if ! docker container inspect "$MILVUS" >/dev/null 2>&1; then
    mkdir -p "$STATE/milvus/data"
    docker run -d --name "$MILVUS" --network "$NET" --restart unless-stopped \
      --security-opt seccomp:unconfined \
      -e ETCD_USE_EMBED=true -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
      -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
      -e COMMON_STORAGETYPE=local -e DEPLOY_MODE=STANDALONE \
      -v "$STATE/milvus/data:/var/lib/milvus" \
      -v "$here/milvus/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml:ro" \
      -v "$here/milvus/user.yaml:/milvus/configs/user.yaml:ro" \
      -p 127.0.0.1:19530:19530 -p 127.0.0.1:9091:9091 \
      "$MILVUS_IMAGE" milvus run standalone >/dev/null
  fi
  docker start "$MILVUS" >/dev/null
  for _ in $(seq 1 90); do
    curl -fsS http://127.0.0.1:9091/healthz >/dev/null 2>&1 && { echo "milvus healthy"; return; }
    sleep 2
  done
  echo "milvus did not become healthy" >&2; docker logs --tail 50 "$MILVUS" >&2; exit 1
}

image() { docker build -q -t "$IMAGE" "$here/image" >/dev/null && echo "built $IMAGE"; }

rbac() {
  docker run --rm --network "$NET" -e MILVUS_ADDRESS="$MILVUS:19530" \
    -e MILVUS_ROOT_PASSWORD -e CODERAG_BUILD_PASSWORD -e CODERAG_SERVE_PASSWORD \
    "$IMAGE" node setup-milvus.mjs
}

build() {
  need_embedding; mkdir -p "$STATE/build"
  for slug in $(pick "$@"); do
    [ -f "$STATE/repos/$slug.digest" ] || { echo "$slug not extracted (run coderag.sh extract $slug)" >&2; exit 1; }
    manifest="$STATE/build/manifest-$slug.json"; log="$STATE/build/build-$slug.log"
    rm -f "$manifest"
    echo "== build $slug"
    if ! docker run --rm --network "$NET" --env-file "$EMBED_ENV" \
      -e MILVUS_ADDRESS="$MILVUS:19530" -e MILVUS_TOKEN="coderag_build:$CODERAG_BUILD_PASSWORD" \
      -e CODE_CHUNKS_COLLECTION_NAME_OVERRIDE="swe_atlas_$slug" \
      -v "$STATE/repos/$slug:/app:ro" -v "$STATE/build:/out" \
      "$IMAGE" node build-index.mjs /app "/out/manifest-$slug.json.pending" > "$log" 2>&1; then
      rm -f "$manifest.pending"
      echo "   REJECTED: see $log" >&2; exit 1
    fi
    python3 -I - "$manifest.pending" "$manifest" "$STATE/repos/$slug.digest" <<'EOF'
import json, os, sys
pending, final, digest = sys.argv[1:]
m = json.load(open(pending))
m["image_digest"] = open(digest).read().strip()
json.dump(m, open(final, "w"), indent=2)
os.remove(pending)
print(f"   accepted ({m['indexed_files']} files, {m['total_chunks']} chunks)")
EOF
  done
}

serve() {
  need_embedding
  for slug in $(pick "$@"); do
    [ -f "$STATE/build/manifest-$slug.json" ] || { echo "no accepted build for $slug" >&2; exit 1; }
    port="$(port_of "$slug")"; name="coderag-serve-$slug"
    docker rm -f "$name" >/dev/null 2>&1 || true
    tool_args=(); for t in $TOOLS; do tool_args+=(--tools "$t"); done
    docker run -d --name "$name" --network "$NET" --restart unless-stopped --env-file "$EMBED_ENV" \
      -e MILVUS_ADDRESS="$MILVUS:19530" -e MILVUS_TOKEN="coderag_serve:$CODERAG_SERVE_PASSWORD" \
      -e CODE_CHUNKS_COLLECTION_NAME_OVERRIDE="swe_atlas_$slug" \
      -e CLAUDE_CONTEXT_BACKGROUND_SYNC=false -e CLAUDE_CONTEXT_TRIGGER_WATCHER=false \
      -v "$STATE/repos/$slug:/app:ro" -p "0.0.0.0:$port:8000" \
      "$IMAGE" supergateway --stdio "node /opt/coderag/serve-mcp.mjs" --port 8000 --healthEndpoint /healthz "${tool_args[@]}" >/dev/null
    echo "$slug -> http://$HOST_ADDR:$port/sse"
  done
}

rows() {  # slug -> current Milvus row count, queried as root
  docker run --rm --network "$NET" -e MILVUS_ADDRESS="$MILVUS:19530" -e TOKEN="root:$MILVUS_ROOT_PASSWORD" \
    -e COLLECTION="$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["collection"])' "$STATE/build/manifest-$1.json")" \
    "$IMAGE" node -e '
      const { MilvusClient } = require("@zilliz/milvus2-sdk-node");
      const c = new MilvusClient({ address: process.env.MILVUS_ADDRESS, token: process.env.TOKEN });
      c.query({ collection_name: process.env.COLLECTION, output_fields: ["count(*)"], consistency_level: "Strong" })
       .then(r => { console.log(r.data[0]["count(*)"]); process.exit(0); });'
}

check() {
  for slug in $(pick "$@"); do
    port="$(port_of "$slug")"
    before="$(rows "$slug")"
    echo "== check $slug (rows $before) via http://$HOST_ADDR:$port/sse"
    for _ in $(seq 1 30); do curl -fsS "http://$HOST_ADDR:$port/healthz" >/dev/null 2>&1 && break; sleep 1; done
    # Default bridge network, not $NET: the same path an ARIES harness container takes.
    docker run --rm "$IMAGE" node mcp-check.mjs "http://$HOST_ADDR:$port/sse" "${CODERAG_CHECK_QUERY:-main entry point}" --try-clear \
      || { echo "   FAILED" >&2; exit 1; }
    after="$(rows "$slug")"
    [ "$before" = "$after" ] || { echo "   rows changed $before -> $after" >&2; exit 1; }
    echo "   ok: rows unchanged ($after)"
  done
}

profiles() {
  base="${1:?usage: coderag.sh profiles BASE_PROFILE.json}"
  mkdir -p "$STATE/profiles"
  python3 -I - "$base" "$HOST_ADDR" "$PORT_BASE" "$(repos_json)" "$STATE/profiles" <<'EOF'
import json, pathlib, sys
base_path, host, port_base = pathlib.Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3])
repos = json.loads(pathlib.Path(sys.argv[4]).read_text())
out = pathlib.Path(sys.argv[5])
base = json.loads(base_path.read_text())
for i, (slug, entry) in enumerate(repos.items()):
    for variant in ("coderag", "baseline"):
        p = json.loads(json.dumps(base))
        p["name"] = f"{base['name']}-{variant}-{slug.replace('_', '-')}"
        # ARIES resolves these against the profile's directory; keep them pointing where the
        # base profile meant. output_dir and benchmark.root stay relative to the repo root.
        for key in ("versions_file", "overrides_file"):
            if p.get(key):
                p[key] = str((base_path.parent / p[key]).resolve())
        runtime_file = (p.get("runtime", {}).get("config") or {}).get("file")
        if runtime_file and not pathlib.Path(runtime_file).is_absolute():
            p["runtime"]["config"]["file"] = str((base_path.parent / runtime_file).resolve())
        p["benchmark"]["tasks"] = entry["tasks"]
        servers = [s for s in p["harness"].get("mcp_servers", []) if s.get("name") != "coderag"]
        if variant == "coderag":
            servers.append({"name": "coderag", "url": f"http://{host}:{port_base + i}/sse",
                            "transport": "sse", "timeout_seconds": 60})
        if servers:
            p["harness"]["mcp_servers"] = servers
        else:
            p["harness"].pop("mcp_servers", None)
        path = out / f"{p['name']}.json"
        path.write_text(json.dumps(p, indent=2) + "\n")
        print(path)
EOF
}

ports() { for slug in $(slugs_all); do echo "$slug $(port_of "$slug") http://$HOST_ADDR:$(port_of "$slug")/sse"; done; }

down() {
  for slug in $(slugs_all); do docker rm -f "coderag-serve-$slug" >/dev/null 2>&1 || true; done
  docker stop "$MILVUS" >/dev/null 2>&1 || true
}

cmd="${1:-}"; shift || true
case "$cmd" in
  repos) repos ;;
  extract) extract "$@" ;;
  up) secrets; milvus_up; image; rbac ;;
  build) secrets; build "$@" ;;
  serve) secrets; serve "$@" ;;
  check) secrets; check "$@" ;;
  profiles) profiles "$@" ;;
  ports) ports ;;
  down) down ;;
  *) sed -n '2,21p' "$0"; exit 2 ;;
esac
