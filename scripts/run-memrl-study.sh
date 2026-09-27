#!/bin/sh
# Runs the Hermes episodic-memory study on swe-atlas-qa (design and
# hypotheses: docs/benchmarks/swe-atlas-qa.md, "The Hermes episodic-memory
# study").
#
#   train: control (no memory) runs the train split once; MemRL trains on it
#          in mini-batches (harness.memrl.batch_size) over several epochs.
#   test:  MemRL's trained store is copied to runs/memrl-study/stores/
#          <study>-memrl. Control and frozen MemRL then run the held-out test
#          split, and the replay run reruns the train split with frozen MemRL.
#
#   scripts/run-memrl-study.sh smoke [train|test]   # 4 train tasks x 2 epochs, batch 2; 2 test tasks
#   scripts/run-memrl-study.sh full [train|test]    # 30 train tasks x 2 epochs, batch 15; 30 test tasks
#
# Every run is already concurrent (execution.concurrency), so by default runs
# go one after another rather than multiplying the load on the one SGLang
# server. PARALLEL=1 runs a phase's runs side by side when it has capacity.
#
# Without a phase both run, train then test. Requires SGLANG_API_KEY and
# DEEPSEEK_API_KEY (the judge), for example from .env, a reachable SGLang
# server per the profiles' model.base_url, and the aries/hermes-memrl image
# pinned in configs/versions-memrl.json, built by:
#   docker build -f docker/hermes-memrl/Dockerfile -t <that tag> .
# Summarize with: scripts/summarize_memrl_study.py runs/memrl-study/*
set -eu
cd "$(dirname "$0")/.."

case "${1:-}" in
smoke) study=memrl-smoke ;;
full) study=memrl ;;
*)
	echo "usage: $0 smoke|full [train|test]" >&2
	exit 2
	;;
esac
phase=${2:-all}
profile() { echo "profiles/hermes-sweatlasqa-$study-$1-sglang.json"; }

# run_each runs one ARIES run per profile, in order or with PARALLEL=1 side
# by side, and fails if any did.
run_each() {
	status=0
	if [ "${PARALLEL:-0}" = 1 ]; then
		pids=""
		for name in "$@"; do
			./bin/aries "$(profile "$name")" &
			pids="$pids $!"
		done
		for pid in $pids; do
			wait "$pid" || status=1
		done
		return "$status"
	fi
	for name in "$@"; do
		./bin/aries "$(profile "$name")" || status=1
	done
	return "$status"
}

# freeze copies the latest training run's store for an arm to where its test
# profile reads it, replacing any store from an earlier training.
# A store with sessions still queued missed a batch update and is refused.
freeze() {
	run=$(ls -d runs/memrl-study/*-"hermes-sweatlasqa-$study-train-$1-sglang" 2>/dev/null | tail -n 1)
	if [ -z "$run" ] || [ ! -f "$run/memrl/memrl.db" ]; then
		echo "no trained $1 store for $study; run the train phase first" >&2
		return 1
	fi
	if [ -n "$(ls -A "$run/memrl/pending" 2>/dev/null)" ]; then
		echo "$run/memrl/pending still holds sessions: a batch update did not finish" >&2
		return 1
	fi
	store="runs/memrl-study/stores/$study-$1"
	rm -rf "$store"
	mkdir -p "$store"
	chmod 700 "$store"
	cp -p "$run/memrl/memrl.db" "$store/memrl.db"
	echo "froze $run/memrl/memrl.db as $store"
}

go build -o bin/aries ./cmd/aries
if [ "$phase" = all ] || [ "$phase" = train ]; then
	run_each train-control train-memrl
fi
if [ "$phase" = all ] || [ "$phase" = test ]; then
	freeze memrl
	run_each test-control test-memrl replay-memrl
fi
