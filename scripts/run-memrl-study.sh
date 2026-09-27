#!/bin/sh
# Runs the Hermes episodic-memory study on swe-atlas-qa (design and
# hypotheses: docs/benchmarks/swe-atlas-qa.md, "The Hermes episodic-memory
# study"). Each replicate's three arms (control, similarity-only memory,
# MemRL) run side by side: every arm is its own run with its own memory
# store, and each run is sequential inside. Replicates run one after another.
#
#   scripts/run-memrl-study.sh smoke    # smoke4: 4 tasks x 3 arms, check first
#   scripts/run-memrl-study.sh pilot    # pilot30 + shuffle1 + shuffle2: 270 tasks
#   scripts/run-memrl-study.sh epoch2   # repeat-exposure manipulation check: 60 tasks
#
# Requires DEEPSEEK_API_KEY (for example from the ignored DEEPSEEK_API.key) and
# the aries/hermes-memrl image pinned in configs/versions-memrl.json, built by:
#   docker build -f docker/hermes-memrl/Dockerfile -t <that tag> .
# Summarize with: scripts/summarize_memrl_study.py runs/memrl-study/*
set -eu
cd "$(dirname "$0")/.."
go build -o bin/aries ./cmd/aries

replicate() {
	status=0
	./bin/aries "profiles/hermes-sweatlasqa-$1-deepseek.json" & control=$!
	./bin/aries "profiles/hermes-sweatlasqa-$1-memrl-sim-deepseek.json" & similarity=$!
	./bin/aries "profiles/hermes-sweatlasqa-$1-memrl-deepseek.json" & memrl=$!
	wait "$control" || status=1
	wait "$similarity" || status=1
	wait "$memrl" || status=1
	return "$status"
}

case "${1:-}" in
smoke) replicate smoke4 ;;
pilot)
	status=0
	for name in pilot30 pilot30-shuffle1 pilot30-shuffle2; do
		replicate "$name" || status=1
	done
	exit "$status"
	;;
epoch2) ./bin/aries profiles/hermes-sweatlasqa-pilot30-epoch2-memrl-deepseek.json ;;
*)
	echo "usage: $0 smoke|pilot|epoch2" >&2
	exit 2
	;;
esac
