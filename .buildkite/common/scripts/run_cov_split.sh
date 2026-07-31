#!/usr/bin/env bash
# Run a model's offline and online E2E tests as separate coverage runs and upload
# both reports, so per-mode attribution survives (see vllm-project/vllm-omni#5332).
#
# Usage: run_cov_split.sh --model-id <id> --offline <paths> --online <paths> \
#                         --markers <expr> --run-level <level> [--timeout <dur>]
#
#   --model-id   Names the artifacts: coverage-<id>-{offline,online}.xml. Use the
#                model's directory name under vllm_omni/*/models/.
#   --offline    Pytest path(s) for the offline half. Quote as one argument to
#                pass more than one: --offline 'a.py b.py'.
#   --online     Pytest path(s) for the online half, same quoting rule.
#   --markers    Passed to pytest -m.
#   --run-level  Passed to pytest --run-level.
#   --timeout    Per-half `timeout` duration (default 40m). Each half gets the
#                whole budget: a half must not fail earlier than the un-split job
#                would have. This bounds a hung half while letting the script
#                continue, which is what still gets the artifacts uploaded — so
#                the step's own timeout_in_minutes must sit above 2x this value.
#
# Exits non-zero if either half or the upload failed.
set -uo pipefail

MODEL_ID=""
OFFLINE=""
ONLINE=""
MARKERS=""
RUN_LEVEL=""
TIMEOUT="40m"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)  MODEL_ID="$2"; shift 2 ;;
        --offline)   OFFLINE="$2"; shift 2 ;;
        --online)    ONLINE="$2"; shift 2 ;;
        --markers)   MARKERS="$2"; shift 2 ;;
        --run-level) RUN_LEVEL="$2"; shift 2 ;;
        --timeout)   TIMEOUT="$2"; shift 2 ;;
        *) echo "run_cov_split.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

for required in MODEL_ID OFFLINE ONLINE MARKERS RUN_LEVEL; do
    if [[ -z "${!required}" ]]; then
        flag="$(echo "${required//_/-}" | tr '[:upper:]' '[:lower:]')"
        echo "run_cov_split.sh: missing --${flag}" >&2
        exit 2
    fi
done

rm -f "coverage-${MODEL_ID}-offline.xml" "coverage-${MODEL_ID}-online.xml"

run_half() {
    local mode="$1"
    shift
    echo "--- coverage: ${MODEL_ID} ${mode}"
    # shellcheck disable=SC2086  # paths are intentionally word-split
    timeout "${TIMEOUT}" pytest -s -v $* \
        -m "${MARKERS}" --run-level "${RUN_LEVEL}" \
        --cov=vllm_omni --cov-report="xml:coverage-${MODEL_ID}-${mode}.xml"
}

EXIT=0
run_half offline ${OFFLINE} || EXIT=1
run_half online ${ONLINE} || EXIT=1

buildkite-agent artifact upload "coverage-${MODEL_ID}-*.xml" || EXIT=1

exit "${EXIT}"
