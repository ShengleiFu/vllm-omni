#!/usr/bin/env bash
# Run a model's E2E tests once per entry mode as separate coverage runs and upload
# the reports, so per-mode attribution survives (see vllm-project/vllm-omni#5332).
#
# Usage: run_cov_split.sh --model-id <id> [--offline <paths>] [--online <paths>] \
#                         --markers <expr> [--pytest-args <args>]
#
#   --model-id     Names the artifacts: coverage-<id>-{offline,online}.xml. Use the
#                  model's directory name under vllm_omni/*/models/.
#   --offline      Pytest path(s) for the offline half. Quote as one argument to
#                  pass more than one: --offline 'a.py b.py'.
#   --online       Pytest path(s) for the online half, same quoting rule.
#   --markers      Passed to pytest -m.
#   --pytest-args  Extra pytest arguments appended to every run, e.g.
#                  '--run-level advanced_model' or '--test-config-file x.json'.
#
# At least one of --offline/--online is required, so a job with only one mode runs
# just that half. Total runtime is bounded by the step's timeout_in_minutes.
#
# Exits non-zero if any half or the upload failed.
set -uo pipefail

MODEL_ID=""
OFFLINE=""
ONLINE=""
MARKERS=""
PYTEST_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)    MODEL_ID="$2"; shift 2 ;;
        --offline)     OFFLINE="$2"; shift 2 ;;
        --online)      ONLINE="$2"; shift 2 ;;
        --markers)     MARKERS="$2"; shift 2 ;;
        --pytest-args) PYTEST_ARGS="$2"; shift 2 ;;
        *) echo "run_cov_split.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

for required in MODEL_ID MARKERS; do
    if [[ -z "${!required}" ]]; then
        flag="$(echo "${required//_/-}" | tr '[:upper:]' '[:lower:]')"
        echo "run_cov_split.sh: missing --${flag}" >&2
        exit 2
    fi
done

if [[ -z "${OFFLINE}" && -z "${ONLINE}" ]]; then
    echo "run_cov_split.sh: need at least one of --offline / --online" >&2
    exit 2
fi

run_half() {
    local mode="$1"
    shift
    echo "--- coverage: ${MODEL_ID} ${mode}"
    rm -f "coverage-${MODEL_ID}-${mode}.xml"
    # shellcheck disable=SC2086  # paths and extra args are intentionally split
    pytest -s -v $* -m "${MARKERS}" ${PYTEST_ARGS} \
        --cov=vllm_omni --cov-report="xml:coverage-${MODEL_ID}-${mode}.xml"
}

EXIT=0
[[ -n "${OFFLINE}" ]] && { run_half offline ${OFFLINE} || EXIT=1; }
[[ -n "${ONLINE}" ]] && { run_half online ${ONLINE} || EXIT=1; }

buildkite-agent artifact upload "coverage-${MODEL_ID}-*.xml" || EXIT=1

exit "${EXIT}"
