#!/usr/bin/env bash
# Run a model's E2E tests once per entry mode as separate coverage runs and upload
# the reports, so per-mode attribution survives (see vllm-project/vllm-omni#5332).
#
# Usage: run_cov_split.sh --model-id <id> [--offline <paths>] [--online <paths>] \
#                         -- <pytest args...>
#
#   --model-id  Names the artifacts: coverage-<id>-{offline,online}.xml. Use the
#               model's directory name under vllm_omni/*/models/.
#   --offline   Pytest path(s) for the offline half. Repeatable.
#   --online    Pytest path(s) for the online half. Repeatable.
#   --          Everything after this is forwarded verbatim to every pytest run,
#               quoting preserved: the job's own -m expression, --run-level, and
#               anything else it needs (e.g. --test-config-file).
#
# At least one of --offline/--online is required, so a job with only one mode runs
# just that half. Total runtime is bounded by the step's timeout_in_minutes.
#
# Exits non-zero if any half or the upload failed.
set -uo pipefail

MODEL_ID=""
OFFLINE=()
ONLINE=()
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --offline)  OFFLINE+=("$2"); shift 2 ;;
        --online)   ONLINE+=("$2"); shift 2 ;;
        --)         shift; PYTEST_ARGS=("$@"); break ;;
        *) echo "run_cov_split.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${MODEL_ID}" ]]; then
    echo "run_cov_split.sh: missing --model-id" >&2
    exit 2
fi

if ((${#OFFLINE[@]} == 0 && ${#ONLINE[@]} == 0)); then
    echo "run_cov_split.sh: need at least one of --offline / --online" >&2
    exit 2
fi

run_half() {
    local mode="$1"
    shift
    echo "--- coverage: ${MODEL_ID} ${mode}"
    pytest -s -v "$@" "${PYTEST_ARGS[@]}" \
        --cov=vllm_omni --cov-report="xml:coverage-${MODEL_ID}-${mode}.xml"
}

# Clear both modes, not just the ones being run: the upload glob matches both, so
# a report left by an earlier run would otherwise be published as if the mode it
# belongs to had run this time.
rm -f "coverage-${MODEL_ID}-offline.xml" "coverage-${MODEL_ID}-online.xml"

EXIT=0
((${#OFFLINE[@]})) && { run_half offline "${OFFLINE[@]}" || EXIT=1; }
((${#ONLINE[@]})) && { run_half online "${ONLINE[@]}" || EXIT=1; }

buildkite-agent artifact upload "coverage-${MODEL_ID}-*.xml" || EXIT=1

exit "${EXIT}"
