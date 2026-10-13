#!/usr/bin/env bash

# Run the same five-paper M13 development regression used by the previous
# Controller/Inspiration iterations, with the current workflow-correctness
# fixes.  The fixed selection seed keeps the paper set comparable.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_BASE_URL:-}" || -z "${DEEPSEEK_MODEL:-}" ]]; then
  echo "Missing one of DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL" >&2
  exit 2
fi

# The previous regression was affected by an inherited proxy/DNS route.
# Keep proxy use opt-in for this run; set M13_KEEP_PROXY=1 to preserve the
# current proxy environment intentionally.
if [[ "${M13_KEEP_PROXY:-0}" != "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

INPUT_DIR="${M13_INPUT_DIR:-data}"
OUTPUT_DIR="${M13_OUTPUT_DIR:-outputs/m13_real_batch_5_correctness_hardening}"
BATCH_NAME="${M13_BATCH_NAME:-m13_real_batch_5_correctness_hardening}"
MAX_STEPS="${M13_MAX_STEPS:-12}"
MAX_SEARCHES="${M13_MAX_SEARCHES:-2}"
export DEEPSEEK_TIMEOUT_SECONDS="${DEEPSEEK_TIMEOUT_SECONDS:-180}"

exec env PYTHONPATH=src \
  conda run -n myrag python -m research_agent batch-eval "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-name "${BATCH_NAME}" \
  --max-papers 5 \
  --selection-seed 1301 \
  --max-steps "${MAX_STEPS}" \
  --max-searches "${MAX_SEARCHES}" \
  --semantic-scholar-config config/semantic_scholar.json \
  --allow-real-services
