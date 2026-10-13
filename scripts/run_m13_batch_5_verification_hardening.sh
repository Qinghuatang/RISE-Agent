#!/usr/bin/env bash

# Run the same five development papers with the current Verification and
# structured-output hardening.  selection-seed=1301 keeps the paper set
# comparable with the earlier five-paper experiments.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_BASE_URL:-}" || -z "${DEEPSEEK_MODEL:-}" ]]; then
  echo "Missing one of DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL" >&2
  exit 2
fi

# Proxy use is opt-in because the previous five-paper run was affected by an
# inherited proxy/DNS route.  Set M13_KEEP_PROXY=1 when proxy use is required.
if [[ "${M13_KEEP_PROXY:-0}" != "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

INPUT_DIR="${M13_INPUT_DIR:-data}"
OUTPUT_DIR="${M13_OUTPUT_DIR:-outputs/m13_real_batch_5_verification_hardening}"
BATCH_NAME="${M13_BATCH_NAME:-m13_real_batch_5_verification_hardening}"
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
