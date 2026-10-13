#!/usr/bin/env bash

# M13 real evaluation pilot: five deterministic random papers, serially.
# This script only invokes the existing real single-paper pilot through the
# batch runner. It never prints or writes API keys.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_BASE_URL:-}" || -z "${DEEPSEEK_MODEL:-}" ]]; then
  echo "Missing one of DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL" >&2
  exit 2
fi

# Keep this output directory fresh. Set M13_OUTPUT_DIR to choose another one;
# the runner refuses to overwrite an existing paper trajectory by default.
OUTPUT_DIR="${M13_OUTPUT_DIR:-outputs/m13_real_batch_5_serial}"
BATCH_NAME="${M13_BATCH_NAME:-m13_real_batch_5_serial}"

# The timeout can be overridden by the caller; 180 seconds matches the
# currently approved real-pilot setting.
export DEEPSEEK_TIMEOUT_SECONDS="${DEEPSEEK_TIMEOUT_SECONDS:-180}"

exec env PYTHONPATH=src \
  conda run -n myrag python -m research_agent batch-eval data \
  --output-dir "${OUTPUT_DIR}" \
  --batch-name "${BATCH_NAME}" \
  --max-papers 5 \
  --selection-seed 1301 \
  --max-steps 12 \
  --max-searches 2 \
  --semantic-scholar-config config/semantic_scholar.json \
  --allow-real-services
