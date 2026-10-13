#!/usr/bin/env bash

# Re-run the previously used five-paper M13 batch after the final Controller
# Inspiration routing prompt update.  Keep selection-seed=1301 unchanged for
# comparison with earlier runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_BASE_URL:-}" || -z "${DEEPSEEK_MODEL:-}" ]]; then
  echo "Missing one of DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL" >&2
  exit 2
fi

INPUT_DIR="${M13_INPUT_DIR:-data}"
OUTPUT_DIR="${M13_OUTPUT_DIR:-outputs/m13_real_batch_5_controller_final_rerun}"
BATCH_NAME="${M13_BATCH_NAME:-m13_real_batch_5_controller_final_rerun}"
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
