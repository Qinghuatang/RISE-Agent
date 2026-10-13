#!/usr/bin/env bash

# Formal 40-paper Direct LLM baseline.
# This script only performs cohort/preflight checks and delegates execution to
# the existing one-shot Direct LLM runner.  It does not invoke the Ours
# workflow or add any paper-specific behavior.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_DIR="${DIRECT_LLM_INPUT_DIR:-${REPO_ROOT}/data/nlp}"
OUTPUT_DIR="${DIRECT_LLM_OUTPUT_DIR:-${REPO_ROOT}/outputs/direct_llm_formal_40}"
BATCH_NAME="${DIRECT_LLM_BATCH_NAME:-direct_llm_formal_40}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Direct LLM input directory does not exist: ${INPUT_DIR}" >&2
  exit 2
fi

mapfile -t PAPER_FILES < <(
  find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.md' -print | LC_ALL=C sort
)
PAPER_COUNT="${#PAPER_FILES[@]}"
if [[ "${PAPER_COUNT}" -ne 40 ]]; then
  echo "Expected exactly 40 Markdown papers, found ${PAPER_COUNT}: ${INPUT_DIR}" >&2
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite an existing output path: ${OUTPUT_DIR}" >&2
  echo "Choose another DIRECT_LLM_OUTPUT_DIR." >&2
  exit 2
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_BASE_URL:-}" || -z "${DEEPSEEK_MODEL:-}" ]]; then
  echo "Missing one of DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL" >&2
  exit 2
fi

# Keep proxy use opt-in. Set DIRECT_LLM_KEEP_PROXY=1 if the configured route
# requires the proxy variables.
if [[ "${DIRECT_LLM_KEEP_PROXY:-0}" != "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

# The Direct runner's defaults are the frozen one-shot baseline settings:
# temperature=0 and no application-level retry or second generation call.
export DEEPSEEK_TIMEOUT_SECONDS="${DEEPSEEK_TIMEOUT_SECONDS:-180}"

echo "Starting Direct LLM formal batch: ${PAPER_COUNT} papers"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Order: filename ascending"
echo "Mode: one structured LLM call per paper; no retrieval or Ours workflow"

exec conda run -n myrag python scripts/run_direct_llm.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-name "${BATCH_NAME}"
