#!/usr/bin/env bash

# Run the frozen Direct-Neutral one-shot baseline on the independent nlp2 cohort.
# Parallelism is paper-level; each paper makes one logical scientific call.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_DIR="${DIRECT_NEUTRAL_NLP2_INPUT_DIR:-${REPO_ROOT}/data/nlp2}"
OUTPUT_DIR="${DIRECT_NEUTRAL_NLP2_OUTPUT_DIR:-${REPO_ROOT}/outputs/direct_llm_neutral_nlp2_workers8}"
BATCH_NAME="${DIRECT_NEUTRAL_NLP2_BATCH_NAME:-direct_llm_neutral_nlp2_workers8}"
WORKERS="${DIRECT_NEUTRAL_NLP2_WORKERS:-8}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Direct-Neutral nlp2 input directory does not exist: ${INPUT_DIR}" >&2
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
  echo "Refusing to overwrite existing output path: ${OUTPUT_DIR}" >&2
  exit 2
fi

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DIRECT_NEUTRAL_NLP2_WORKERS must be a positive integer: ${WORKERS}" >&2
  exit 2
fi

# Keep the established real-run timeout unless explicitly overridden.
export DEEPSEEK_TIMEOUT_SECONDS="${DEEPSEEK_TIMEOUT_SECONDS:-180}"

echo "Starting Direct-Neutral nlp2 batch: ${PAPER_COUNT} papers"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Workers: ${WORKERS}"
echo "Order: filename ascending"

exec env PYTHONPATH=src \
  conda run -n myrag python scripts/run_direct_llm_neutral_nlp40.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --workers "${WORKERS}" \
  --batch-name "${BATCH_NAME}"
