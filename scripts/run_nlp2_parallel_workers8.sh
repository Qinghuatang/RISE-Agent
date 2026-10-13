#!/usr/bin/env bash

# Run the frozen Ours workflow on the independent 40-paper nlp2 cohort.
# Parallelism is paper-level; each paper remains internally serial.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_DIR="${NLP2_INPUT_DIR:-${REPO_ROOT}/data/nlp2}"
OUTPUT_DIR="${NLP2_OUTPUT_DIR:-${REPO_ROOT}/outputs/ours_parallel_nlp2_workers8}"
BATCH_NAME="${NLP2_BATCH_NAME:-ours_parallel_nlp2_workers8}"
WORKERS="${NLP2_WORKERS:-8}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "NLP2 input directory does not exist: ${INPUT_DIR}" >&2
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
  echo "NLP2_WORKERS must be a positive integer: ${WORKERS}" >&2
  exit 2
fi

echo "Starting Ours nlp2 batch: ${PAPER_COUNT} papers"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Workers: ${WORKERS}"
echo "Order: filename ascending"

# Keep the established real-run timeout unless explicitly overridden.
export DEEPSEEK_TIMEOUT_SECONDS="${DEEPSEEK_TIMEOUT_SECONDS:-180}"

exec env PYTHONPATH=src \
  conda run -n myrag python scripts/run_nlp40_parallel.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --workers "${WORKERS}" \
  --batch-name "${BATCH_NAME}" \
  --semantic-scholar-config config/semantic_scholar.json
