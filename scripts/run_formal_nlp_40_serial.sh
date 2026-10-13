#!/usr/bin/env bash

# Formal 40-paper NLP evaluation.  This script only performs preflight checks
# and delegates every paper to the existing M13 batch runner / M12 pilot.
# There is no paper-specific workflow logic here.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_DIR="${FORMAL_NLP_INPUT_DIR:-${REPO_ROOT}/data/nlp}"
OUTPUT_DIR="${FORMAL_NLP_OUTPUT_DIR:-${REPO_ROOT}/outputs/formal_nlp_40}"
BATCH_NAME="${FORMAL_NLP_BATCH_NAME:-formal_nlp_40}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Formal NLP input directory does not exist: ${INPUT_DIR}" >&2
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

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to use an existing output path; choose a new FORMAL_NLP_OUTPUT_DIR: ${OUTPUT_DIR}" >&2
  exit 2
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_BASE_URL:-}" || -z "${DEEPSEEK_MODEL:-}" ]]; then
  echo "Missing one of DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL" >&2
  exit 2
fi

# Keep proxy use opt-in, matching the existing real-paper runners.  Set
# FORMAL_NLP_KEEP_PROXY=1 when the configured network requires a proxy.
if [[ "${FORMAL_NLP_KEEP_PROXY:-0}" != "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

# Frozen formal workflow budget.  These are intentionally not environment
# overrides: the 40 papers must use the same production configuration.
export DEEPSEEK_TIMEOUT_SECONDS="${DEEPSEEK_TIMEOUT_SECONDS:-180}"

echo "Starting formal NLP batch: ${PAPER_COUNT} papers"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Order: filename ascending"

# Omitting --max-papers is deliberate: the existing BatchEvaluationRunner
# discovers all immediate .md files in sorted order and isolates each paper in
# OUTPUT_DIR/<paper_id>.  It catches ordinary per-paper exceptions, records
# batch_error.json, and continues; KeyboardInterrupt is not swallowed.
exec env PYTHONPATH=src \
  conda run -n myrag python -m research_agent batch-eval "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-name "${BATCH_NAME}" \
  --max-steps 12 \
  --max-searches 2 \
  --semantic-scholar-config config/semantic_scholar.json \
  --allow-real-services
