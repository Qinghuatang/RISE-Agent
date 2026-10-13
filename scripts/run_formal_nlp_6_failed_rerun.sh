#!/usr/bin/env bash

# Re-run the six papers that failed in the previous hardening rerun.  The
# scientific workflow remains delegated to the existing batch-eval runner.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_DIR="${FORMAL_NLP_6_INPUT_DIR:-${REPO_ROOT}/data/nlp}"
OUTPUT_DIR="${FORMAL_NLP_6_OUTPUT_DIR:-${REPO_ROOT}/outputs/formal_nlp_6_failed_rerun}"
BATCH_NAME="${FORMAL_NLP_6_BATCH_NAME:-formal_nlp_6_failed_rerun}"
SEMANTIC_SCHOLAR_CONFIG="${FORMAL_NLP_6_SEMANTIC_SCHOLAR_CONFIG:-${REPO_ROOT}/config/semantic_scholar.json}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory does not exist: ${INPUT_DIR}" >&2
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite an existing output path: ${OUTPUT_DIR}" >&2
  echo "Choose another FORMAL_NLP_6_OUTPUT_DIR." >&2
  exit 2
fi

if [[ ! -f "${SEMANTIC_SCHOLAR_CONFIG}" ]]; then
  echo "Semantic Scholar config does not exist: ${SEMANTIC_SCHOLAR_CONFIG}" >&2
  exit 2
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_BASE_URL:-}" || -z "${DEEPSEEK_MODEL:-}" ]]; then
  echo "Missing one of DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL" >&2
  exit 2
fi

# These are the six papers with RetrievalHTTPError (HTTP 401 Invalid token) in
# outputs/formal_nlp_14_hardening_rerun/evaluation_report.json.
PAPER_IDS=(
  "MinerU_markdown_P023_Efficient_One-shot_Compression_2091868057381396483"
  "MinerU_markdown_P028_Pretraining_with_Active_Forgetting_2091868057381396481"
  "MinerU_markdown_P031_Shaping_the_Safety_Boundaries_2091868057381396491"
  "MinerU_markdown_P032_Goal-Conditioned_DPO_2091868057381396492"
  "MinerU_markdown_P033_Layer-Level_Self-Exposure_and_Patch_2091868057381396493"
  "MinerU_markdown_P035_Unraveling_the_Mystery_2091868057381396495"
)

if [[ "${#PAPER_IDS[@]}" -ne 6 ]]; then
  echo "Internal cohort error: expected 6 paper IDs, found ${#PAPER_IDS[@]}" >&2
  exit 2
fi

# BatchEvaluationRunner accepts a directory.  Temporary symlinks make the
# six-paper cohort explicit without copying or modifying data/nlp Markdown.
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/research-agent-formal-nlp-6.XXXXXX")"
cleanup() {
  rm -rf "${STAGING_DIR}"
}
trap cleanup EXIT

for paper_id in "${PAPER_IDS[@]}"; do
  source_path="${INPUT_DIR}/${paper_id}.md"
  if [[ ! -f "${source_path}" ]]; then
    echo "Expected Markdown file is missing: ${source_path}" >&2
    exit 2
  fi
  ln -s "${source_path}" "${STAGING_DIR}/${paper_id}.md"
done

export DEEPSEEK_TIMEOUT_SECONDS="${DEEPSEEK_TIMEOUT_SECONDS:-180}"

# Keep proxy use opt-in.  Set FORMAL_NLP_6_KEEP_PROXY=1 if the configured
# network route requires the proxy variables.
if [[ "${FORMAL_NLP_6_KEEP_PROXY:-0}" != "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

echo "Starting failed-paper rerun: ${#PAPER_IDS[@]} papers"
echo "Input source: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Order: P023, P028, P031, P032, P033, P035"
echo "Budget: max_steps=12 max_searches=2"

set +e
env PYTHONPATH=src \
  conda run -n myrag python -m research_agent batch-eval "${STAGING_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-name "${BATCH_NAME}" \
  --max-steps 12 \
  --max-searches 2 \
  --semantic-scholar-config "${SEMANTIC_SCHOLAR_CONFIG}" \
  --allow-real-services
STATUS=$?
set -e

exit "${STATUS}"
