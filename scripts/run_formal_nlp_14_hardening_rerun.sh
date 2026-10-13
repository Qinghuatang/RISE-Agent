#!/usr/bin/env bash

# Re-run the 14 non-committed papers from the previous formal NLP batch after
# infrastructure hardening.  This script only defines the input cohort and
# delegates the scientific workflow to the existing batch-eval runner.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_DIR="${FORMAL_NLP_14_INPUT_DIR:-${REPO_ROOT}/data/nlp}"
OUTPUT_DIR="${FORMAL_NLP_14_OUTPUT_DIR:-${REPO_ROOT}/outputs/formal_nlp_14_hardening_rerun}"
BATCH_NAME="${FORMAL_NLP_14_BATCH_NAME:-formal_nlp_14_hardening_rerun}"
SEMANTIC_SCHOLAR_CONFIG="${FORMAL_NLP_14_SEMANTIC_SCHOLAR_CONFIG:-${REPO_ROOT}/config/semantic_scholar.json}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory does not exist: ${INPUT_DIR}" >&2
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite an existing output path: ${OUTPUT_DIR}" >&2
  echo "Choose another FORMAL_NLP_14_OUTPUT_DIR." >&2
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

# This is the fixed set of non-committed papers in the previous
# outputs/formal_nlp_40/evaluation_report.json: 13 failed papers plus P032,
# which stopped through the unchanged legal budget_exhausted outcome.
PAPER_IDS=(
  "MinerU_markdown_P003_SoftCoT_2091868130714607620"
  "MinerU_markdown_P004_Stepwise_Informativeness_Search_2091868130714607617"
  "MinerU_markdown_P005_Fine-Tuning_on_Diverse_Reasoning_Chains_2091868130714607621"
  "MinerU_markdown_P009_Interactive_and_Expressive_Code-Augmented_Planning_2091868130714607625"
  "MinerU_markdown_P011_RetroLLM_2091868130714607627"
  "MinerU_markdown_P016_Intuitive_Fine-Tuning_2091868130714607632"
  "MinerU_markdown_P018_In-context_DPO_2091868130714607634"
  "MinerU_markdown_P021_RefreshKV_2091868057381396482"
  "MinerU_markdown_P023_Efficient_One-shot_Compression_2091868057381396483"
  "MinerU_markdown_P028_Pretraining_with_Active_Forgetting_2091868057381396481"
  "MinerU_markdown_P031_Shaping_the_Safety_Boundaries_2091868057381396491"
  "MinerU_markdown_P032_Goal-Conditioned_DPO_2091868057381396492"
  "MinerU_markdown_P033_Layer-Level_Self-Exposure_and_Patch_2091868057381396493"
  "MinerU_markdown_P035_Unraveling_the_Mystery_2091868057381396495"
)

if [[ "${#PAPER_IDS[@]}" -ne 14 ]]; then
  echo "Internal cohort error: expected 14 paper IDs, found ${#PAPER_IDS[@]}" >&2
  exit 2
fi

# BatchEvaluationRunner accepts a directory.  Use a temporary directory of
# symlinks so the cohort is explicit while the original Markdown files remain
# untouched.  The batch runner then sorts these names deterministically.
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/research-agent-formal-nlp-14.XXXXXX")"
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

# Keep proxy use opt-in.  Set FORMAL_NLP_14_KEEP_PROXY=1 if the configured
# network route requires the proxy variables.
if [[ "${FORMAL_NLP_14_KEEP_PROXY:-0}" != "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

echo "Starting infrastructure-hardening rerun: ${#PAPER_IDS[@]} papers"
echo "Input source: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Order: P003, P004, P005, P009, P011, P016, P018, P021, P023, P028, P031, P032, P033, P035"
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
