#!/usr/bin/env bash

# Run the technical-soundness prompt diagnostic cohort only.  The existing
# batch-eval runner remains responsible for the complete scientific workflow.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_DIR="${TECHNICAL_SOUNDNESS_INPUT_DIR:-${REPO_ROOT}/data/nlp}"
OUTPUT_DIR="${TECHNICAL_SOUNDNESS_OUTPUT_DIR:-${REPO_ROOT}/outputs/technical_soundness_prompt_v1_final}"
BATCH_NAME="${TECHNICAL_SOUNDNESS_BATCH_NAME:-technical_soundness_prompt_v1_final}"
SEMANTIC_SCHOLAR_CONFIG="${TECHNICAL_SOUNDNESS_SEMANTIC_SCHOLAR_CONFIG:-${REPO_ROOT}/config/semantic_scholar.json}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory does not exist: ${INPUT_DIR}" >&2
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite an existing output path: ${OUTPUT_DIR}" >&2
  echo "Choose another TECHNICAL_SOUNDNESS_OUTPUT_DIR." >&2
  exit 2
fi

if [[ ! -f "${SEMANTIC_SCHOLAR_CONFIG}" ]]; then
  echo "Semantic Scholar config does not exist: ${SEMANTIC_SCHOLAR_CONFIG}" >&2
  exit 2
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_BASE_URL:-}" || -z "${DEEPSEEK_MODEL:-}" ]]; then
  echo "Missing one of DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL" >&2
  echo "Load the configured environment before running this script." >&2
  exit 2
fi

PAPER_IDS=(
  "MinerU_markdown_P017_Gradient-Adaptive_Policy_Optimization_2091868130714607633"
  "MinerU_markdown_P019_MPPO_2091868130714607635"
  "MinerU_markdown_P021_RefreshKV_2091868057381396482"
  "MinerU_markdown_P025_SEED_2091868057381396486"
)

if [[ "${#PAPER_IDS[@]}" -ne 4 ]]; then
  echo "Internal cohort error: expected 4 paper IDs, found ${#PAPER_IDS[@]}" >&2
  exit 2
fi

# batch-eval accepts a directory.  Symlinks make the fixed cohort explicit
# while preserving the original Markdown files and their contents.
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/research-agent-technical-soundness.XXXXXX")"
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

# Match the existing formal runners.  Set TECHNICAL_SOUNDNESS_KEEP_PROXY=1
# when the configured network route requires the proxy variables.
if [[ "${TECHNICAL_SOUNDNESS_KEEP_PROXY:-0}" != "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

echo "Starting technical-soundness prompt cohort: ${#PAPER_IDS[@]} papers"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Order: P017, P019, P021, P025"
echo "Budget: max_steps=12 max_searches=2"

exec env PYTHONPATH=src \
  conda run -n myrag python -m research_agent batch-eval "${STAGING_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-name "${BATCH_NAME}" \
  --max-steps 12 \
  --max-searches 2 \
  --semantic-scholar-config "${SEMANTIC_SCHOLAR_CONFIG}" \
  --allow-real-services
