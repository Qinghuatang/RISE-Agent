# State Evolution Ablation Pilot (RISE-Full reference vs RISE-Flat)

This pilot tests whether explicit Research-State evolution is useful compared
with a flat, multi-step proposal loop under the same seed papers, backbone,
retrieval tools, and high-level budgets.

The Full side is **read-only in this experiment**.  It references the already
completed and validated runs under `outputs/output_final_80/papers`; it does not
instantiate the Full runner and makes no LLM or retrieval request.  The only
newly executed branch is Flat.

Flat removes the explicit Research-State Evolution machinery, not the
underlying LLM, retrieval, or scientific tools.  Its persistent execution
record contains only a complete current proposal, evidence history, action
history, and remaining budget.  Existing verification and inspiration modules
are used only through short-lived adapter objects and are not serialized as
Flat state.

## Frozen pilot seeds

| Direction | Seed | Historical trajectory variety |
| --- | --- | --- |
| S1 Reasoning & Test-Time Computation | P004 | canonical shortest path |
| S2 Agentic Planning & Tool Use | P010 | inspiration + targeted revision |
| S3 Retrieval & Knowledge-Augmented Generation | P015 | canonical shortest path |
| S4 Post-Training & Preference Optimization | P019 | canonical shortest path |
| S5 Efficient Inference & Model Compression | P021 | targeted revision |
| S6 Multilingual & Cross-Lingual Adaptation | P026 | canonical shortest path |
| S7 Safety Alignment & Jailbreak Robustness | P031 | targeted revision |
| S8 Information Extraction & Structured NLP | P040 | inspiration + targeted revision |

The exact filenames are frozen in `pilot8_config.py`.

## Run

From the repository root, use the `myrag` environment.  A real Flat run
requires an explicit opt-in flag because it can use DeepSeek, Semantic Scholar
and arXiv through the existing adapters.

```bash
cd /home/tmh/research-agent
source ~/.bashrc
PYTHONPATH=src:. conda run -n myrag python \
  experiments/state_evolution_ablation/run_pilot8.py --dry-run
```

Run all eight Flat branches (and save read-only Full references):

```bash
PYTHONPATH=src:. conda run -n myrag python \
  experiments/state_evolution_ablation/run_pilot8.py \
  --all --system both --workers 8 --allow-real-services
```

Run one Flat seed first:

```bash
PYTHONPATH=src:. conda run -n myrag python \
  experiments/state_evolution_ablation/run_pilot8.py \
  --seed P004 --system flat --allow-real-services
```

Completed branches are skipped on a later invocation.  Use `--overwrite` only
when deliberately rerunning an existing branch.  A changed output directory
can be supplied with `--output-dir`; the historical Full source can be changed
with `--full-output-dir`.

## Validate

Validation is offline and does not call any service:

```bash
PYTHONPATH=src:. conda run -n myrag python \
  experiments/state_evolution_ablation/validate_pilot.py
```

The report is written to
`outputs/state_evolution_ablation_pilot8/validation_report.json`.
