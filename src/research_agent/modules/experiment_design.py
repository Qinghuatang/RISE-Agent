"""M8 Experiment Design.

The module turns a verified Current Solution into a small, claim-covered set
of canonical Experiments.  LLM output is only a closed internal draft;
identity, status, coverage, replacement, and State mutation are deterministic.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from ..errors import ModuleContractError, PreconditionError
from ..llm.client import (
    LLMCallMetadata,
    LLMClient,
    LLMMessage,
    LLMRequest,
    StructuredLLMResponse,
)
from ..memory.store import ResearchMemoryStore
from ..models.gap import CandidateGap
from ..models.memory import VerificationRecord
from ..models.experiment_design import (
    ClaimIdentifiabilityDraft,
    ExperimentDesignDraft,
    ExperimentProposalDraft,
    KeyClaimDraft,
)
from ..models.seed import SeedPaperContext
from ..models.state import CurrentSolution, Experiment, ResearchState
from ..models.verification import (
    GapVerificationResult,
    SolutionVerificationResult,
    VerificationFocus,
)
from ..runtime.state_store import ResearchStateStore


@dataclass(frozen=True)
class ExperimentDesignConfig:
    max_tokens: int = 4_000

    def __post_init__(self) -> None:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "ExperimentDesignConfig":
        values = os.environ if environ is None else environ
        raw = values.get("EXPERIMENT_DESIGN_MAX_TOKENS")
        if raw is None or not raw.strip():
            return cls()
        try:
            return cls(max_tokens=int(raw))
        except ValueError as exc:
            raise ValueError("EXPERIMENT_DESIGN_MAX_TOKENS must be an integer") from exc


@dataclass(frozen=True)
class ExperimentDesignExecutionResult:
    status: Literal["ready", "unable_to_design"]
    experiments: list[Experiment]
    design_summary: str
    key_claims: list[KeyClaimDraft]
    redesigned_experiment_ids: list[str]
    failure_reason: str | None = None
    next_required_action: None = None


def _prompt_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / name


def _inject_schema(name: str, placeholder: str, schema_type: type[Any]) -> str:
    template = _prompt_path(name).read_text(encoding="utf-8")
    schema = json.dumps(
        schema_type.model_json_schema(), ensure_ascii=False, separators=(",", ":")
    )
    return template.replace(placeholder, schema)


def _copy_state(state: ResearchState) -> ResearchState:
    return ResearchState.model_validate(state.model_dump(mode="python"))


def _copy_experiment(experiment: Experiment) -> Experiment:
    return Experiment.model_validate(experiment.model_dump(mode="python"))


def _copy_claim(claim: KeyClaimDraft) -> KeyClaimDraft:
    return KeyClaimDraft.model_validate(claim.model_dump(mode="python"))


def _contains_token(text: str, token: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) is not None


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _verification_summary(record: VerificationRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "task": record.task,
        "target_ref": record.target_ref,
        "verification_focus": (
            record.verification_focus.model_dump(mode="json")
            if record.verification_focus is not None
            else None
        ),
        "result": record.result.model_dump(mode="json"),
        "paper_refs": record.paper_refs,
        "created_at_step": record.created_at_step,
    }


def _solution_payload(solution: CurrentSolution) -> dict[str, Any]:
    return solution.model_dump(mode="json")


def _proposal_to_experiment(
    proposal: ExperimentProposalDraft,
    experiment_id: str,
) -> Experiment:
    return Experiment(
        id=experiment_id,
        objective=proposal.objective,
        setup=proposal.setup,
        baselines=list(proposal.baselines),
        metrics=list(proposal.metrics),
        what_it_validates=proposal.what_it_validates,
        validation_criterion=proposal.validation_criterion,
        status="ready",
    )


def _claim_anchor(claim: KeyClaimDraft) -> str:
    return claim.target_unit


def _validate_claims(
    claims: list[KeyClaimDraft],
    solution: CurrentSolution,
    *,
    mode: Literal["initial", "redesign"],
) -> dict[str, KeyClaimDraft]:
    minimum = 2 if mode == "initial" else 1
    if not minimum <= len(claims) <= 4:
        raise ModuleContractError(
            f"{mode} Experiment Design requires {minimum}-4 Key Claims"
        )
    expected_ids = [f"K{index}" for index in range(1, len(claims) + 1)]
    actual_ids = [claim.id for claim in claims]
    if actual_ids != expected_ids:
        raise ModuleContractError("Key Claim IDs must be ordered K1..Kn")
    if len(set(actual_ids)) != len(actual_ids):
        raise ModuleContractError("Key Claim IDs must be unique")
    component_ids = {component.id for component in solution.components}
    for claim in claims:
        if claim.target_unit != "core_solution" and claim.target_unit not in component_ids:
            raise ModuleContractError(
                f"Key Claim {claim.id} targets an unknown Solution unit {claim.target_unit}"
            )
    if mode == "initial":
        if not any(claim.target_unit == "core_solution" for claim in claims):
            raise ModuleContractError("Initial Key Claims require a core_solution claim")
        if not any(claim.target_unit in component_ids for claim in claims):
            raise ModuleContractError("Initial Key Claims require a Component claim")
    return {claim.id: claim for claim in claims}


def _validate_proposal_fields(
    proposal: ExperimentProposalDraft,
    claims: dict[str, KeyClaimDraft],
) -> set[str]:
    if not proposal.baselines:
        raise ModuleContractError("each Experiment requires at least one baseline")
    if not proposal.metrics:
        raise ModuleContractError("each Experiment requires at least one metric")
    if not proposal.validates_claims:
        raise ModuleContractError("each Experiment must validate at least one Key Claim")
    if len(set(proposal.validates_claims)) != len(proposal.validates_claims):
        raise ModuleContractError("validates_claims must not contain duplicates")
    unknown = set(proposal.validates_claims) - set(claims)
    if unknown:
        raise ModuleContractError(
            "Experiment references unknown Key Claims: " + ", ".join(sorted(unknown))
        )
    selected = set(proposal.validates_claims)
    for claim_id in selected:
        claim = claims[claim_id]
        if not _contains_token(proposal.what_it_validates, _claim_anchor(claim)):
            raise ModuleContractError(
                f"Experiment what_it_validates must contain anchor {claim.target_unit}"
            )
    return selected


def _validate_identifiability(
    checks: list[ClaimIdentifiabilityDraft],
    claims: dict[str, KeyClaimDraft],
) -> list[str]:
    """Validate the check structure and return normal scientific failures.

    A false ``identifiable`` value is a valid conclusion of Experiment Design,
    not a malformed structured response.  Structural problems remain contract
    errors; scientific non-identifiability is returned to the caller.
    """

    expected = list(claims)
    actual = [check.claim_id for check in checks]
    if actual != expected or len(set(actual)) != len(actual):
        raise ModuleContractError(
            "identifiability_checks must contain each Key Claim exactly once in order"
        )
    return [
        f"Key Claim {check.claim_id}: no reliable identifiability path"
        for check in checks
        if not check.identifiable
    ]


def _validate_initial_experiment_uniqueness(
    experiments: list[Experiment],
    coverage: dict[str, list[str]],
) -> None:
    objectives: set[str] = set()
    for experiment in experiments:
        objective = _normalized(experiment.objective)
        if objective in objectives:
            raise ModuleContractError("Experiment objectives must not be duplicates")
        objectives.add(objective)
    experiment_claims: dict[str, set[str]] = {
        experiment_id: set() for experiment in experiments for experiment_id in [experiment.id]
    }
    for claim_id, experiment_ids in coverage.items():
        for experiment_id in experiment_ids:
            experiment_claims.setdefault(experiment_id, set()).add(claim_id)
    signatures = {
        (_normalized(experiment.setup), frozenset(experiment_claims[experiment.id]))
        for experiment in experiments
    }
    if len(signatures) != len(experiments):
        raise ModuleContractError(
            "Experiments must not duplicate setup and Key Claim coverage"
        )


def _validate_initial_coverage(
    claims: dict[str, KeyClaimDraft],
    final_experiments: list[Experiment],
    proposal_claims: dict[str, set[str]],
    solution: CurrentSolution,
) -> dict[str, list[str]]:
    coverage: dict[str, list[str]] = {claim_id: [] for claim_id in claims}
    for experiment in final_experiments:
        if experiment.id in proposal_claims:
            claim_ids = proposal_claims[experiment.id]
        else:
            claim_ids = {
                claim.id
                for claim in claims.values()
                if _contains_token(experiment.what_it_validates, _claim_anchor(claim))
            }
        if not claim_ids:
            raise ModuleContractError(
                f"Experiment {experiment.id} does not cover a Key Claim"
            )
        for claim_id in sorted(claim_ids):
            coverage[claim_id].append(experiment.id)

    for claim_id, experiment_ids in coverage.items():
        if not experiment_ids:
            raise ModuleContractError(f"Key Claim {claim_id} is not covered")

    core_claim_ids = [
        claim.id for claim in claims.values() if claim.target_unit == "core_solution"
    ]
    component_ids = {component.id for component in solution.components}
    mechanism_claim_ids = [
        claim.id for claim in claims.values() if claim.target_unit in component_ids
    ]
    if not any(coverage[claim_id] for claim_id in core_claim_ids):
        raise ModuleContractError("no Experiment covers the core_solution claim")
    if not any(coverage[claim_id] for claim_id in mechanism_claim_ids):
        raise ModuleContractError("no Experiment covers a Component claim")
    return coverage


def _validate_redesign_coverage(
    claims: dict[str, KeyClaimDraft],
    proposal_claims: dict[str, set[str]],
    stale_ids: list[str],
) -> dict[str, list[str]]:
    """Map only new Claims to the stale proposals that explicitly cover them."""

    if set(proposal_claims) != set(stale_ids):
        raise ModuleContractError(
            "redesign coverage must be defined only for stale Experiments"
        )
    coverage = {claim_id: [] for claim_id in claims}
    for experiment_id, claim_ids in proposal_claims.items():
        if not claim_ids:
            raise ModuleContractError(
                f"stale Experiment {experiment_id} does not validate a Key Claim"
            )
        for claim_id in claim_ids:
            if claim_id not in claims:
                raise ModuleContractError(
                    f"redesign references unknown Key Claim {claim_id}"
                )
            coverage[claim_id].append(experiment_id)
    for claim_id, experiment_ids in coverage.items():
        if not experiment_ids:
            raise ModuleContractError(
                f"redesign Key Claim {claim_id} is not covered by a stale proposal"
            )
    return coverage


def _validate_final_experiment_structure(
    experiments: list[Experiment],
    solution: CurrentSolution,
) -> None:
    """Validate canonical final experiments without interpreting new Claims."""

    if not 2 <= len(experiments) <= 4:
        raise ModuleContractError("final Experiment count must be 2-4")
    if len({experiment.id for experiment in experiments}) != len(experiments):
        raise ModuleContractError("Experiment IDs must be unique")
    component_ids = {component.id for component in solution.components}
    for experiment in experiments:
        if not experiment.objective.strip() or not experiment.setup.strip():
            raise ModuleContractError("Experiment objective and setup must be non-empty")
        if not experiment.what_it_validates.strip():
            raise ModuleContractError("Experiment what_it_validates must be non-empty")
        if not experiment.validation_criterion.strip():
            raise ModuleContractError("Experiment validation_criterion must be non-empty")
        if not experiment.baselines or any(not value.strip() for value in experiment.baselines):
            raise ModuleContractError("Experiment baselines must be non-empty")
        if not experiment.metrics or any(not value.strip() for value in experiment.metrics):
            raise ModuleContractError("Experiment metrics must be non-empty")
    if not any(
        _contains_token(experiment.what_it_validates, "core_solution")
        for experiment in experiments
    ):
        raise ModuleContractError("final Experiments require a core_solution anchor")
    if not any(
        any(
            _contains_token(experiment.what_it_validates, component_id)
            for component_id in component_ids
        )
        for experiment in experiments
    ):
        raise ModuleContractError("final Experiments require a Component anchor")
    if not any(
        _contains_token(experiment.what_it_validates, "core_solution")
        and "Seed Method" in experiment.baselines
        for experiment in experiments
    ):
        raise ModuleContractError(
            'at least one core effectiveness Experiment requires the exact "Seed Method" baseline'
        )


def _ensure_seed_method_baseline(
    experiments: list[Experiment],
) -> list[Experiment]:
    """Materialize the canonical seed baseline for an initial core claim.

    ``Seed Method`` is an internal contract label, not a paper-specific method
    name.  A model may describe the original method using the paper's title or
    another phrase even after being instructed to use the canonical label.  Add
    the required comparison deterministically to the first core-effectiveness
    proposal instead of rejecting an otherwise valid experiment plan.
    """

    if any(
        _contains_token(experiment.what_it_validates, "core_solution")
        and "Seed Method" in experiment.baselines
        for experiment in experiments
    ):
        return experiments

    target_index = next(
        (
            index
            for index, experiment in enumerate(experiments)
            if _contains_token(experiment.what_it_validates, "core_solution")
        ),
        None,
    )
    if target_index is None:
        return experiments

    materialized = [_copy_experiment(experiment) for experiment in experiments]
    materialized[target_index].baselines.append("Seed Method")
    return materialized


def _validate_redesign_experiment_uniqueness(
    experiments: list[Experiment],
    solution: CurrentSolution,
) -> None:
    """Reject obvious canonical duplicates without using new Claim semantics."""

    component_ids = {component.id for component in solution.components}
    signatures = []
    for experiment in experiments:
        anchors = tuple(
            sorted(
                [
                    anchor
                    for anchor in ["core_solution", *sorted(component_ids)]
                    if _contains_token(experiment.what_it_validates, anchor)
                ]
            )
        )
        signatures.append(
            (
                _normalized(experiment.objective),
                _normalized(experiment.setup),
                anchors,
            )
        )
    if len(signatures) != len(set(signatures)):
        raise ModuleContractError("final Experiments must not be obvious canonical duplicates")


class ExperimentDesignModule:
    """Design or deterministically redesign canonical Experiments once."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: ExperimentDesignConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or ExperimentDesignConfig.from_env()
        self._prompt = _inject_schema(
            "experiment_design.md",
            "{{EXPERIMENT_DESIGN_DRAFT_JSON_SCHEMA}}",
            ExperimentDesignDraft,
        )
        self.call_count = 0
        self.last_call_metadata: LLMCallMetadata | None = None
        self.last_input_chars: int | None = None

    def _complete(
        self,
        *,
        system_prompt: str,
        user_content: str,
        paper_id: str,
    ) -> StructuredLLMResponse[Any]:
        request = LLMRequest(
            messages=(
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_content),
            ),
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            metadata={"task": "experiment_design", "paper_id": paper_id},
        )
        response = self.llm_client.complete_structured(request, ExperimentDesignDraft)
        self.call_count += 1
        self.last_call_metadata = response.metadata
        self.last_input_chars = sum(len(message.content) for message in request.messages)
        return response

    @staticmethod
    def _resolve_inputs(
        active_gap,
        current_solution,
        seed_context: SeedPaperContext | None,
        context,
        state_store: ResearchStateStore | None,
        memory_store: ResearchMemoryStore | None,
    ) -> tuple[SeedPaperContext, ResearchMemoryStore, ResearchState | None, ResearchStateStore | None]:
        if context is not None:
            if seed_context is not None and seed_context.paper_id != context.paper_id:
                raise PreconditionError("SeedPaperContext does not match trajectory")
            seed = context.seed_context
            store = context.memory_store
            state = context.get_state()
            resolved_state_store = context.state_store
        else:
            if seed_context is None:
                raise TypeError("Experiment Design requires SeedPaperContext or context")
            seed = SeedPaperContext.model_validate(
                seed_context.model_dump(mode="python")
            )
            store = memory_store or ResearchMemoryStore(seed.paper_id)
            state = state_store.get_state() if state_store is not None else None
            resolved_state_store = state_store
        if store.seed_paper_id != seed.paper_id:
            raise PreconditionError("Research Memory does not match Seed Paper")
        return seed, store, state, resolved_state_store

    def _preconditions(
        self,
        active_gap,
        current_solution,
        state: ResearchState | None,
        store: ResearchMemoryStore,
    ) -> tuple[
        CandidateGap,
        CurrentSolution,
        VerificationRecord,
        VerificationRecord,
        list[str],
        list[Experiment],
    ]:
        if state is None:
            raise PreconditionError(
                "Experiment Design requires a TrajectoryContext or ResearchStateStore"
            )
        state_gap = state.active_gap
        state_solution = state.current_solution
        if state_gap is None or state_gap.status != "active":
            raise PreconditionError("Experiment Design requires an active Research State gap")
        if state_solution is None:
            raise PreconditionError("Experiment Design requires a Current Solution")
        if active_gap is not None and active_gap.id != state_gap.id:
            raise PreconditionError("active_gap does not match Research State")
        if current_solution is not None and current_solution.id != state_solution.id:
            raise PreconditionError("current_solution does not match Research State")
        gap_value = CandidateGap.model_validate(state_gap.model_dump(mode="python"))
        solution = CurrentSolution.model_validate(
            state_solution.model_dump(mode="python")
        )
        if solution.novelty_status != "distinct":
            raise PreconditionError(
                "Experiment Design requires Current Solution novelty_status=distinct"
            )
        unfinished = [
            component.id
            for component in solution.components
            if component.status != "ready"
        ]
        if unfinished:
            raise PreconditionError(
                "Experiment Design requires all Components to be ready: "
                + ", ".join(unfinished)
            )
        gap_verification = store.get_latest_gap_verification(gap_value.id)
        if gap_verification is None or not isinstance(
            gap_verification.result, GapVerificationResult
        ):
            raise PreconditionError("Experiment Design requires a latest Gap Verification")
        if gap_verification.result.status != "valid":
            raise PreconditionError("latest Gap Verification must be valid")
        solution_verification = store.get_latest_verification(
            solution.id, task="VERIFY_SOLUTION"
        )
        if solution_verification is None or not isinstance(
            solution_verification.result, SolutionVerificationResult
        ):
            raise PreconditionError(
                "Experiment Design requires a latest Solution Verification"
            )
        if solution_verification.result.novelty_status != "distinct":
            raise PreconditionError("latest Solution Verification must be distinct")
        changed_units: list[str] = []
        focus = solution_verification.verification_focus
        if focus is not None and focus.mode == "changed_units":
            changed_units = list(focus.changed_units)
            known = {component.id for component in solution.components}
            if set(changed_units) - known:
                raise ModuleContractError(
                    "latest verification focus references an unknown Component"
                )
        experiments = [
            _copy_experiment(experiment) for experiment in state.experiments
        ]
        if len(experiments) == 1:
            raise PreconditionError("Experiment Design requires 2-4 existing Experiments")
        if len(experiments) > 4:
            raise PreconditionError("Experiment Design does not silently truncate Experiments")
        ids = [experiment.id for experiment in experiments]
        if len(set(ids)) != len(ids):
            raise ModuleContractError("Experiment IDs must be unique")
        return (
            gap_value,
            solution,
            gap_verification,
            solution_verification,
            changed_units,
            experiments,
        )

    @staticmethod
    def _validate_and_materialize(
        draft: ExperimentDesignDraft,
        solution: CurrentSolution,
        existing: list[Experiment],
    ) -> tuple[list[Experiment], dict[str, list[str]], list[str], list[str]]:
        mode: Literal["initial", "redesign"] = "initial" if not existing else "redesign"
        claims = _validate_claims(draft.key_claims, solution, mode=mode)
        identifiability_issues = _validate_identifiability(
            draft.identifiability_checks, claims
        )
        if identifiability_issues:
            return (
                [_copy_experiment(experiment) for experiment in existing],
                {},
                [],
                identifiability_issues,
            )
        stale = [experiment for experiment in existing if experiment.status == "stale"]
        if not existing:
            if any(proposal.existing_experiment_id is not None for proposal in draft.experiments):
                raise ModuleContractError(
                    "initial Experiment Design proposals must not provide existing IDs"
                )
            proposals = list(draft.experiments)
            if not 2 <= len(proposals) <= 4:
                raise ModuleContractError("initial Experiment Design requires 2-4 proposals")
            proposal_ids = [f"E{index}" for index in range(1, len(proposals) + 1)]
        else:
            if not stale:
                raise ModuleContractError("non-empty non-ready Experiments require stale redesign")
            target_ids = [experiment.id for experiment in stale]
            returned_ids = [proposal.existing_experiment_id for proposal in draft.experiments]
            if any(identifier is None for identifier in returned_ids):
                raise ModuleContractError(
                    "stale redesign proposals must identify existing Experiments"
                )
            if len(returned_ids) != len(set(returned_ids)):
                raise ModuleContractError("stale redesign must not duplicate Experiment IDs")
            if set(returned_ids) != set(target_ids):
                raise ModuleContractError(
                    "stale redesign must include each stale Experiment exactly once"
                )
            proposals = list(draft.experiments)
            proposal_ids = [identifier for identifier in returned_ids if identifier is not None]

        proposal_claims: dict[str, set[str]] = {}
        proposal_by_id: dict[str, Experiment] = {}
        for proposal, experiment_id in zip(proposals, proposal_ids):
            claim_ids = _validate_proposal_fields(proposal, claims)
            proposal_claims[experiment_id] = claim_ids
            proposal_by_id[experiment_id] = _proposal_to_experiment(proposal, experiment_id)

        if not existing:
            final_experiments = _ensure_seed_method_baseline(
                [proposal_by_id[identifier] for identifier in proposal_ids]
            )
        else:
            final_experiments = [
                proposal_by_id[experiment.id] if experiment.status == "stale" else experiment
                for experiment in existing
            ]
        _validate_final_experiment_structure(final_experiments, solution)
        if mode == "initial":
            coverage = _validate_initial_coverage(
                claims, final_experiments, proposal_claims, solution
            )
            _validate_initial_experiment_uniqueness(final_experiments, coverage)
        else:
            coverage = _validate_redesign_coverage(
                claims, proposal_claims, [experiment.id for experiment in stale]
            )
            _validate_redesign_experiment_uniqueness(final_experiments, solution)
        redesigned_ids = [experiment.id for experiment in stale]
        return final_experiments, coverage, redesigned_ids, []

    def design(
        self,
        active_gap=None,
        current_solution: CurrentSolution | None = None,
        seed_context: SeedPaperContext | None = None,
        *,
        context=None,
        state_store: ResearchStateStore | None = None,
        memory_store: ResearchMemoryStore | None = None,
        step: int = 1,
    ) -> ExperimentDesignExecutionResult:
        if context is None and hasattr(active_gap, "seed_context"):
            context = active_gap
            active_gap = None
        if isinstance(active_gap, SeedPaperContext) and seed_context is None:
            seed_context = active_gap
            active_gap = None
        seed, store, state, resolved_state_store = self._resolve_inputs(
            active_gap,
            current_solution,
            seed_context,
            context,
            state_store,
            memory_store,
        )
        (
            gap_value,
            solution,
            gap_verification,
            solution_verification,
            changed_units,
            existing,
        ) = self._preconditions(active_gap, current_solution, state, store)
        if context is not None:
            try:
                context.trajectory.validate_next_step(step)
            except ValueError as exc:
                raise PreconditionError(str(exc)) from exc

        if existing and all(experiment.status == "ready" for experiment in existing):
            return ExperimentDesignExecutionResult(
                status="ready",
                experiments=[_copy_experiment(experiment) for experiment in existing],
                design_summary="Existing Experiments are already ready; no redesign was needed.",
                key_claims=[],
                redesigned_experiment_ids=[],
                next_required_action=None,
            )

        user_content = json.dumps(
            {
                "active_gap": gap_value.model_dump(mode="json"),
                "current_solution": _solution_payload(solution),
                "seed_experiment_context": seed.experiment_context.model_dump(mode="json"),
                "latest_gap_verification": _verification_summary(gap_verification),
                "latest_solution_verification": _verification_summary(solution_verification),
                "changed_units": changed_units,
                "existing_experiments": [
                    experiment.model_dump(mode="json") for experiment in existing
                ],
                "stale_redesign_targets": [
                    experiment.model_dump(mode="json")
                    for experiment in existing
                    if experiment.status == "stale"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        before = _copy_state(state) if state is not None else None
        response = self._complete(
            system_prompt=self._prompt,
            user_content=user_content,
            paper_id=seed.paper_id,
        )
        draft = response.output
        if not isinstance(draft, ExperimentDesignDraft):
            raise ModuleContractError("Experiment Design response has the wrong schema")
        (
            final_experiments,
            coverage,
            redesigned_ids,
            identifiability_issues,
        ) = self._validate_and_materialize(draft, solution, existing)
        assert before is not None
        if identifiability_issues:
            failure_reason = "; ".join(identifiability_issues)
            if context is not None:
                context.record_transition(
                    step,
                    "experiments_design_unresolved",
                    before,
                    before,
                    action="DESIGN_EXPERIMENT",
                    result_ref=solution.id,
                    result_payload={
                        "status": "unable_to_design",
                        "failure_reason": failure_reason,
                        "key_claims": [
                            claim.model_dump(mode="json") for claim in draft.key_claims
                        ],
                        "identifiability_checks": [
                            check.model_dump(mode="json")
                            for check in draft.identifiability_checks
                        ],
                    },
                    memory_writes=[],
                )
            return ExperimentDesignExecutionResult(
                status="unable_to_design",
                experiments=[_copy_experiment(experiment) for experiment in existing],
                design_summary=draft.design_summary,
                key_claims=[_copy_claim(claim) for claim in draft.key_claims],
                redesigned_experiment_ids=[],
                failure_reason=failure_reason,
                next_required_action=None,
            )
        after = _copy_state(before)
        after.experiments = [_copy_experiment(experiment) for experiment in final_experiments]
        payload = {
            "design_mode": "initial" if not existing else "redesign",
            "key_claims": [claim.model_dump(mode="json") for claim in draft.key_claims],
            "design_summary": draft.design_summary,
            "experiment_ids": [experiment.id for experiment in final_experiments],
            "redesigned_experiment_ids": redesigned_ids,
            "claim_coverage": coverage,
        }
        if context is not None:
            context.record_transition(
                step,
                "experiments_designed",
                before,
                after,
                action="DESIGN_EXPERIMENT",
                result_ref=solution.id,
                result_payload=payload,
                memory_writes=[],
            )
        elif resolved_state_store is not None:
            resolved_state_store.replace_state(after)
        return ExperimentDesignExecutionResult(
            status="ready",
            experiments=[_copy_experiment(experiment) for experiment in final_experiments],
            design_summary=draft.design_summary,
            key_claims=[_copy_claim(claim) for claim in draft.key_claims],
            redesigned_experiment_ids=list(redesigned_ids),
            failure_reason=None,
            next_required_action=None,
        )

    run = design
    execute = design


__all__ = [
    "ExperimentDesignConfig",
    "ExperimentDesignExecutionResult",
    "ExperimentDesignModule",
]
