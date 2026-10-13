"""Dynamic Entity Bridging and inspiration retrieval contracts."""

from typing import Literal

from pydantic import Field, model_validator

from .base import NonEmptyStr, StrictModel


class InspirationNeed(StrictModel):
    current_problem: str
    required_function: str
    current_mechanism: str
    rejected_mechanisms: list[str] = Field(default_factory=list)
    relevant_context: str


class BridgeEntity(StrictModel):
    name: NonEmptyStr
    type: Literal[
        "scientific concept",
        "mechanism",
        "method",
        "phenomenon",
        "system principle",
    ]
    bridge_reason: str


class BridgeGenerationDraft(StrictModel):
    """Internal Call-1 output for Dynamic Entity Bridging.

    This is an LLM reasoning artifact.  It is validated before any search or
    Research Memory write and is never materialized into Research State.
    """

    inspiration_need: InspirationNeed
    proposed_entities: list[BridgeEntity] = Field(min_length=5, max_length=5)
    selected_entities: list[BridgeEntity] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_entity_selection(self) -> "BridgeGenerationDraft":
        def key(entity: BridgeEntity) -> str:
            return " ".join(entity.name.casefold().split())

        proposed_keys = [key(entity) for entity in self.proposed_entities]
        if len(set(proposed_keys)) != len(proposed_keys):
            raise ValueError("proposed bridge entity names must be unique")
        selected_keys = [key(entity) for entity in self.selected_entities]
        if len(set(selected_keys)) != len(selected_keys):
            raise ValueError("selected bridge entity names must be unique")
        if not set(selected_keys) <= set(proposed_keys):
            raise ValueError("selected bridge entities must come from proposed_entities")
        proposed_by_key = {
            key(entity): entity for entity in self.proposed_entities
        }
        # Selection is an index into the proposed canonical definitions.  This
        # prevents a same-name entity from silently changing its type or reason.
        object.__setattr__(
            self,
            "selected_entities",
            [proposed_by_key[key(entity)] for entity in self.selected_entities],
        )
        return self


class InspirationCandidate(StrictModel):
    bridge_entity: BridgeEntity
    paper_ref: str | None = None
    source_problem: str
    source_mechanism: str
    transferable_principle: str
    adaptation_hint: str
    grounding: Literal["arxiv", "none"]


class InspirationResult(StrictModel):
    status: Literal["found", "no_useful_inspiration"]
    inspirations: list[InspirationCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_inspiration_count(self) -> "InspirationResult":
        count = len(self.inspirations)
        if self.status == "found" and not 1 <= count <= 3:
            raise ValueError("found InspirationResult requires 1 to 3 inspirations")
        if self.status == "no_useful_inspiration" and count != 0:
            raise ValueError("no_useful_inspiration must have no inspirations")
        return self
