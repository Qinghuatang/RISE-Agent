"""Shared validation primitives for scientific schemas."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints


class StrictModel(BaseModel):
    """Base for all canonical contracts.

    Scientific payloads are deliberately closed: an unexpected field should
    fail loudly instead of silently changing the method contract.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

ScientificAction = Literal[
    "SOLUTION_IDEATION",
    "SEEK_INSPIRATION",
    "VERIFY_RELATED_WORK",
    "TARGETED_REVISION",
    "DESIGN_EXPERIMENT",
    "SWITCH_GAP",
    "COMMIT",
]
