from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ParamSpec(BaseModel):
    type: Literal["str", "int", "decimal", "bool"]
    required: bool = True
    sensitive: bool = False
    description: str


class OutputSpec(BaseModel):
    type: Literal["str", "int", "decimal", "bool"]
    sensitive: bool = False
    description: str


class Locator(BaseModel):
    strategy: Literal["role_name", "aria_label", "text_exact", "css_fallback", "xpath"]
    value: str
    role: str | None = None
    fallback: Locator | None = None

    model_config = {"arbitrary_types_allowed": False}


# Required so the forward ref in fallback resolves
Locator.model_rebuild()


class Step(BaseModel):
    index: int
    action: Literal["navigate", "click", "type", "read", "wait_for"]
    locator: Locator | None = None
    value: str | None = None
    output_key: str | None = None
    reversible: bool

    @model_validator(mode="after")
    def navigate_has_no_locator(self) -> Step:
        if self.action == "navigate" and self.locator is not None:
            raise ValueError("navigate steps must not have a locator")
        if self.action == "navigate" and self.value is None:
            raise ValueError("navigate steps must have a value (the URL)")
        return self

    @model_validator(mode="after")
    def read_has_output_key(self) -> Step:
        if self.action == "read" and self.output_key is None:
            raise ValueError("read steps must declare an output_key")
        return self


class Checkpoint(BaseModel):
    kind: Literal["element_visible", "text_present", "url_matches"]
    locator: Locator | None = None
    expected: str


class BusinessOutcomeSpec(BaseModel):
    """
    Declares a known non-success business state the replay may reach.

    The executor checks these after a step fails or after the primary
    checkpoint fails. On the first match it returns status="business_outcome"
    rather than hard_failure.  Without a matching spec, any checkpoint or
    step failure is hard_failure by default — there is no positive signal
    to distinguish a known outcome from an unexpected breakage.

    outcome_code   — short machine-readable token (e.g. "MEMBER_NOT_FOUND")
    outcome_message — human-readable description for logs / UI
    checkpoint      — the page-state condition that positively identifies
                      this outcome (same schema as the success checkpoint)
    """
    outcome_code: str
    outcome_message: str
    checkpoint: Checkpoint


class Capability(BaseModel):
    id: str
    version: int = 1
    name: str
    goal: str
    target: dict
    inputs: dict[str, ParamSpec]
    outputs: dict[str, OutputSpec]
    steps: list[Step]
    business_outcomes: list[BusinessOutcomeSpec] = []
    checkpoint: Checkpoint
    created_from_run_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def steps_are_indexed(self) -> Capability:
        for i, step in enumerate(self.steps):
            if step.index != i:
                raise ValueError(
                    f"Step at position {i} has index {step.index}; must match position"
                )
        return self

    @model_validator(mode="after")
    def output_keys_declared(self) -> Capability:
        declared = set(self.outputs.keys())
        referenced = {s.output_key for s in self.steps if s.output_key}
        undeclared = referenced - declared
        if undeclared:
            raise ValueError(
                f"Steps reference output keys not declared in outputs: {undeclared}"
            )
        return self


class ReplayResult(BaseModel):
    status: Literal["success", "business_outcome", "hard_failure"]
    outputs: dict | None = None
    outcome_code: str | None = None
    outcome_message: str | None = None
    failure_step_index: int | None = None
    failure_expected: str | None = None
    failure_observed: str | None = None
    screenshot_path: str | None = None
