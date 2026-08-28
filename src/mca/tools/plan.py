"""Plan-mode exit tool: a validated, approval-gated request to leave plan mode.

The tool has no filesystem or shell effect of its own. Approving it is the
user's decision to leave plan mode; the executor records that as a durable
``plan_mode_set`` fact. It reuses the same one-time approval gate as the
side-effecting tools, so plan review needs no separate interaction channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class PlanToolError(ValueError):
    """Raised when an exit_plan_mode request is not a reviewable plan."""


@dataclass(frozen=True)
class PreparedPlanExit:
    """A validated plan awaiting the user's approval to leave plan mode."""

    plan: str


def prepare_exit_plan_mode(arguments: dict[str, Any]) -> PreparedPlanExit:
    if not isinstance(arguments, dict):
        raise PlanToolError("arguments must be an object")
    plan = arguments.get("plan")
    if not isinstance(plan, str) or not plan.strip():
        raise PlanToolError("plan must be a non-empty string")
    if not plan.lstrip().startswith("#"):
        raise PlanToolError("plan must be markdown starting with a # heading")
    return PreparedPlanExit(plan=plan)


__all__ = ["PlanToolError", "PreparedPlanExit", "prepare_exit_plan_mode"]
