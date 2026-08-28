"""Environment-backed configuration for mca."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_CONTEXT_WINDOW = 65_536


@dataclass(frozen=True, repr=False)
class Config:
    """Immutable model and runtime configuration."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int = 8_192
    max_steps: int = 20
    max_tool_calls_per_batch: int = 8
    request_timeout: float = 120.0
    max_attempts: int = 3
    retry_budget_seconds: float = 60.0
    verbose: bool = False
    yolo: bool = False

    @classmethod
    def from_env(cls, *, require_api_key: bool = True) -> Config:
        """Build configuration from environment variables.

        ``MCA_API_KEY`` takes precedence over the compatibility fallback
        ``DEEPSEEK_API_KEY``. Tests and non-live commands can explicitly skip
        credential validation with ``require_api_key=False``.
        """

        api_key = os.environ.get("MCA_API_KEY") or os.environ.get(
            "DEEPSEEK_API_KEY"
        )
        if require_api_key and not api_key:
            raise ValueError(
                "API key is required; set MCA_API_KEY or DEEPSEEK_API_KEY"
            )

        raw_context_window = os.environ.get(
            "MCA_CONTEXT_WINDOW", str(DEFAULT_CONTEXT_WINDOW)
        )
        try:
            context_window = int(raw_context_window)
        except ValueError:
            raise ValueError(
                "MCA_CONTEXT_WINDOW must be a positive integer"
            ) from None
        if context_window <= 0:
            raise ValueError(
                "MCA_CONTEXT_WINDOW must be a positive integer"
            )

        return cls(
            base_url=os.environ.get("MCA_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key,
            model=os.environ.get("MCA_MODEL", DEFAULT_MODEL),
            context_window=context_window,
        )

    def __repr__(self) -> str:
        api_key = "<redacted>" if self.api_key else None
        return (
            f"Config(base_url={self.base_url!r}, api_key={api_key!r}, "
            f"model={self.model!r}, context_window={self.context_window!r}, "
            f"max_output_tokens={self.max_output_tokens!r}, "
            f"max_steps={self.max_steps!r}, "
            "max_tool_calls_per_batch="
            f"{self.max_tool_calls_per_batch!r}, "
            f"request_timeout={self.request_timeout!r}, "
            f"max_attempts={self.max_attempts!r}, "
            f"retry_budget_seconds={self.retry_budget_seconds!r}, "
            f"verbose={self.verbose!r}, yolo={self.yolo!r})"
        )
