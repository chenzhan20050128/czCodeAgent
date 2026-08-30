"""Environment-backed configuration for mca."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_CONTEXT_WINDOW = 1_000_000
DEFAULT_MAX_OUTPUT_TOKENS = 512_000
_THINKING_MODES = frozenset({"enabled", "disabled"})
DEEPSEEK_MAX_OUTPUT_TOKENS = 384_000


@dataclass(frozen=True, repr=False)
class Config:
    """Immutable model and runtime configuration."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    thinking: str = "enabled"
    max_steps: int = 64
    max_tool_calls_per_batch: int = 8
    max_parallel_tool_calls: int = 4
    request_timeout: float = 600.0
    max_attempts: int = 3
    retry_budget_seconds: float = 900.0
    verbose: bool = False
    yolo: bool = False

    @property
    def request_max_output_tokens(self) -> int:
        """Return the output cap actually sent to the configured provider."""

        if self.base_url.rstrip("/").lower().startswith(
            "https://api.deepseek.com"
        ):
            return min(self.max_output_tokens, DEEPSEEK_MAX_OUTPUT_TOKENS)
        return self.max_output_tokens

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

        raw_max_output = os.environ.get(
            "MCA_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)
        )
        try:
            max_output_tokens = int(raw_max_output)
        except ValueError:
            raise ValueError(
                "MCA_MAX_OUTPUT_TOKENS must be a positive integer"
            ) from None
        if max_output_tokens <= 0:
            raise ValueError(
                "MCA_MAX_OUTPUT_TOKENS must be a positive integer"
            )

        thinking = os.environ.get("MCA_THINKING", "enabled").strip().lower()
        if thinking not in _THINKING_MODES:
            raise ValueError("MCA_THINKING must be enabled or disabled")

        max_steps = _positive_int_from_env("MCA_MAX_STEPS", 64)
        max_parallel_tool_calls = _positive_int_from_env(
            "MCA_MAX_PARALLEL_TOOL_CALLS", 4
        )
        request_timeout = _positive_float_from_env("MCA_REQUEST_TIMEOUT", 600.0)
        retry_budget_seconds = _positive_float_from_env(
            "MCA_RETRY_BUDGET_SECONDS", 900.0
        )

        return cls(
            base_url=os.environ.get("MCA_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key,
            model=os.environ.get("MCA_MODEL", DEFAULT_MODEL),
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            thinking=thinking,
            max_steps=max_steps,
            max_parallel_tool_calls=max_parallel_tool_calls,
            request_timeout=request_timeout,
            retry_budget_seconds=retry_budget_seconds,
        )

    def __repr__(self) -> str:
        api_key = "<redacted>" if self.api_key else None
        return (
            f"Config(base_url={self.base_url!r}, api_key={api_key!r}, "
            f"model={self.model!r}, context_window={self.context_window!r}, "
            f"max_output_tokens={self.max_output_tokens!r}, "
            f"thinking={self.thinking!r}, "
            f"max_steps={self.max_steps!r}, "
            "max_tool_calls_per_batch="
            f"{self.max_tool_calls_per_batch!r}, "
            f"max_parallel_tool_calls={self.max_parallel_tool_calls!r}, "
            f"request_timeout={self.request_timeout!r}, "
            f"max_attempts={self.max_attempts!r}, "
            f"retry_budget_seconds={self.retry_budget_seconds!r}, "
            f"verbose={self.verbose!r}, yolo={self.yolo!r})"
        )


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive number") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value
