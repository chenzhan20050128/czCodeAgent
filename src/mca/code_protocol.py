"""Size-bounded canonical JSONL protocol for the Code Mode worker."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


DEFAULT_MAX_FRAME_BYTES = 1024 * 1024


class ProtocolFrameError(ValueError):
    """Raised when a worker protocol frame is invalid or oversized."""


def encode_frame(
    value: Mapping[str, Any], *, max_bytes: int = DEFAULT_MAX_FRAME_BYTES
) -> bytes:
    if not isinstance(value, Mapping):
        raise ProtocolFrameError("protocol frame must be an object")
    try:
        encoded = (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ProtocolFrameError("protocol frame is not lossless JSON") from error
    if len(encoded) > max_bytes:
        raise ProtocolFrameError("protocol frame is too large")
    return encoded


def decode_frame(
    raw: bytes, *, max_bytes: int = DEFAULT_MAX_FRAME_BYTES
) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise TypeError("protocol frame must be bytes")
    if len(raw) > max_bytes:
        raise ProtocolFrameError("protocol frame is too large")
    if not raw.endswith(b"\n"):
        raise ProtocolFrameError("protocol frame must end with a newline")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: _reject_constant(token),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ProtocolFrameError("protocol frame is not valid JSON") from error
    if not isinstance(value, dict):
        raise ProtocolFrameError("protocol frame must be an object")
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


__all__ = ["DEFAULT_MAX_FRAME_BYTES", "ProtocolFrameError", "decode_frame", "encode_frame"]
