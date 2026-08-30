"""Tests for environment-backed mca configuration."""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mca.config import Config


class ConfigFromEnvTests(unittest.TestCase):
    def test_defaults_are_safe_for_non_live_usage(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env(require_api_key=False)

        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.model, "deepseek-v4-flash")
        self.assertIsNone(config.api_key)
        self.assertIsInstance(config.context_window, int)
        self.assertGreater(config.context_window, 0)
        self.assertEqual(config.context_window, 1_000_000)
        self.assertIsInstance(config.max_output_tokens, int)
        self.assertGreater(config.max_output_tokens, 0)
        self.assertEqual(config.max_output_tokens, 512_000)
        self.assertEqual(config.thinking, "enabled")
        self.assertEqual(config.request_max_output_tokens, 384_000)
        self.assertEqual(config.max_steps, 64)
        self.assertEqual(config.request_timeout, 600.0)
        self.assertEqual(config.retry_budget_seconds, 900.0)
        self.assertIsInstance(config.max_steps, int)
        self.assertGreater(config.max_steps, 0)
        self.assertIsInstance(config.max_tool_calls_per_batch, int)
        self.assertGreater(config.max_tool_calls_per_batch, 0)
        self.assertEqual(config.max_parallel_tool_calls, 4)
        self.assertIsInstance(config.request_timeout, float)
        self.assertGreater(config.request_timeout, 0)
        self.assertIsInstance(config.max_attempts, int)
        self.assertGreater(config.max_attempts, 0)
        self.assertIsInstance(config.retry_budget_seconds, float)
        self.assertGreater(config.retry_budget_seconds, 0)
        self.assertIs(config.verbose, False)
        self.assertIs(config.yolo, False)

    def test_live_mode_rejects_a_missing_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "API key"):
                Config.from_env()

    def test_mca_api_key_takes_precedence_over_deepseek_fallback(self) -> None:
        environment = {
            "MCA_API_KEY": "mca-secret",
            "DEEPSEEK_API_KEY": "deepseek-secret",
        }

        with patch.dict(os.environ, environment, clear=True):
            config = Config.from_env()

        self.assertEqual(config.api_key, "mca-secret")

    def test_deepseek_api_key_is_used_as_a_fallback(self) -> None:
        with patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "fallback-secret"}, clear=True
        ):
            config = Config.from_env()

        self.assertEqual(config.api_key, "fallback-secret")

    def test_context_window_is_read_as_an_integer(self) -> None:
        environment = {
            "MCA_API_KEY": "secret",
            "MCA_CONTEXT_WINDOW": "65536",
        }

        with patch.dict(os.environ, environment, clear=True):
            config = Config.from_env()

        self.assertEqual(config.context_window, 65_536)

    def test_context_window_rejects_zero(self) -> None:
        with patch.dict(
            os.environ, {"MCA_CONTEXT_WINDOW": "0"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "MCA_CONTEXT_WINDOW"):
                Config.from_env(require_api_key=False)

    def test_context_window_rejects_negative_values(self) -> None:
        with patch.dict(
            os.environ, {"MCA_CONTEXT_WINDOW": "-1"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "MCA_CONTEXT_WINDOW"):
                Config.from_env(require_api_key=False)

    def test_context_window_rejects_an_empty_value(self) -> None:
        with patch.dict(
            os.environ, {"MCA_CONTEXT_WINDOW": ""}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "MCA_CONTEXT_WINDOW"):
                Config.from_env(require_api_key=False)

    def test_context_window_rejects_a_non_integer(self) -> None:
        with patch.dict(
            os.environ, {"MCA_CONTEXT_WINDOW": "many"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "MCA_CONTEXT_WINDOW"):
                Config.from_env(require_api_key=False)

    def test_output_budget_and_thinking_mode_are_loaded_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MCA_MAX_OUTPUT_TOKENS": "98304",
                "MCA_THINKING": "enabled",
                "MCA_MAX_STEPS": "96",
                "MCA_MAX_PARALLEL_TOOL_CALLS": "3",
                "MCA_REQUEST_TIMEOUT": "750",
                "MCA_RETRY_BUDGET_SECONDS": "1200",
            },
            clear=True,
        ):
            config = Config.from_env(require_api_key=False)

        self.assertEqual(config.max_output_tokens, 98_304)
        self.assertEqual(config.thinking, "enabled")
        self.assertEqual(config.request_max_output_tokens, 98_304)
        self.assertEqual(config.max_steps, 96)
        self.assertEqual(config.max_parallel_tool_calls, 3)
        self.assertEqual(config.request_timeout, 750.0)
        self.assertEqual(config.retry_budget_seconds, 1200.0)

    def test_output_budget_and_thinking_mode_reject_invalid_values(self) -> None:
        cases = (
            ({"MCA_MAX_OUTPUT_TOKENS": "0"}, "MCA_MAX_OUTPUT_TOKENS"),
            ({"MCA_MAX_OUTPUT_TOKENS": "lots"}, "MCA_MAX_OUTPUT_TOKENS"),
            ({"MCA_THINKING": "sometimes"}, "MCA_THINKING"),
            ({"MCA_MAX_STEPS": "0"}, "MCA_MAX_STEPS"),
            ({"MCA_MAX_PARALLEL_TOOL_CALLS": "0"}, "MCA_MAX_PARALLEL_TOOL_CALLS"),
            ({"MCA_REQUEST_TIMEOUT": "0"}, "MCA_REQUEST_TIMEOUT"),
            ({"MCA_RETRY_BUDGET_SECONDS": "no"}, "MCA_RETRY_BUDGET_SECONDS"),
        )
        for environment, message in cases:
            with self.subTest(environment=environment):
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(ValueError, message):
                        Config.from_env(require_api_key=False)

    def test_repr_never_contains_the_api_key(self) -> None:
        secret = "unique-secret-that-must-not-leak"
        with patch.dict(os.environ, {"MCA_API_KEY": secret}, clear=True):
            config = Config.from_env()

        representation = repr(config)

        self.assertNotIn(secret, representation)
        self.assertIn("<redacted>", representation)

    def test_config_is_frozen(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env(require_api_key=False)

        with self.assertRaises(FrozenInstanceError):
            config.model = "another-model"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
