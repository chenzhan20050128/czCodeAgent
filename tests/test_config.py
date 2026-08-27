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
        self.assertIsInstance(config.max_output_tokens, int)
        self.assertGreater(config.max_output_tokens, 0)
        self.assertIsInstance(config.max_steps, int)
        self.assertGreater(config.max_steps, 0)
        self.assertIsInstance(config.max_tool_calls_per_batch, int)
        self.assertGreater(config.max_tool_calls_per_batch, 0)
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
