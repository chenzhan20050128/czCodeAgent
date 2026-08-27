"""Tests for the minimal mca command-line entry points."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


class CliHelpTests(unittest.TestCase):
    def test_main_help_exits_zero_without_api_key(self) -> None:
        from mca.cli import main

        with patch.dict(os.environ, {}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["--help"])

        self.assertEqual(raised.exception.code, 0)

    def test_python_module_help_exits_zero_without_api_key(self) -> None:
        environment = os.environ.copy()
        environment.pop("MCA_API_KEY", None)
        environment.pop("DEEPSEEK_API_KEY", None)
        environment["PYTHONPATH"] = str(SRC_ROOT)

        result = subprocess.run(
            [sys.executable, "-m", "mca", "--help"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
