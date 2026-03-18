from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class StrictCliTests(unittest.TestCase):
    def test_strict_cli_no_longer_returns_fake_success_without_pixels(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "run_scan.py"

        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--input-lock",
                "--anchor-profile",
                "--anchor-agents",
                "--anchor-equipment",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["code"], "LOW_CONFIDENCE")
        self.assertEqual(payload["uid"], "")
        self.assertIn("uid_missing", payload["lowConfReasons"])


if __name__ == "__main__":
    unittest.main()
