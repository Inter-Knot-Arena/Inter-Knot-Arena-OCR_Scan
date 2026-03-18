from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scanner import screen_runtime


class ScreenRuntimeTests(unittest.TestCase):
    def test_normalize_runtime_captures_derives_multi_page_roster_icons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            first_page = temp_root / "roster_page_0.png"
            second_page = temp_root / "roster_page_1.png"
            square_page = temp_root / "square.png"

            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            square = np.zeros((1024, 1024, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(first_page), image))
            self.assertTrue(cv2.imwrite(str(second_page), image))
            self.assertTrue(cv2.imwrite(str(square_page), square))

            normalized = screen_runtime.normalize_runtime_captures(
                {
                    "sessionId": "multi-page-normalization",
                    "anchors": {"profile": False, "agents": False, "equipment": False},
                    "screenCaptures": [
                        {"role": "roster", "path": str(first_page), "screenAlias": "page_zero", "pageIndex": 0},
                        {"role": "roster", "path": str(second_page), "screenAlias": "page_one", "pageIndex": 1},
                    ],
                },
                "1080p",
            )

            derived_icons = normalized["agentIconPaths"]
            self.assertEqual(len(derived_icons), 6)
            self.assertEqual([entry["agentSlotIndex"] for entry in derived_icons], [1, 2, 3, 4, 5, 6])
            self.assertEqual([entry["pageIndex"] for entry in derived_icons], [0, 0, 0, 1, 1, 1])
            self.assertEqual(normalized["anchors"], {"profile": False, "agents": False, "equipment": False})

            resolution = screen_runtime.normalize_runtime_resolution(
                {
                    "screenCaptures": [
                        {"role": "roster", "path": str(square_page)},
                    ]
                },
                "1080p",
            )
            self.assertEqual(resolution, "")


if __name__ == "__main__":
    unittest.main()
