from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            self.assertEqual([entry["agentSlotIndex"] for entry in derived_icons], [1, 2, 3, 1, 2, 3])
            self.assertEqual([entry["pageIndex"] for entry in derived_icons], [0, 0, 0, 1, 1, 1])
            self.assertEqual(normalized["anchors"], {"profile": False, "agents": True, "equipment": False})

            resolution = screen_runtime.normalize_runtime_resolution(
                {
                    "screenCaptures": [
                        {"role": "roster", "path": str(square_page)},
                    ]
                },
                "1080p",
            )
            self.assertEqual(resolution, "")

    def test_normalize_runtime_captures_sets_profile_anchor_from_derived_uid_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            agent_detail = temp_root / "agent_detail.png"
            equipment = temp_root / "equipment.png"
            derived_uid = temp_root / "derived_uid.png"

            image = np.zeros((1440, 2560, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(agent_detail), image))
            self.assertTrue(cv2.imwrite(str(equipment), image))
            self.assertTrue(cv2.imwrite(str(derived_uid), np.zeros((32, 240, 3), dtype=np.uint8)))

            with patch.object(screen_runtime, "_derive_uid_from_capture", return_value=str(derived_uid)):
                normalized = screen_runtime.normalize_runtime_captures(
                    {
                        "sessionId": "derived-uid-normalization",
                        "anchors": {"profile": False, "agents": False, "equipment": False},
                        "screenCaptures": [
                            {"role": "agent_detail", "path": str(agent_detail), "screenAlias": "detail"},
                            {"role": "equipment", "path": str(equipment), "screenAlias": "equipment"},
                        ],
                    },
                    "1440p",
                )

            self.assertEqual(normalized["uidImagePath"], str(derived_uid))
            self.assertEqual(normalized["anchors"], {"profile": True, "agents": True, "equipment": True})


if __name__ == "__main__":
    unittest.main()
