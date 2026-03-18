from __future__ import annotations

import unittest

from scanner import pipeline


class PipelineTests(unittest.TestCase):
    def test_merge_agents_prefers_icon_identity_and_order(self) -> None:
        parsed_agents, _ = pipeline._extract_agents(
            [
                {"agentId": "agent_anby", "level": 60, "levelCap": 60},
                {"agentId": "agent_nicole", "level": 50, "levelCap": 60},
            ],
            {},
        )
        icon_agents = [
            pipeline._default_agent_payload("agent_nicole", 0.97),
            pipeline._default_agent_payload("agent_anby", 0.96),
        ]

        merged, reasons = pipeline._merge_agents(parsed_agents, icon_agents)

        self.assertEqual([agent["agentId"] for agent in merged], ["agent_nicole", "agent_anby"])
        self.assertEqual(reasons, [])
        self.assertEqual(merged[0]["fieldSources"]["agentId"], "onnx_agent_icon")
        self.assertEqual(merged[1]["fieldSources"]["agentId"], "onnx_agent_icon")

    def test_scan_roster_ignores_uid_candidates_in_strict_mode(self) -> None:
        with self.assertRaises(pipeline.ScanFailure) as exc_info:
            pipeline.scan_roster(
                session_context={
                    "sessionId": "strict-no-uid",
                    "inputLockActive": True,
                    "anchors": {"profile": True, "agents": True, "equipment": True},
                    "uidCandidates": ["123456789"],
                    "agents": [{"agentId": "agent_anby"}],
                },
                calibration={"requiredAnchors": ["profile", "agents", "equipment"]},
                locale="EN",
                resolution="1080p",
            )

        exc = exc_info.exception
        self.assertEqual(exc.code, pipeline.ScanFailureCode.LOW_CONFIDENCE)
        self.assertIn("uid_missing", exc.low_conf_reasons)
        self.assertIsInstance(exc.partial_result, dict)
        self.assertEqual(exc.partial_result["uid"], "")
        self.assertEqual(exc.partial_result["fieldSources"]["uid"], "missing")

    def test_infer_full_roster_coverage_requires_terminal_partial_page(self) -> None:
        terminal_context = {
            "screenCaptures": [
                {"role": "roster", "path": "page0.png", "pageIndex": 0},
                {"role": "roster", "path": "page1.png", "pageIndex": 1},
            ],
            "agentIconPaths": [
                {"path": "icon_1.png", "pageIndex": 0, "agentSlotIndex": 1, "rosterPageSlotIndex": 1},
                {"path": "icon_2.png", "pageIndex": 0, "agentSlotIndex": 2, "rosterPageSlotIndex": 2},
                {"path": "icon_3.png", "pageIndex": 0, "agentSlotIndex": 3, "rosterPageSlotIndex": 3},
                {"path": "icon_4.png", "pageIndex": 1, "agentSlotIndex": 4, "rosterPageSlotIndex": 1},
                {"path": "icon_5.png", "pageIndex": 1, "agentSlotIndex": 5, "rosterPageSlotIndex": 2},
            ],
        }
        non_terminal_context = {
            "screenCaptures": [
                {"role": "roster", "path": "page0.png", "pageIndex": 0},
                {"role": "roster", "path": "page1.png", "pageIndex": 1},
            ],
            "agentIconPaths": [
                {"path": "icon_1.png", "pageIndex": 0, "agentSlotIndex": 1, "rosterPageSlotIndex": 1},
                {"path": "icon_2.png", "pageIndex": 0, "agentSlotIndex": 2, "rosterPageSlotIndex": 2},
                {"path": "icon_3.png", "pageIndex": 0, "agentSlotIndex": 3, "rosterPageSlotIndex": 3},
                {"path": "icon_4.png", "pageIndex": 1, "agentSlotIndex": 4, "rosterPageSlotIndex": 1},
                {"path": "icon_5.png", "pageIndex": 1, "agentSlotIndex": 5, "rosterPageSlotIndex": 2},
                {"path": "icon_6.png", "pageIndex": 1, "agentSlotIndex": 6, "rosterPageSlotIndex": 3},
            ],
        }

        self.assertTrue(pipeline._infer_full_roster_coverage(terminal_context))
        self.assertFalse(pipeline._infer_full_roster_coverage(non_terminal_context))


if __name__ == "__main__":
    unittest.main()
