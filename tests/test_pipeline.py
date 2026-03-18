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
        pipeline._copy_visible_slot_metadata(icon_agents[0], {"pageIndex": 0, "agentSlotIndex": 1})
        pipeline._copy_visible_slot_metadata(icon_agents[1], {"pageIndex": 1, "agentSlotIndex": 1})

        merged, reasons = pipeline._merge_agents(parsed_agents, icon_agents)

        self.assertEqual([agent["agentId"] for agent in merged], ["agent_nicole", "agent_anby"])
        self.assertEqual(reasons, [])
        self.assertEqual(merged[0]["fieldSources"]["agentId"], "onnx_agent_icon")
        self.assertEqual(merged[1]["fieldSources"]["agentId"], "onnx_agent_icon")
        self.assertEqual(merged[0]["_pageIndex"], 0)
        self.assertEqual(merged[0]["_agentSlotIndex"], 1)
        self.assertEqual(merged[1]["_pageIndex"], 1)
        self.assertEqual(merged[1]["_agentSlotIndex"], 1)

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

    def test_infer_full_roster_coverage_only_respects_explicit_terminal_signal(self) -> None:
        self.assertTrue(pipeline._infer_full_roster_coverage({"fullRosterCoverage": True}))
        self.assertTrue(pipeline._infer_full_roster_coverage({"fullRosterTerminalSliceReached": True}))
        self.assertTrue(pipeline._infer_full_roster_coverage({"terminalSliceReached": True}))
        self.assertFalse(pipeline._infer_full_roster_coverage({}))
        self.assertFalse(pipeline._infer_full_roster_coverage({"fullRosterCoverage": False}))

    def test_visible_slot_key_preserves_first_page_index(self) -> None:
        self.assertEqual(pipeline._visible_slot_key(0, 1), (0, 1))
        payload: dict[str, object] = {}
        pipeline._copy_visible_slot_metadata(payload, {"pageIndex": 0, "agentSlotIndex": 2})
        self.assertEqual(payload["_pageIndex"], 0)
        self.assertEqual(payload["_agentSlotIndex"], 2)

    def test_lookup_agent_id_by_slot_is_page_aware(self) -> None:
        slot_to_agent = {
            (0, 1): "agent_anby",
            (1, 1): "agent_nicole",
        }

        self.assertEqual(pipeline._lookup_agent_id_by_slot(slot_to_agent, 0, 1), "agent_anby")
        self.assertEqual(pipeline._lookup_agent_id_by_slot(slot_to_agent, 1, 1), "agent_nicole")
        self.assertEqual(pipeline._lookup_agent_id_by_slot(slot_to_agent, None, 1), "")


if __name__ == "__main__":
    unittest.main()
