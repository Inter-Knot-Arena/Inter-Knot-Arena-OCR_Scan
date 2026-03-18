from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

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

    def test_equipment_overview_occupancy_detects_synthetic_slots(self) -> None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        rng = np.random.default_rng(42)

        def fill_patch(center: tuple[float, float], patch_size: tuple[float, float]) -> None:
            patch = pipeline._fractional_patch_from_center(
                image,
                center_x=center[0],
                center_y=center[1],
                width_fraction=patch_size[0] * 0.72,
                height_fraction=patch_size[1] * 0.72,
            )
            self.assertIsNotNone(patch)
            assert patch is not None
            patch[:] = rng.integers(32, 255, size=patch.shape, dtype=np.uint8)

        fill_patch(pipeline._EQUIPMENT_WEAPON_CENTER, pipeline._EQUIPMENT_WEAPON_PATCH)
        fill_patch(pipeline._EQUIPMENT_DISC_SLOT_CENTERS[1], pipeline._EQUIPMENT_DISC_PATCH)
        fill_patch(pipeline._EQUIPMENT_DISC_SLOT_CENTERS[3], pipeline._EQUIPMENT_DISC_PATCH)
        fill_patch(pipeline._EQUIPMENT_DISC_SLOT_CENTERS[5], pipeline._EQUIPMENT_DISC_PATCH)

        occupancy, reasons = pipeline._derive_equipment_overview_occupancy_from_image(image)

        self.assertEqual(reasons, [])
        self.assertTrue(occupancy["weaponPresent"])
        self.assertEqual(
            occupancy["discSlotOccupancy"],
            {"1": True, "2": False, "3": True, "4": False, "5": True, "6": False},
        )

    def test_filter_resolved_low_conf_reasons_respects_known_empty_equipment(self) -> None:
        filtered = pipeline._filter_resolved_low_conf_reasons(
            [
                {
                    "agentId": "agent_anby",
                    "weaponPresent": False,
                    "discSlotOccupancy": {str(slot): False for slot in range(1, 7)},
                    "weapon": {},
                    "discs": [],
                }
            ],
            ["agent_anby.weapon_missing", "agent_anby.discs_missing"],
        )

        self.assertEqual(filtered, [])

    def test_enrich_agents_with_pixel_weapons_merges_detail_fields_into_existing_weapon(self) -> None:
        agent = pipeline._default_agent_payload("agent_anby", 0.97)
        agent["weapon"] = {"weaponId": "amp_deep_sea_visitor"}
        agent["weaponPresent"] = True
        agent["fieldSources"]["weapon"] = "session_payload"
        agent["fieldSources"]["weaponPresent"] = "derived_from_weapon_payload"
        agent["confidenceByField"]["weapon"] = 0.41

        merged_agents, used = pipeline._enrich_agents_with_pixel_weapons(
            [agent],
            {
                "agent_anby": {
                    "weaponId": "amp_deep_sea_visitor",
                    "displayName": "Deep Sea Visitor",
                    "level": 60,
                    "levelCap": 60,
                    "baseStatKey": "attack_flat",
                    "baseStatValue": 713,
                    "advancedStatKey": "crit_rate_pct",
                    "advancedStatValue": 24.0,
                    "_confidence": 0.995,
                }
            },
        )

        self.assertTrue(used)
        weapon = merged_agents[0]["weapon"]
        self.assertEqual(weapon["weaponId"], "amp_deep_sea_visitor")
        self.assertEqual(weapon["displayName"], "Deep Sea Visitor")
        self.assertEqual(weapon["level"], 60)
        self.assertEqual(weapon["levelCap"], 60)
        self.assertEqual(weapon["baseStatKey"], "attack_flat")
        self.assertEqual(weapon["baseStatValue"], 713)
        self.assertEqual(weapon["advancedStatKey"], "crit_rate_pct")
        self.assertEqual(weapon["advancedStatValue"], 24.0)
        self.assertEqual(weapon["agentId"], "agent_anby")
        self.assertTrue(merged_agents[0]["weaponPresent"])
        self.assertEqual(merged_agents[0]["fieldSources"]["weapon"], "amplifier_detail_ocr")
        self.assertEqual(merged_agents[0]["fieldSources"]["weaponPresent"], "derived_from_amplifier_detail_ocr")
        self.assertEqual(merged_agents[0]["confidenceByField"]["weapon"], 0.995)

    def test_pixel_equipment_occupancy_preserves_page_index_without_name_error(self) -> None:
        occupancy = {
            "weaponPresent": True,
            "discSlotOccupancy": {str(slot): False for slot in range(1, 7)},
            "_weaponConfidence": 0.91,
            "_discConfidence": 0.87,
        }

        with (
            patch.object(pipeline, "_resolve_capture_agent_id", return_value=("agent_anby", "screen_capture_agent_id", 0.99)),
            patch.object(pipeline.cv2, "imread", return_value=np.zeros((32, 32, 3), dtype=np.uint8)),
            patch.object(pipeline, "_derive_equipment_overview_occupancy_from_image", return_value=(occupancy, [])),
        ):
            by_agent, reasons = pipeline._pixel_equipment_occupancy_from_captures(
                {
                    "screenCaptures": [
                        {
                            "role": "equipment",
                            "path": "ignored.png",
                            "pageIndex": 1,
                            "agentSlotIndex": 2,
                        }
                    ]
                },
                {},
            )

        self.assertEqual(reasons, [])
        self.assertIn("agent_anby", by_agent)
        self.assertEqual(by_agent["agent_anby"]["pageIndex"], 1)
        self.assertEqual(by_agent["agent_anby"]["agentSlotIndex"], 2)


if __name__ == "__main__":
    unittest.main()
