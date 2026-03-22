from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scanner import pipeline


_EMPTY_WEAPON_OVERVIEW_SAMPLE = Path(
    r"d:\Inter-Knot Arena\Inter-Knot Arena VerifierApp\artifacts\live_capture_mirror\20260322_004059\screen_captures\1dad9ca6ffc24b1e894810feea660407-page-06\02_equipment_agent_slot_1_page_06_agent_1_equipment.png"
)
_WEAPON_ONLY_OVERVIEW_SAMPLE = Path(
    r"d:\Inter-Knot Arena\Inter-Knot Arena VerifierApp\artifacts\live_capture_mirror\20260322_004059\screen_captures\1dad9ca6ffc24b1e894810feea660407-page-07\20_equipment_agent_slot_2_page_07_agent_2_equipment.png"
)


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

    def test_equipment_overview_occupancy_keeps_borderline_weapon_patch_ambiguous_when_all_discs_are_empty(self) -> None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)

        with patch.object(
            pipeline,
            "_presence_from_patch",
            side_effect=[
                (True, 0.61),
                (False, 0.91),
                (False, 0.92),
                (False, 0.93),
                (False, 0.94),
                (False, 0.95),
                (False, 0.96),
            ],
        ):
            occupancy, reasons = pipeline._derive_equipment_overview_occupancy_from_image(image)

        self.assertEqual(reasons, ["equipment_overview_weapon_presence_ambiguous"])
        self.assertNotIn("weaponPresent", occupancy)
        self.assertEqual(
            occupancy["discSlotOccupancy"],
            {str(slot): False for slot in range(1, 7)},
        )

    def test_equipment_overview_occupancy_keeps_borderline_weapon_patch_when_any_disc_is_equipped(self) -> None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)

        with patch.object(
            pipeline,
            "_presence_from_patch",
            side_effect=[
                (True, 0.61),
                (False, 0.91),
                (False, 0.92),
                (True, 0.93),
                (False, 0.94),
                (False, 0.95),
                (False, 0.96),
            ],
        ):
            occupancy, reasons = pipeline._derive_equipment_overview_occupancy_from_image(image)

        self.assertEqual(reasons, [])
        self.assertTrue(occupancy["weaponPresent"])
        self.assertTrue(occupancy["discSlotOccupancy"]["3"])

    def test_equipment_overview_occupancy_keeps_borderline_empty_weapon_patch_ambiguous_when_all_discs_are_empty(self) -> None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)

        with patch.object(
            pipeline,
            "_presence_from_patch",
            side_effect=[
                (False, 0.71),
                (False, 0.91),
                (False, 0.92),
                (False, 0.93),
                (False, 0.94),
                (False, 0.95),
                (False, 0.96),
            ],
        ):
            occupancy, reasons = pipeline._derive_equipment_overview_occupancy_from_image(image)

        self.assertEqual(reasons, ["equipment_overview_weapon_presence_ambiguous"])
        self.assertNotIn("weaponPresent", occupancy)
        self.assertEqual(
            occupancy["discSlotOccupancy"],
            {str(slot): False for slot in range(1, 7)},
        )

    def test_equipment_overview_occupancy_keeps_confident_empty_weapon_patch_when_all_discs_are_empty(self) -> None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)

        with patch.object(
            pipeline,
            "_presence_from_patch",
            side_effect=[
                (False, 0.88),
                (False, 0.91),
                (False, 0.92),
                (False, 0.93),
                (False, 0.94),
                (False, 0.95),
                (False, 0.96),
            ],
        ):
            occupancy, reasons = pipeline._derive_equipment_overview_occupancy_from_image(image)

        self.assertEqual(reasons, [])
        self.assertFalse(occupancy["weaponPresent"])
        self.assertEqual(
            occupancy["discSlotOccupancy"],
            {str(slot): False for slot in range(1, 7)},
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

    def test_all_disc_slots_empty_requires_complete_contract(self) -> None:
        self.assertFalse(pipeline._all_disc_slots_empty({"1": False, "2": False}))
        self.assertTrue(pipeline._all_disc_slots_empty({str(slot): False for slot in range(1, 7)}))

    def test_enrich_agents_with_pixel_equipment_occupancy_marks_known_empty_weapon_as_confident(self) -> None:
        agent = pipeline._default_agent_payload("agent_anby", 0.97)

        merged_agents, used = pipeline._enrich_agents_with_pixel_equipment_occupancy(
            [agent],
            {
                "agent_anby": {
                    "weaponPresent": False,
                    "discSlotOccupancy": {str(slot): True for slot in range(1, 7)},
                    "_weaponConfidence": 0.8459,
                    "_discConfidence": 0.91,
                }
            },
        )

        self.assertTrue(used)
        self.assertFalse(merged_agents[0]["weaponPresent"])
        self.assertEqual(merged_agents[0]["fieldSources"]["weapon"], "known_empty_from_equipment_occupancy")
        self.assertGreaterEqual(merged_agents[0]["confidenceByField"]["weapon"], 0.9)

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

    def test_enrich_agents_with_pixel_discs_boosts_consistent_full_set_confidence(self) -> None:
        agent = pipeline._default_agent_payload("agent_anby", 0.97)
        pixel_discs = {
            "agent_anby": [
                {
                    "slot": slot,
                    "setId": "set_fanged_metal",
                    "_confidence": 0.31,
                }
                for slot in range(1, 7)
            ]
        }

        merged_agents, used = pipeline._enrich_agents_with_pixel_discs([agent], pixel_discs)

        self.assertTrue(used)
        self.assertEqual(merged_agents[0]["confidenceByField"]["discs"], 0.92)
        self.assertEqual(merged_agents[0]["confidenceByField"]["occupancy"], 0.92)

    def test_filter_resolved_low_conf_reasons_drops_disk_low_conf_for_consistent_full_set(self) -> None:
        filtered = pipeline._filter_resolved_low_conf_reasons(
            [
                {
                    "agentId": "agent_anby",
                    "confidenceByField": {"discs": 0.92},
                    "discs": [
                        {"slot": slot, "setId": "set_fanged_metal"}
                        for slot in range(1, 7)
                    ],
                }
            ],
            [
                "disk_detail_low_conf:agent_anby:1:set_fanged_metal",
                "disk_detail_low_conf:agent_anby:2:set_fanged_metal",
            ],
        )

        self.assertEqual(filtered, [])

    def test_filter_resolved_low_conf_reasons_drops_resolved_equipment_overview_noise(self) -> None:
        filtered = pipeline._filter_resolved_low_conf_reasons(
            [
                {
                    "agentId": "agent_anby",
                    "weaponPresent": True,
                    "weapon": {"weaponId": "amp_ii"},
                    "discSlotOccupancy": {str(slot): True for slot in range(1, 7)},
                    "discs": [
                        {"slot": slot, "setId": "set_fanged_metal"}
                        for slot in range(1, 7)
                    ],
                }
            ],
            [
                "agents_empty_after_parse",
                "agent_anby:equipment_overview_weapon_presence_ambiguous",
                "agent_anby:equipment_overview_slot_ambiguous:3",
            ],
        )

        self.assertEqual(filtered, [])

    def test_filter_resolved_low_conf_reasons_drops_amplifier_unclassified_for_known_empty_weapon(self) -> None:
        filtered = pipeline._filter_resolved_low_conf_reasons(
            [
                {
                    "agentId": "agent_anby",
                    "weaponPresent": False,
                    "weapon": {},
                }
            ],
            ["amplifier_detail_unclassified:agent_anby"],
        )

        self.assertEqual(filtered, [])

    def test_pixel_weapons_from_captures_skips_known_empty_weapon_slots(self) -> None:
        with patch.object(
            pipeline,
            "_resolve_capture_agent_id",
            return_value=("agent_anby", "screen_capture_agent_id", 0.99),
        ):
            weapons, reasons = pipeline._pixel_weapons_from_captures(
                {
                    "screenCaptures": [
                        {
                            "role": "amplifier_detail",
                            "path": "ignored.png",
                            "agentSlotIndex": 1,
                        }
                    ]
                },
                {},
                {"agent_anby": {"weaponPresent": False}},
                "RU",
            )

        self.assertEqual(weapons, {})
        self.assertEqual(reasons, [])

    def test_pixel_discs_from_captures_uses_title_ocr_fallback_for_low_conf_predictions(self) -> None:
        prediction = type("Prediction", (), {"label": "set_astral_voice", "confidence": 0.61})()

        with (
            patch.object(pipeline.ModelRegistry, "has_disk_model", return_value=True),
            patch.object(pipeline, "_resolve_capture_agent_id", return_value=("agent_anby", "screen_capture_agent_id", 0.99)),
            patch.object(pipeline.cv2, "imread", return_value=np.zeros((32, 32, 3), dtype=np.uint8)),
            patch.object(pipeline, "classify_disk_detail", return_value=prediction),
            patch.object(pipeline, "crop_title_image", return_value=None),
            patch.object(
                pipeline,
                "run_winrt_ocr_batch",
                return_value={"disk_0_agent_anby_1:title": "Фридом-блюз [1] О Ур. 15/15 Базовый параметр"},
            ),
        ):
            discs, reasons = pipeline._pixel_discs_from_captures(
                {
                    "screenCaptures": [
                        {
                            "role": "disk_detail",
                            "path": "ignored.png",
                            "slotIndex": 1,
                            "agentSlotIndex": 1,
                        }
                    ]
                },
                {},
                {},
                "RU",
            )

        self.assertEqual(reasons, [])
        self.assertEqual(discs["agent_anby"][0]["setId"], "set_freedom_blues")
        self.assertGreaterEqual(discs["agent_anby"][0]["_confidence"], 0.9)

    def test_pixel_discs_from_captures_skips_known_empty_slots(self) -> None:
        with (
            patch.object(pipeline.ModelRegistry, "has_disk_model", return_value=True),
            patch.object(pipeline, "_resolve_capture_agent_id", return_value=("agent_anby", "screen_capture_agent_id", 0.99)),
        ):
            discs, reasons = pipeline._pixel_discs_from_captures(
                {
                    "screenCaptures": [
                        {
                            "role": "disk_detail",
                            "path": "ignored.png",
                            "slotIndex": 2,
                            "agentSlotIndex": 1,
                        }
                    ]
                },
                {},
                {"agent_anby": {"discSlotOccupancy": {"2": False}}},
                "RU",
            )

        self.assertEqual(discs, {})
        self.assertEqual(reasons, [])

    def test_inspect_equipment_capture_returns_occupancy_snapshot(self) -> None:
        with (
            patch.object(pipeline.cv2, "imread", return_value=np.zeros((32, 32, 3), dtype=np.uint8)),
            patch.object(
                pipeline,
                "_derive_equipment_overview_occupancy_from_image",
                return_value=(
                    {
                        "weaponPresent": False,
                        "discSlotOccupancy": {"1": True, "2": False, "3": True, "4": False, "5": True, "6": False},
                        "_weaponConfidence": 0.91,
                        "_discConfidence": 0.87,
                    },
                    [],
                ),
            ),
        ):
            inspection = pipeline.inspect_equipment_capture("ignored.png")

        self.assertFalse(inspection["weaponPresent"])
        self.assertEqual(inspection["discSlotOccupancy"]["1"], True)
        self.assertEqual(inspection["discSlotOccupancy"]["2"], False)
        self.assertEqual(inspection["confidence"], 0.91)
        self.assertEqual(inspection["lowConfReasons"], [])

    def test_equipment_overview_occupancy_keeps_confident_slots_when_one_slot_is_ambiguous(self) -> None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)

        with patch.object(
            pipeline,
            "_presence_from_patch",
            side_effect=[
                (True, 0.83),
                (True, 0.91),
                (False, 0.92),
                (None, 0.52),
                (False, 0.94),
                (True, 0.88),
                (False, 0.9),
            ],
        ):
            occupancy, reasons = pipeline._derive_equipment_overview_occupancy_from_image(image)

        self.assertEqual(reasons, ["equipment_overview_slot_ambiguous:3"])
        self.assertTrue(occupancy["weaponPresent"])
        self.assertEqual(
            occupancy["discSlotOccupancy"],
            {"1": True, "2": False, "4": False, "5": True, "6": False},
        )

    def test_inspect_equipment_capture_infers_empty_weapon_when_all_slots_are_empty(self) -> None:
        inspection = pipeline.inspect_equipment_capture(_EMPTY_WEAPON_OVERVIEW_SAMPLE)

        self.assertFalse(inspection["weaponPresent"])
        self.assertEqual(inspection["discSlotOccupancy"], {str(slot): False for slot in range(1, 7)})
        self.assertNotIn("equipment_overview_weapon_presence_ambiguous", inspection["lowConfReasons"])

    def test_inspect_equipment_capture_keeps_weapon_only_agent_equipped(self) -> None:
        inspection = pipeline.inspect_equipment_capture(_WEAPON_ONLY_OVERVIEW_SAMPLE)

        self.assertTrue(inspection["weaponPresent"])
        self.assertEqual(inspection["discSlotOccupancy"], {str(slot): False for slot in range(1, 7)})

    def test_drop_stale_top_level_confidence_reasons_respects_current_confidence(self) -> None:
        filtered = pipeline._drop_stale_top_level_confidence_reasons(
            ["equipment_low_confidence", "uid_low_confidence"],
            {"equipment": 0.9343, "uid": 0.88},
        )

        self.assertEqual(filtered, ["uid_low_confidence"])

    def test_enrich_agents_with_agent_detail_pixels_accepts_near_threshold_level_confidence(self) -> None:
        agent = pipeline._default_agent_payload("agent_anby", 0.97)
        agent["level"] = None
        agent["levelCap"] = None

        merged_agents, used = pipeline._enrich_agents_with_agent_detail_pixels(
            [agent],
            {
                "agent_anby": {
                    "agentId": "agent_anby",
                    "agentSource": "screen_capture_agent_id",
                    "agentConfidence": 0.99,
                    "level": 11,
                    "levelCap": 20,
                    "levelConfidence": 0.7871,
                    "mindscape": None,
                    "mindscapeCap": None,
                    "mindscapeConfidence": 0.0,
                    "stats": {},
                    "statsConfidence": 0.0,
                }
            },
            {},
        )

        self.assertTrue(used)
        self.assertEqual(merged_agents[0]["level"], 11)
        self.assertEqual(merged_agents[0]["levelCap"], 20)
        self.assertEqual(merged_agents[0]["fieldSources"]["level"], "agent_detail_digit_ocr")
        self.assertEqual(merged_agents[0]["confidenceByField"]["level"], 0.7871)

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
