from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scanner import agent_detail_runtime


_DATASET_ROOT = Path(r"D:\IKA_DATA\ocr\raw\manual_screens\batch_001_ru\agent_detail")
_LIVE_PROBE_SAMPLE = Path(
    r"d:\Inter-Knot Arena\Inter-Knot Arena VerifierApp\artifacts\live_probe\step_base\after.png"
)
_BEN_SAMPLE = Path(r"D:\IKA_DATA\ocr\drops\batch_20260309_agents_129_255\agent_ben\agent_detail.png")
_BILLY_SAMPLE = Path(r"D:\IKA_DATA\ocr\drops\batch_20260309_agents_129_255\agent_billy\agent_detail.png")
_LUCY_SAMPLE = Path(r"D:\IKA_DATA\ocr\drops\batch_20260309_agents_129_255\agent_lucy\agent_detail.png")


def _sample_path(name: str) -> Path:
    return _DATASET_ROOT / name


class AgentDetailRuntimeTests(unittest.TestCase):
    def test_detect_mindscape_label_box_prefers_lower_left_candidate_over_top_noise(self) -> None:
        region = np.zeros((605, 717), dtype=np.uint8)
        region[0:158, 294:562] = 255
        region[426:538, 71:268] = 255
        region[375:466, 123:297] = 255

        label = agent_detail_runtime._detect_mindscape_label_box(region)

        self.assertIsNotNone(label)
        assert label is not None
        self.assertLess(label[0], 200)
        self.assertGreater(label[1], 300)

    @unittest.skipUnless(
        _sample_path("batch_001_ru_agent_detail_0c9e38651d02.png").exists(),
        "local OCR dataset sample is unavailable",
    )
    def test_read_agent_detail_parses_level_10_of_10_with_upgrade_button(self) -> None:
        image = cv2.imread(
            str(_sample_path("batch_001_ru_agent_detail_0c9e38651d02.png")),
            cv2.IMREAD_COLOR,
        )
        assert image is not None

        reading = agent_detail_runtime.read_agent_detail(image)

        self.assertEqual(reading.level, 10)
        self.assertEqual(reading.level_cap, 10)

    @unittest.skipUnless(
        _sample_path("batch_001_ru_agent_detail_08ca11c966bf.png").exists(),
        "local OCR dataset sample is unavailable",
    )
    def test_read_agent_detail_parses_level_10_of_20_with_upgrade_button(self) -> None:
        image = cv2.imread(
            str(_sample_path("batch_001_ru_agent_detail_08ca11c966bf.png")),
            cv2.IMREAD_COLOR,
        )
        assert image is not None

        reading = agent_detail_runtime.read_agent_detail(image)

        self.assertEqual(reading.level, 10)
        self.assertEqual(reading.level_cap, 20)

    @unittest.skipUnless(
        _sample_path("batch_001_ru_agent_detail_63314ce0a9cb.png").exists(),
        "local OCR dataset sample is unavailable",
    )
    def test_read_agent_detail_parses_level_01_of_10_with_upgrade_button(self) -> None:
        image = cv2.imread(
            str(_sample_path("batch_001_ru_agent_detail_63314ce0a9cb.png")),
            cv2.IMREAD_COLOR,
        )
        assert image is not None

        reading = agent_detail_runtime.read_agent_detail(image)

        self.assertEqual(reading.level, 1)
        self.assertEqual(reading.level_cap, 10)

    @unittest.skipUnless(
        _sample_path("batch_001_ru_agent_detail_6b392fdf5121.png").exists(),
        "local OCR dataset sample is unavailable",
    )
    def test_read_agent_detail_parses_level_11_of_20_with_upgrade_button(self) -> None:
        image = cv2.imread(
            str(_sample_path("batch_001_ru_agent_detail_6b392fdf5121.png")),
            cv2.IMREAD_COLOR,
        )
        assert image is not None

        reading = agent_detail_runtime.read_agent_detail(image)

        self.assertEqual(reading.level, 11)
        self.assertEqual(reading.level_cap, 20)

    @unittest.skipUnless(
        _sample_path("batch_001_ru_agent_detail_1976aa0d5ead.png").exists(),
        "local OCR dataset sample is unavailable",
    )
    def test_read_agent_detail_parses_level_49_of_50_with_upgrade_button(self) -> None:
        image = cv2.imread(
            str(_sample_path("batch_001_ru_agent_detail_1976aa0d5ead.png")),
            cv2.IMREAD_COLOR,
        )
        assert image is not None

        reading = agent_detail_runtime.read_agent_detail(image)

        self.assertEqual(reading.level, 49)
        self.assertEqual(reading.level_cap, 50)

    @unittest.skipUnless(_LIVE_PROBE_SAMPLE.exists(), "local live probe sample is unavailable")
    def test_read_agent_detail_keeps_maxed_sample_stable(self) -> None:
        image = cv2.imread(str(_LIVE_PROBE_SAMPLE), cv2.IMREAD_COLOR)
        assert image is not None

        reading = agent_detail_runtime.read_agent_detail(image)

        self.assertEqual(reading.level, 60)
        self.assertEqual(reading.level_cap, 60)

    @unittest.skipUnless(_BEN_SAMPLE.exists(), "local Ben sample is unavailable")
    def test_read_agent_detail_recovers_zero_mindscape_for_ben(self) -> None:
        image = cv2.imread(str(_BEN_SAMPLE), cv2.IMREAD_COLOR)
        assert image is not None

        reading = agent_detail_runtime.read_agent_detail(image)

        self.assertEqual(reading.mindscape, 0)
        self.assertEqual(reading.mindscape_cap, 6)
        self.assertGreaterEqual(reading.mindscape_confidence, 0.68)

    @unittest.skipUnless(_LUCY_SAMPLE.exists(), "local Lucy sample is unavailable")
    def test_read_agent_detail_recovers_zero_mindscape_for_lucy(self) -> None:
        image = cv2.imread(str(_LUCY_SAMPLE), cv2.IMREAD_COLOR)
        assert image is not None

        reading = agent_detail_runtime.read_agent_detail(image)

        self.assertEqual(reading.mindscape, 0)
        self.assertEqual(reading.mindscape_cap, 6)
        self.assertGreaterEqual(reading.mindscape_confidence, 0.68)

    @unittest.skipUnless(_BILLY_SAMPLE.exists(), "local Billy sample is unavailable")
    def test_read_agent_detail_recovers_billy_mindscape_three(self) -> None:
        image = cv2.imread(str(_BILLY_SAMPLE), cv2.IMREAD_COLOR)
        assert image is not None

        reading = agent_detail_runtime.read_agent_detail(image)

        self.assertEqual(reading.mindscape, 3)
        self.assertEqual(reading.mindscape_cap, 6)
        self.assertGreaterEqual(reading.mindscape_confidence, 0.6)

    def test_extract_stats_returns_partial_fields_with_degraded_confidence(self) -> None:
        specs = [
            {"source": "hp", "canonical": "hp_flat", "kind": "int", "min": 0, "max": 5000},
            {"source": "atk", "canonical": "attack_flat", "kind": "int", "min": 0, "max": 5000},
            {"source": "def", "canonical": "defense_flat", "kind": "int", "min": 0, "max": 5000},
            {"source": "impact", "canonical": "impact", "kind": "int", "min": 0, "max": 5000},
            {"source": "crit_rate", "canonical": "crit_rate_pct", "kind": "percent", "min": 0, "max": 100},
        ]

        def extract_block(_image: np.ndarray, _boxes: object, field_name: str) -> np.ndarray | None:
            if field_name == "crit_rate":
                return None
            return np.ones((16, 16), dtype=np.uint8)

        def parse_value(_block: np.ndarray, kind: str) -> tuple[int | float | None, float, str | None]:
            values = {
                "int": 1200,
                "percent": 8.0,
            }
            return values[kind], 0.9, None

        with (
            patch.object(agent_detail_runtime, "_STAT_SPECS", specs),
            patch.object(agent_detail_runtime, "_detect_level_boxes", return_value=((0, 0, 16, 16), (0, 0, 16, 16))),
            patch.object(agent_detail_runtime, "_extract_stat_value_block", side_effect=extract_block),
            patch.object(agent_detail_runtime, "_parse_stat_value", side_effect=parse_value),
        ):
            stats, confidence, reasons = agent_detail_runtime._extract_stats(np.zeros((32, 32), dtype=np.uint8))

        self.assertEqual(set(stats.keys()), {"hp_flat", "attack_flat", "defense_flat", "impact"})
        self.assertGreater(confidence, 0.0)
        self.assertLess(confidence, 0.9)
        self.assertIn("agent_detail_stats_insufficient_fields", reasons)


if __name__ == "__main__":
    unittest.main()
