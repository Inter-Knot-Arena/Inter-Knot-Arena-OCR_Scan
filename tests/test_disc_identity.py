from __future__ import annotations

import unittest

from disc_identity import classify_disc_title, extract_disk_title_candidate


class DiscIdentityTests(unittest.TestCase):
    def test_extract_disk_title_candidate_strips_runtime_noise(self) -> None:
        candidate = extract_disk_title_candidate(
            "15 \u0421\u0432\u0438\u0440\u0435\u043f\u044b\u0439 \u0445\u044d\u0432\u0438-\u043c\u0435\u0442\u0430\u043b [5] \u041e \u0423\u0440. 15/15 "
            "\u0411\u0430\u0437\u043e\u0432\u044b\u0439 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440 \u0411\u043e\u043d\u0443\u0441 \u0444\u0438\u0437\u0438\u0447\u0435\u0441\u043a\u043e\u0433\u043e 300/0 "
            "\u0423\u0440\u043e\u043d\u0430 \u0434\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b"
        )

        self.assertEqual(candidate, "\u0421\u0432\u0438\u0440\u0435\u043f\u044b\u0439 \u0445\u044d\u0432\u0438-\u043c\u0435\u0442\u0430\u043b \u041e")

    def test_classify_disc_title_maps_russian_alias(self) -> None:
        prediction = classify_disc_title(
            "\u0424\u0440\u0438\u0434\u043e\u043c-\u0431\u043b\u044e\u0437 [6] \u041e \u0423\u0440. 15/15 "
            "\u0411\u0430\u0437\u043e\u0432\u044b\u0439 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440 \u0421\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438"
        )

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.set_id, "set_freedom_blues")
        self.assertEqual(prediction.display_name, "Freedom Blues")
        self.assertGreaterEqual(prediction.confidence, 0.9)

    def test_classify_disc_title_maps_common_ocr_misspelling(self) -> None:
        prediction = classify_disc_title("astrai voice [3] lv. 15/15 main stat")

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.set_id, "set_astral_voice")

    def test_classify_disc_title_maps_russian_fuzzy_alias(self) -> None:
        prediction = classify_disc_title(
            "\u0421\u0432\u0438\u0440\u0435\u043f\u044b\u0439 \u0445\u044d\u0432\u0438-\u043c\u0435\u0442\u0441\u043b [2] \u041e \u0423\u0440. 15/15 "
            "\u0411\u0430\u0437\u043e\u0432\u044b\u0439 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440"
        )

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.set_id, "set_fanged_metal")


if __name__ == "__main__":
    unittest.main()
