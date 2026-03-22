from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from amplifier_identity import (
    crop_advanced_stat_image,
    crop_effect_image,
    crop_info_image,
    crop_title_image,
    language_tag_for_locale,
    normalize_alias_key,
    parse_amplifier_detail,
    run_winrt_ocr_batch,
)


_ELLEN_LIVE_AMP_SAMPLE = Path(
    r"d:\Inter-Knot Arena\Inter-Knot Arena VerifierApp\artifacts\live_capture_mirror\20260322_004059\screen_captures\1dad9ca6ffc24b1e894810feea660407-page-01\03_amplifier_detail_agent_slot_1_page_01_agent_1_amplifier.png"
)
_HARUMASA_LIVE_AMP_SAMPLE = Path(
    r"d:\Inter-Knot Arena\Inter-Knot Arena VerifierApp\artifacts\live_capture_mirror\20260322_004059\screen_captures\1dad9ca6ffc24b1e894810feea660407-page-03\39_amplifier_detail_agent_slot_3_page_03_agent_3_amplifier.png"
)


class AmplifierIdentityTests(unittest.TestCase):
    def _read_live_sample(self, source_path: Path) -> object:
        with tempfile.TemporaryDirectory(prefix="amp_identity_test_") as raw_tmp:
            temp_root = Path(raw_tmp)
            title_path = temp_root / "title.png"
            info_path = temp_root / "info.png"
            advanced_path = temp_root / "advanced.png"
            effect_path = temp_root / "effect.png"
            crop_title_image(source_path, title_path)
            crop_info_image(source_path, info_path)
            crop_advanced_stat_image(source_path, advanced_path)
            crop_effect_image(source_path, effect_path)
            ocr_results = run_winrt_ocr_batch(
                [
                    {"id": "title", "path": str(title_path)},
                    {"id": "info", "path": str(info_path)},
                    {"id": "advanced", "path": str(advanced_path)},
                    {"id": "effect", "path": str(effect_path)},
                ],
                language_tag=language_tag_for_locale("RU"),
                temp_root=temp_root,
            )
        return parse_amplifier_detail(
            str(ocr_results.get("title") or ""),
            info_text=str(ocr_results.get("info") or ""),
            advanced_text=str(ocr_results.get("advanced") or ""),
            effect_text=str(ocr_results.get("effect") or ""),
        )

    def test_normalize_alias_key_removes_level_suffix_before_slash_cleanup(self) -> None:
        self.assertEqual(normalize_alias_key("Deep Sea Visitor Lv. 60/60"), "deep sea visitor")
        self.assertEqual(
            normalize_alias_key("\u0413\u043e\u0441\u0442\u044c \u0438\u0437 \u0433\u043b\u0443\u0431\u0438\u043d \u0423\u0440. 60/60"),
            "\u0433\u043e\u0441\u0442\u044c \u0438\u0437 \u0433\u043b\u0443\u0431\u0438\u043d",
        )

    def test_parse_amplifier_detail_reads_english_weapon_fields(self) -> None:
        readout = parse_amplifier_detail(
            "Hellfire Gears O Lv. 50/50",
            info_text=(
                "Hellfire Gears O Lv. 50/50 "
                "Base Stat Base ATK Advanced Stat Impact W-Engine Effect 570 15.8%"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.identity.weapon_id, "amp_hellfire_gears")
        self.assertEqual(readout.level, 50)
        self.assertEqual(readout.level_cap, 50)
        self.assertEqual(readout.base_stat_key, "attack_flat")
        self.assertEqual(readout.base_stat_value, 570)
        self.assertEqual(readout.advanced_stat_key, "impact")
        self.assertEqual(readout.advanced_stat_value, 15.8)

    def test_parse_amplifier_detail_reads_russian_crit_rate(self) -> None:
        readout = parse_amplifier_detail(
            "\u0413\u043e\u0441\u0442\u044c \u0438\u0437 \u0433\u043b\u0443\u0431\u0438\u043d \u0423\u0440. 60160",
            info_text=(
                "\u0413\u043e\u0441\u0442\u044c \u0438\u0437 \u0433\u043b\u0443\u0431\u0438\u043d \u0423\u0440. 60160 "
                "\u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u041f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0428\u0430\u043d\u0441 \u043a\u0440\u0438\u0442. \u043f\u043e\u043f\u0430\u0434\u0430\u043d\u0438\u044f "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 713 240/0"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.identity.weapon_id, "amp_deep_sea_visitor")
        self.assertEqual(readout.level, 60)
        self.assertEqual(readout.level_cap, 60)
        self.assertEqual(readout.base_stat_key, "attack_flat")
        self.assertEqual(readout.base_stat_value, 713)
        self.assertEqual(readout.advanced_stat_key, "crit_rate_pct")
        self.assertEqual(readout.advanced_stat_value, 24.0)

    def test_parse_amplifier_detail_falls_back_to_info_text_identity_when_title_is_empty(self) -> None:
        readout = parse_amplifier_detail(
            "",
            info_text=(
                "Hellfire Gears O Lv. 50/50 "
                "Base Stats Base ATK "
                "Advanced Stats Impact "
                "W-Engine Effect 570 15.8%"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.identity.weapon_id, "amp_hellfire_gears")
        self.assertEqual(readout.level, 50)
        self.assertEqual(readout.level_cap, 50)
        self.assertEqual(readout.base_stat_value, 570)
        self.assertEqual(readout.advanced_stat_key, "impact")
        self.assertEqual(readout.advanced_stat_value, 15.8)

    def test_parse_amplifier_detail_combines_partial_title_and_info_for_identity(self) -> None:
        readout = parse_amplifier_detail(
            "Hellfire",
            info_text=(
                "Gears O Lv. 50/50 "
                "Base Stats Base ATK "
                "Advanced Stats Impact "
                "W-Engine Effect 570 15.8%"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.identity.weapon_id, "amp_hellfire_gears")
        self.assertEqual(readout.level, 50)
        self.assertEqual(readout.level_cap, 50)
        self.assertEqual(readout.base_stat_value, 570)
        self.assertEqual(readout.advanced_stat_key, "impact")
        self.assertEqual(readout.advanced_stat_value, 15.8)

    def test_parse_amplifier_detail_keeps_advanced_value_without_inventing_label(self) -> None:
        readout = parse_amplifier_detail(
            "\u041a\u043b\u0435\u0442\u043a\u0430 \u041d\u0435\u0431\u0435\u0441\u043d\u043e\u0439 \u043f\u0442\u0438\u0446\u044b \u0423\u0440. 60160",
            info_text=(
                "\u041a\u043b\u0435\u0442\u043a\u0430 \u041d\u0435\u0431\u0435\u0441\u043d\u043e\u0439 \u043f\u0442\u0438\u0446\u044b \u0423\u0440. 60160 "
                "\u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u041f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 743 300/0"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.identity.weapon_id, "amp_qingming_birdcage")
        self.assertEqual(readout.level, 60)
        self.assertEqual(readout.level_cap, 60)
        self.assertEqual(readout.base_stat_key, "attack_flat")
        self.assertEqual(readout.base_stat_value, 743)
        self.assertIsNone(readout.advanced_stat_key)
        self.assertEqual(readout.advanced_stat_value, 30.0)

    def test_parse_amplifier_detail_prefers_title_base_and_post_effect_advanced_values(self) -> None:
        readout = parse_amplifier_detail(
            "\u0420\u0435\u0432\u0443\u0449\u0430\u044f \u0442\u0430\u0447\u043a\u0430 \u0423\u0440. 60160 \u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 624",
            info_text=(
                "\u0420\u0435\u0432\u0443\u0449\u0430\u044f \u0442\u0430\u0447\u043a\u0430 \u0423\u0440. 60160 "
                "\u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u041f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0421\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 624 250/0"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.base_stat_key, "attack_flat")
        self.assertEqual(readout.base_stat_value, 624)
        self.assertEqual(readout.advanced_stat_key, "attack_pct")
        self.assertEqual(readout.advanced_stat_value, 25.0)

    def test_parse_amplifier_detail_prefers_advanced_line_value_over_effect_passive_percent(self) -> None:
        readout = parse_amplifier_detail(
            "\u0413\u043e\u0441\u0442\u044c \u0438\u0437 \u0433\u043b\u0443\u0431\u0438\u043d \u0423\u0440. 60160 \u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 713",
            info_text=(
                "\u0413\u043e\u0441\u0442\u044c \u0438\u0437 \u0433\u043b\u0443\u0431\u0438\u043d \u0423\u0440. 60160 "
                "\u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u041f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0428\u0430\u043d\u0441 \u043a\u0440\u0438\u0442. \u043f\u043e\u043f\u0430\u0434\u0430\u043d\u0438\u044f "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 713"
            ),
            advanced_text=(
                "\u041f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0428\u0430\u043d\u0441 \u043a\u0440\u0438\u0442. \u043f\u043e\u043f\u0430\u0434\u0430\u043d\u0438\u044f 240/0"
            ),
            effect_text=(
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 "
                "\u041f\u043e\u0432\u044b\u0448\u0430\u0435\u0442 \u043b\u0435\u0434\u044f\u043d\u043e\u0439 \u0443\u0440\u043e\u043d \u043d\u0430 250/0"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.base_stat_value, 713)
        self.assertEqual(readout.advanced_stat_key, "crit_rate_pct")
        self.assertEqual(readout.advanced_stat_value, 24.0)

    def test_parse_amplifier_detail_uses_single_post_effect_value_for_advanced_stat(self) -> None:
        readout = parse_amplifier_detail(
            "\u0411\u0443\u043c-\u043f\u0443\u0448\u043a\u0430 \u0423\u0440. 30/30 \u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438",
            info_text=(
                "\u0411\u0443\u043c-\u043f\u0443\u0448\u043a\u0430 \u0423\u0440. 30/30 "
                "\u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u041f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0412\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435 \u044d\u043d\u0435\u0440\u0433\u0438\u0438 "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 320/0"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.level, 30)
        self.assertEqual(readout.level_cap, 30)
        self.assertEqual(readout.base_stat_value, 320)
        self.assertEqual(readout.advanced_stat_key, "energy_regen")
        self.assertEqual(readout.advanced_stat_value, 32.0)

    def test_parse_amplifier_detail_normalizes_shifted_impact_value(self) -> None:
        readout = parse_amplifier_detail(
            "\u0410\u043a\u043a\u0443\u043c\u0443\u043b\u044f\u0442\u043e\u0440 \u0434\u0435\u043c\u0430\u0440\u044b II \u0423\u0440. 60/60 \u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 624",
            info_text=(
                "\u0410\u043a\u043a\u0443\u043c\u0443\u043b\u044f\u0442\u043e\u0440 \u0434\u0435\u043c\u0430\u0440\u044b II \u0423\u0440. 60/60 "
                "\u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u041f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0418\u043c\u043f\u0443\u043b\u044c\u0441 "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 624 150,6"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.base_stat_value, 624)
        self.assertEqual(readout.advanced_stat_key, "impact")
        self.assertEqual(readout.advanced_stat_value, 15.06)

    def test_parse_amplifier_detail_does_not_reuse_base_value_as_advanced_stat(self) -> None:
        readout = parse_amplifier_detail(
            "Deep Sea Visitor Lv. 60/60 Base ATK 713",
            info_text=(
                "Deep Sea Visitor Lv. 60/60 "
                "Base Stats Base ATK "
                "Advanced Stats Crit Rate "
                "W-Engine Effect 713"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.base_stat_value, 713)
        self.assertEqual(readout.advanced_stat_key, "crit_rate_pct")
        self.assertIsNone(readout.advanced_stat_value)

    def test_parse_amplifier_detail_does_not_invent_duplicate_base_value_without_advanced_label(self) -> None:
        readout = parse_amplifier_detail(
            "Drill Rig - Red Axis Lv. 50/50",
            info_text=(
                "Drill Rig - Red Axis Lv. 50/50 "
                "Base Stats Base ATK "
                "W-Engine Effect 521 521"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.base_stat_value, 521)
        self.assertIsNone(readout.advanced_stat_key)
        self.assertIsNone(readout.advanced_stat_value)

    def test_parse_amplifier_detail_recovers_split_advanced_label_around_numeric_noise(self) -> None:
        readout = parse_amplifier_detail(
            "\u0411\u0443\u0440 \u2014 \u043a\u0440\u0430\u0441\u043d\u0430\u044f \u043e\u0441\u044c 60 \u041e \u0423\u0440. 50/50 \u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 521",
            info_text=(
                "\u0411\u0443\u0440 \u2014 \u043a\u0440\u0430\u0441\u043d\u0430\u044f \u043e\u0441\u044c \u0445 60 \u0423\u0440. 50/50 "
                "\u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u043f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0432\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435 521 \u044d\u043d\u0435\u0440\u0433\u0438\u0438 "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.base_stat_value, 521)
        self.assertEqual(readout.advanced_stat_key, "energy_regen")
        self.assertIsNone(readout.advanced_stat_value)

    def test_parse_amplifier_detail_recovers_shifted_energy_regen_decimal(self) -> None:
        readout = parse_amplifier_detail(
            "\u0411\u0443\u0440 \u2014 \u043a\u0440\u0430\u0441\u043d\u0430\u044f \u043e\u0441\u044c 60 \u041e \u0423\u0440. 50/50 \u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 521",
            info_text=(
                "\u0411\u0443\u0440 \u2014 \u043a\u0440\u0430\u0441\u043d\u0430\u044f \u043e\u0441\u044c 60 \u0423\u0440. 50/50 "
                "\u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u041f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0412\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435 \u044d\u043d\u0435\u0440\u0433\u0438\u0438 "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 521 440/0"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.base_stat_value, 521)
        self.assertEqual(readout.advanced_stat_key, "energy_regen")
        self.assertEqual(readout.advanced_stat_value, 4.4)

    @unittest.skipUnless(_ELLEN_LIVE_AMP_SAMPLE.exists(), "local Ellen live amplifier sample is unavailable")
    def test_parse_amplifier_detail_recovers_ellen_advanced_value_from_live_sample(self) -> None:
        readout = self._read_live_sample(_ELLEN_LIVE_AMP_SAMPLE)

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.identity.weapon_id, "amp_deep_sea_visitor")
        self.assertEqual(readout.base_stat_value, 713)
        self.assertEqual(readout.advanced_stat_key, "crit_rate_pct")
        self.assertEqual(readout.advanced_stat_value, 24.0)

    @unittest.skipUnless(_HARUMASA_LIVE_AMP_SAMPLE.exists(), "local Harumasa live amplifier sample is unavailable")
    def test_parse_amplifier_detail_keeps_harumasa_live_sample_honest(self) -> None:
        readout = self._read_live_sample(_HARUMASA_LIVE_AMP_SAMPLE)

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.identity.weapon_id, "amp_drill_rig_red_axis")
        self.assertEqual(readout.base_stat_value, 521)
        self.assertEqual(readout.advanced_stat_key, "energy_regen")
        self.assertEqual(readout.advanced_stat_value, 4.4)

    def test_parse_amplifier_detail_parses_lookalike_level_token(self) -> None:
        readout = parse_amplifier_detail(
            "\u0414\u043e\u043c\u0440\u0430\u0431\u043e\u0442\u043d\u0438\u0446\u0430 \u0423\u0440. \u041e\u041b\u041e \u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 42",
            info_text=(
                "\u0414\u043e\u043c\u0440\u0430\u0431\u043e\u0442\u043d\u0438\u0446\u0430 \u0423\u0440. \u041e\u041b\u041e "
                "\u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u041f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0421\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 42 100/0"
            ),
            effect_text="\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 \u0423\u0440. 15/15",
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.level, 1)
        self.assertEqual(readout.level_cap, 10)
        self.assertEqual(readout.base_stat_value, 42)
        self.assertEqual(readout.advanced_stat_key, "attack_pct")
        self.assertEqual(readout.advanced_stat_value, 10.0)

    def test_parse_amplifier_detail_prefers_advanced_value_after_base_flat_stat(self) -> None:
        readout = parse_amplifier_detail(
            "\u0412\u0441\u0442\u0440\u043e\u0435\u043d\u043d\u044b\u0439 \u043a\u043e\u043c\u043f\u0438\u043b\u044f\u0442\u043e\u0440 \u041e \u0423\u0440. 50/50",
            info_text=(
                "\u0412\u0441\u0442\u0440\u043e\u0435\u043d\u043d\u044b\u0439 \u043a\u043e\u043c\u043f\u0438\u043b\u044f\u0442\u043e\u0440 \u041e \u0423\u0440. 50150 "
                "\u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u041f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "570 \u041f\u0440\u043e\u0446\u0435\u043d\u0442 \u043f\u0440\u043e\u0431\u0438\u0432\u0430\u043d\u0438\u044f 21,10/0 "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 "
                "\u0421\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 \u043f\u043e\u0432\u044b\u0448\u0430\u0435\u0442\u0441\u044f \u043d\u0430 120/0"
            ),
            effect_text=(
                "\u041f\u0440\u043e\u0446\u0435\u043d\u0442 \u043f\u0440\u043e\u0431\u0438\u0432\u0430\u043d\u0438\u044f "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 "
                "21,10/0 \u0421\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 \u043f\u043e\u0432\u044b\u0448\u0430\u0435\u0442\u0441\u044f \u043d\u0430 120/0"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.base_stat_value, 570)
        self.assertEqual(readout.advanced_stat_key, "pen_ratio_pct")
        self.assertEqual(readout.advanced_stat_value, 21.1)

    def test_parse_amplifier_detail_recovers_missing_advanced_value_from_global_candidates(self) -> None:
        readout = parse_amplifier_detail(
            "\u0420\u0435\u0432\u0443\u0449\u0430\u044f \u0442\u0430\u0447\u043a\u0430 \u041e \u0423\u0440. 60160",
            info_text=(
                "\u0420\u0435\u0432\u0443\u0449\u0430\u044f \u0442\u0430\u0447\u043a\u0430 \u0423\u0440. 60160 "
                "\u0411\u0430\u0437\u043e\u0432\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0411\u0430\u0437\u043e\u0432\u0430\u044f \u0441\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u041f\u0440\u043e\u0434\u0432\u0438\u043d\u0443\u0442\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b "
                "\u0421\u0438\u043b\u0430 \u0430\u0442\u0430\u043a\u0438 "
                "\u042d\u0444\u0444\u0435\u043a\u0442 \u0430\u043c\u043f\u043b\u0438\u0444\u0438\u043a\u0430\u0442\u043e\u0440\u0430 250/0"
            ),
        )

        self.assertIsNotNone(readout)
        assert readout is not None
        self.assertEqual(readout.advanced_stat_key, "attack_pct")
        self.assertEqual(readout.advanced_stat_value, 25.0)


if __name__ == "__main__":
    unittest.main()
