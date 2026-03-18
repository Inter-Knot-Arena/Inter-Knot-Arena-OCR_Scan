from __future__ import annotations

import unittest

from amplifier_identity import normalize_alias_key, parse_amplifier_detail


class AmplifierIdentityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
