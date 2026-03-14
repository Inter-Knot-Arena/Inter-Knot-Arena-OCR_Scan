from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageOps

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equipment_taxonomy import display_name_for_id, known_disc_set_ids, normalize_display_text, normalize_name_and_id
from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, AMPLIFIER_DETAIL_ROLE, DISK_DETAIL_ROLE, source_index_from_manifest
from roster_taxonomy import canonicalize_agent_label


REPO_ROOT = Path(__file__).resolve().parents[1]
WIN_OCR_SCRIPT = Path(__file__).resolve().with_name("win_ocr_batch.ps1")
TITLE_CROP_BOX = (0.28, 0.10, 0.66, 0.36)

EN_LANGUAGE_TAG = "en-US"
RU_LANGUAGE_TAG = "ru"

EN_AMPLIFIER_TITLE_FIXES = {
    "i the brimstone": "The Brimstone",
    "weeping gemini": "Weeping Gemini",
    "weeping cradle": "Weeping Cradle",
    "grilco wisp": "Grill O'Wisp",
    "fligft of fancy": "Flight of Fancy",
    "riginal transmorpher": "Original Transmorpher",
    "i spectral gaze": "Spectral Gaze",
    "i peacekeeper specialized": "Peacekeeper - Specialized",
    "identity base": "Identity Base",
    "identity inflection": "Identity Inflection",
    "vortexl revolver": "[Vortex] Revolver",
    "magnetic storm i brav": "Magnetic Storm Bravo",
    "lunarl pleniluna": "Lunar Pleniluna",
    "bunny band": "Bunny Band",
    "roaring fur nace": "Roaring Fur-nace",
    "housekeeper": "Housekeeper",
    "peacekeeper specialized": "Peacekeeper - Specialized",
    "riot suppressor mark vi": "Riot Suppressor Mark VI",
    "demara battery mark ii": "Demara Battery Mark II",
    "drill rig red axis": "Drill Rig - Red Axis",
    "starlight engine replica": "Starlight Engine Replica",
    "heartstring nocturne": "Heartstring Nocturne",
    "cloudcleave radiance": "Cloudcleave Radiance",
    "qingming birdcage": "Qingming Birdcage",
    "rainforest gourmet": "Rainforest Gourmet",
    "precious fossilized core": "Precious Fossilized Core",
    "tremor trigram vessel": "Tremor Trigram Vessel",
    "flamemaker shaker": "Flamemaker Shaker",
    "electro lip gloss": "Electro-Lip Gloss",
    "angel in the shell": "Angel in the Shell",
    "gilded blossom": "Gilded Blossom",
    "magnetic storm bravo": "Magnetic Storm Bravo",
    "marcato desire": "Marcato Desire",
    "thoughtbop": "Thoughtbop",
    "metanukimorphosis": "Metanukimorphosis",
    "ice jade teapot": "Ice-Jade Teapot",
    "slice of time": "Slice of Time",
    "vortex revolver": "[Vortex] Revolver",
    "vortex revolver o": "[Vortex] Revolver",
    "roaring fur nace o": "Roaring Fur-nace",
    "roaring fur nace e": "Roaring Fur-nace",
    "roaring fur nace r": "Roaring Fur-nace",
}

RU_DISC_SET_ALIASES = {
    "дятлокор электро": "set_woodpecker_electro",
    "сказания юнькуй": "set_yunkui_tales",
    "шокстар диско": "set_shockstar_disco",
    "песнь о ветке и клинке": "set_branch_and_blade_song",
    "цветок на рассвете": "set_dawn_s_bloom",
    "владыка горы": "set_king_of_the_summit",
    "гармония теней": "set_shadow_harmony",
    "грозовой хэви металл": "set_thunder_metal",
    "лучезарная ария": "set_shining_aria",
    "гормон панк": "set_hormone_punk",
    "баллада о фаэтоне": "set_phaethon_s_melody",
    "свинг джаз": "set_swing_jazz",
    "соул рок": "set_soul_rock",
    "свирепый хэви металл": "set_fanged_metal",
    "фугу электро": "set_puffer_electro",
    "протопанк": "set_proto_punk",
    "хаос джаз": "set_chaos_jazz",
    "хаос металл": "set_chaotic_metal",
    "полярный хэви металл": "set_polar_metal",
    "астральный голос": "set_astral_voice",
    "песнь о синих водах": "set_white_water_ballad",
    "свобода блюз": "set_freedom_blues",
    "фридом блюз": "set_freedom_blues",
    "инферно металл": "set_inferno_metal",
    "лунная колыбельная": "set_moonlight_lullaby",
    "полярный металл": "set_polar_metal",
    "грозовой металл": "set_thunder_metal",
    "свирепый металл": "set_fanged_metal",
    "шокстар-диско": "set_shockstar_disco",
    "свирепый хэви-метал": "set_fanged_metal",
    "свинг-джаз": "set_swing_jazz",
    "полярный хэви-метал": "set_polar_metal",
    "фридом-блюз": "set_freedom_blues",
    "грозовой хэви-метал": "set_thunder_metal",
    "дятлокор-электро": "set_woodpecker_electro",
    "хаос-джаз": "set_chaos_jazz",
    "хаос-метал": "set_chaotic_metal",
    "гормон-панк": "set_hormone_punk",
    "фугу-электро": "set_puffer_electro",
    "инферно-метал": "set_inferno_metal",
    "соул-рок": "set_soul_rock",
    "ятлокор электро": "set_woodpecker_electro",
    "woodpecker eiectro": "set_woodpecker_electro",
    "soui rock": "set_soul_rock",
    "astrai voice": "set_astral_voice",
}

RU_AMPLIFIER_ALIASES = {
    "всесторонняя отточенность": "Practiced Perfection",
    "плачущие близнецы": "Weeping Gemini",
    "храм вьюги": "Hailstorm Shrine",
    "проблеск в облаках": "Cloudcleave Radiance",
    "встроенный компилятор": "Fusion Compiler",
    "сдержанность": "The Restrained",
    "клетка небесной птицы": "Qingming Birdcage",
    "звездный двигатель": "Starlight Engine",
    "звёздный двигатель": "Starlight Engine",
    "реплика звездного двигателя": "Starlight Engine Replica",
    "реплика звёздного двигателя": "Starlight Engine Replica",
    "полет среди грез": "Flight of Fancy",
    "полет среди грез": "Flight of Fancy",
    "полёт среди грёз": "Flight of Fancy",
    "полет среди грёз": "Flight of Fancy",
    "бум пушка": "Kaboom the Cannon",
    "бум пушка о": "Kaboom the Cannon",
    "треножник поднебесья": "Tremor Trigram Vessel",
    "большой цилиндр": "Big Cylinder",
    "струны ночи": "Heartstring Nocturne",
    "тропический гурман": "Rainforest Gourmet",
    "драгоценная окаменелость": "Precious Fossilized Core",
    "пламя брани": "Bellicose Blaze",
    "сундук сокровищ": "The Vault",
    "оригинальный трансформатор": "Original Transmorpher",
    "пароварка": "Steam Oven",
    "шестерни адского пламени": "Hellfire Gears",
    "сера": "Sulfur",
    "заостренные шипы": "Sharpened Stinger",
    "заострённые шипы": "Sharpened Stinger",
    "бумагорезка": "Box Cutter",
    "колыбель плача": "Weeping Cradle",
    "призрачный взор": "Spectral Gaze",
    "застенчивый монстр": "Bashful Demon",
    "шейкер огнемейкер": "Flamemaker Shaker",
    "шейкер огнемейкер": "Flamemaker Shaker",
    "шейкер огнемейкер о": "Flamemaker Shaker",
    "шейкер огнемейкер": "Flamemaker Shaker",
    "домработница": "Housekeeper",
    "гость из глубин": "Deep Sea Visitor",
    "миротворец специализированный": "Peacekeeper - Specialized",
    "мысли в бит": "Thoughtbop",
    "семь уловок тануки": "Metanukimorphosis",
    "усмиритель беспорядков vi": "Riot Suppressor Mark VI",
    "звонок из прошлого": "Yesterday Calls",
    "златоцвет": "Gilded Blossom",
    "ревущая тачка": "Roaring Ride",
    "бур красная ось": "Drill Rig - Red Axis",
    "тысяча затмений": "A Thousand Eclipses",
    "чистота утраты": "Severed Innocence",
    "клыки ярости": "Wrathful Vajra",
    "стильная штучка": "Elegant Vanity",
    "магнитная буря альфа": "Magnetic Storm Alpha",
    "кадр на память": "Slice of Time",
    "душа в доспехах": "Soul in Steel",
    "шар головоломка": "Puzzle Sphere",
    "грозный страж": "Wrathful Vajra",
    "артиллерийский ротор": "Artillery Rotor",
    "маркато реверб": "Marcato Desire",
    "аптечка дзансин": "Zanshin Herb Case",
    "турбулентность револьвер": "[Vortex] Revolver",
    "очаг грез": "Dreamlit Hearth",
    "очаг грёз": "Dreamlit Hearth",
    "вместилище фурчаний": "Roaring Fur-nace",
    "рукотворное сердце": "Cordis Germina",
    "колыбель кракена": "Kraken's Cradle",
    "пламенный венец": "Blazing Laurel",
    "заячья корзинка": "Bunny Band",
    "чайник нефритовой чистоты": "Ice-Jade Teapot",
    "жарево заката": "Grill O'Wisp",
}

GENERIC_TITLE_MARKERS = {
    "new",
    "demo",
    "uid",
    "базовые параметры",
    "базовый параметр",
    "base stat",
    "main stat",
    "sub stats",
    "sub-stat",
    "дополнительные параметры",
}


def _slugify(value: str) -> str:
    token = unicodedata.normalize("NFKC", str(value or "").strip().lower())
    token = token.replace("&", " and ")
    token = token.replace("/", " ")
    token = token.replace("-", " ")
    token = re.sub(r"[^0-9a-zA-Z]+", " ", token)
    return "_".join(part for part in token.split() if part)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_alias_key(value: str) -> str:
    normalized = normalize_display_text(value)
    normalized = normalized.replace("ё", "е")
    normalized = normalized.replace("—", "-")
    normalized = normalized.replace("–", "-")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\bnew\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^[\W_]+", " ", normalized)
    normalized = re.sub(r"^[a-zа-я]\s+", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^[0-9]+\s+", " ", normalized)
    normalized = re.sub(r"^\b[оo]\b\s*", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(?:ур|yp|lv)[a-zа-я0-9./-]*\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bo\b$", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bо\b$", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b[еe]\b$", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bпемо\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdemo\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b[0-9]{1,2}\b", " ", normalized)
    normalized = normalized.translate(str.maketrans({"1": "l", "0": "o", "5": "s", "8": "b"}))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _fuzzy_lookup(mapping: Mapping[str, str], candidate: str, *, threshold: float, min_margin: float = 0.05) -> Tuple[str, float]:
    normalized_candidate = _normalize_alias_key(candidate)
    if not normalized_candidate:
        return "", 0.0
    best_value = ""
    best_score = 0.0
    second_best = 0.0
    for raw_key, mapped_value in mapping.items():
        normalized_key = _normalize_alias_key(str(raw_key))
        if not normalized_key:
            continue
        score = difflib.SequenceMatcher(None, normalized_candidate, normalized_key).ratio()
        if score > best_score:
            second_best = best_score
            best_score = score
            best_value = str(mapped_value)
        elif score > second_best:
            second_best = score
    if best_value and best_score >= threshold and (best_score - second_best) >= min_margin:
        return best_value, best_score
    return "", 0.0


def _normalized_mapping_lookup(mapping: Mapping[str, str], candidate: str) -> str:
    normalized_candidate = _normalize_alias_key(candidate)
    if not normalized_candidate:
        return ""
    for raw_key, mapped_value in mapping.items():
        if _normalize_alias_key(str(raw_key)) == normalized_candidate:
            return str(mapped_value)
    return ""


def _language_tag_for_locale(locale: str) -> str:
    return EN_LANGUAGE_TAG if str(locale or "").strip().upper() == "EN" else RU_LANGUAGE_TAG


def _crop_title_image(source_path: Path, output_path: Path) -> None:
    image = Image.open(source_path).convert("L")
    width, height = image.size
    crop = image.crop(
        (
            int(width * TITLE_CROP_BOX[0]),
            int(height * TITLE_CROP_BOX[1]),
            int(width * TITLE_CROP_BOX[2]),
            int(height * TITLE_CROP_BOX[3]),
        )
    )
    crop = ImageOps.autocontrast(crop)
    crop = crop.resize((crop.width * 2, crop.height * 2))
    crop.save(output_path)


def _run_ocr_batch(crops: Sequence[Dict[str, str]], *, language_tag: str, temp_root: Path) -> Dict[str, str]:
    if not crops:
        return {}
    input_path = temp_root / f"ocr_input_{language_tag}.json"
    output_path = temp_root / f"ocr_output_{language_tag}.json"
    input_path.write_text(json.dumps(list(crops), ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WIN_OCR_SCRIPT),
            "-InputJson",
            str(input_path),
            "-OutputJson",
            str(output_path),
            "-LanguageTag",
            language_tag,
        ],
        check=True,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8-sig"))
    return {
        str(item.get("id") or ""): _text(item.get("text"))
        for item in payload
        if isinstance(item, dict) and str(item.get("id") or "")
    }


def _extract_disk_title_candidate(text: str) -> str:
    normalized = " ".join(_text(text).split())
    if not normalized:
        return ""
    candidate = re.split(
        r"(?i)\b(?:lv(?:\.|[a-z0-9/.-]+)?|ур(?:\.|[a-zа-я0-9/.-]+)?|main stat|sub-?stats|[a-zа-я0-9]+\s+stat|базов(?:ый|ые)|дополнительные параметры)\b",
        normalized,
    )[0]
    candidate = re.sub(r"\[[^\]]*\]", " ", candidate)
    candidate = re.sub(r"\[[^\]]*$", " ", candidate)
    candidate = re.sub(r"\bnew\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^[\W_]+", " ", candidate)
    candidate = re.sub(r"^[a-zа-я]\s+", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^[0-9]+\s+", " ", candidate)
    candidate = re.sub(r"^\b[оo]\b\s*", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b[0-9]{1,2}\b", " ", candidate)
    candidate = re.sub(r"\b(?:ур|yp|lv)[a-zа-я0-9/.-]*\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bo\b$", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bо\b$", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b[еe]\b$", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -")
    return candidate


def _extract_amplifier_title_candidate(text: str) -> str:
    normalized = " ".join(_text(text).split())
    if not normalized:
        return ""
    candidate = re.split(
        r"(?i)\b(?:lv(?:\.|[a-z0-9/.-]+)?|ур(?:\.|[a-zа-я0-9/.-]+)?|base stat|базов(?:ые|ый))\b",
        normalized,
    )[0]
    candidate = re.sub(r"\bnew\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^[\W_]+", " ", candidate)
    candidate = re.sub(r"^[a-zа-я]\s+", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^\b[оo]\b\s*", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b(?:ур|yp|lv)[a-zа-я0-9/.-]*\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bo\b$", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bо\b$", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b[еe]\b$", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -")
    return candidate


def _canonical_disc_from_title(candidate: str) -> Tuple[str, str, str]:
    raw = _text(candidate)
    if not raw:
        return "", "", ""
    mapped_id = _normalized_mapping_lookup(RU_DISC_SET_ALIASES, raw)
    if mapped_id:
        return mapped_id, display_name_for_id(mapped_id), "ru_alias"
    fuzzy_id, _ = _fuzzy_lookup(RU_DISC_SET_ALIASES, raw, threshold=0.84)
    if fuzzy_id:
        return fuzzy_id, display_name_for_id(fuzzy_id), "ru_alias_fuzzy"

    try:
        canonical_id, display_name = normalize_name_and_id(raw, "", kind="disc_set")
        return canonical_id, display_name, "taxonomy_direct"
    except ValueError:
        pass

    best_id = ""
    best_display = ""
    best_score = 0.0
    normalized_candidate = _normalize_alias_key(raw)
    for known_id in sorted(str(item) for item in known_disc_set_ids()):
        display_name = display_name_for_id(known_id)
        if not display_name:
            continue
        score = difflib.SequenceMatcher(None, normalized_candidate, _normalize_alias_key(display_name)).ratio()
        if score > best_score:
            best_id = known_id
            best_display = display_name
            best_score = score
    if best_id and best_score >= 0.72:
        return best_id, best_display, "taxonomy_fuzzy"
    return "", "", ""


def _title_case_display(value: str) -> str:
    raw = _text(value)
    if not raw:
        return ""
    if re.search(r"[A-Z]", raw):
        return raw
    return " ".join(part.capitalize() for part in raw.split())


def _canonical_amp_from_title(candidate: str) -> Tuple[str, str, str]:
    raw = _text(candidate)
    if not raw:
        return "", "", ""
    display_name = _normalized_mapping_lookup(EN_AMPLIFIER_TITLE_FIXES, raw)
    if display_name:
        pass
    else:
        display_name = _normalized_mapping_lookup(RU_AMPLIFIER_ALIASES, raw)
    if display_name:
        pass
    else:
        fuzzy_display, _ = _fuzzy_lookup(EN_AMPLIFIER_TITLE_FIXES, raw, threshold=0.90)
        if fuzzy_display:
            display_name = fuzzy_display
        else:
            fuzzy_display, _ = _fuzzy_lookup(RU_AMPLIFIER_ALIASES, raw, threshold=0.84)
            if fuzzy_display:
                display_name = fuzzy_display
            else:
                display_name = _title_case_display(raw)

    if not display_name:
        return "", "", ""

    if _normalize_alias_key(display_name) in GENERIC_TITLE_MARKERS:
        return "", "", ""

    try:
        canonical_id, canonical_display = normalize_name_and_id(display_name, "", kind="amplifier")
        return canonical_id, canonical_display, "taxonomy_direct"
    except ValueError:
        pass

    slug = _slugify(display_name)
    if not slug:
        return "", "", ""
    return f"amp_{slug}", display_name, "generated_slug"


def _capture_slot(record: Mapping[str, Any], source: Mapping[str, Any]) -> int | None:
    for payload in (record.get("captureHints"), source.get("captureHints")):
        if not isinstance(payload, dict):
            continue
        value = payload.get("slotIndex")
        if isinstance(value, (int, float)):
            slot = int(value)
            if 1 <= slot <= 6:
                return slot
    return None


def _focus_agent_id(record: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    for raw in (record.get("focusAgentId"), source.get("focusAgentId")):
        canonical = canonicalize_agent_label(raw)
        if canonical:
            return canonical
    return ""


def _review_payload(*, reviewer: str, notes: str) -> Dict[str, Any]:
    return {"reviewer": reviewer, "reviewedAt": utc_now(), "notes": notes}


def _apply_disk_review(record: Dict[str, Any], *, set_id: str, set_name: str, slot: int, agent_id: str, reviewer: str, source_mode: str) -> None:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        labels = {}
        record["labels"] = labels
    labels["head"] = "equipment"
    labels["disc_set_id"] = set_id
    labels["disc_set_name"] = set_name
    labels["disc_slot"] = slot
    labels["equipment_agent_id"] = agent_id
    labels["label"] = set_id
    labels["disc"] = {
        "slot": slot,
        "setId": set_id,
        "displayName": set_name,
        "agentId": agent_id,
    }
    labels["reviewFinal"] = _review_payload(
        reviewer=reviewer,
        notes=f"Auto-reviewed from WinRT OCR title crop; minimal disc identity labels only ({source_mode}).",
    )
    record["head"] = "equipment"
    record["qaStatus"] = "reviewed"


def _apply_amplifier_review(record: Dict[str, Any], *, amp_id: str, amp_name: str, agent_id: str, reviewer: str, source_mode: str) -> None:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        labels = {}
        record["labels"] = labels
    labels["head"] = "equipment"
    labels["amplifier_id"] = amp_id
    labels["amplifier_name"] = amp_name
    labels["equipment_agent_id"] = agent_id
    labels["weapon_present"] = True
    labels["label"] = amp_id
    labels["weapon"] = {
        "weaponId": amp_id,
        "displayName": amp_name,
        "weaponPresent": True,
        "agentId": agent_id,
    }
    labels["reviewFinal"] = _review_payload(
        reviewer=reviewer,
        notes=f"Auto-reviewed from WinRT OCR title crop; minimal amplifier identity labels only ({source_mode}).",
    )
    record["head"] = "equipment"
    record["qaStatus"] = "reviewed"


def _reviewed_final(record: Mapping[str, Any]) -> bool:
    labels = record.get("labels")
    return isinstance(labels, dict) and isinstance(labels.get("reviewFinal"), dict) and str(record.get("qaStatus") or "").strip().lower() == "reviewed"


def _iter_target_records(
    records: Iterable[Dict[str, Any]],
    source_index: Mapping[str, Dict[str, Any]],
    session_filter: set[str],
) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("workflow") or "").strip() != ACCOUNT_IMPORT_WORKFLOW:
            continue
        if str(record.get("screenRole") or "").strip() not in {DISK_DETAIL_ROLE, AMPLIFIER_DETAIL_ROLE}:
            continue
        if session_filter and str(record.get("sessionId") or "").strip() not in session_filter:
            continue
        source = source_index.get(str(record.get("sourceId") or "").strip())
        if not isinstance(source, dict):
            continue
        yield record, source


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-review minimal disk_detail and amplifier_detail identity labels from WinRT OCR title crops.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--reviewer-id", default="auto_equipment_identity")
    parser.add_argument("--output-json", default="docs/auto_review_equipment_identity.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    source_index = source_index_from_manifest(manifest.get("sources", []))
    session_filter = {str(item).strip() for item in args.session if str(item).strip()}

    targets: List[Tuple[Dict[str, Any], Dict[str, Any]]] = list(_iter_target_records(records, source_index, session_filter))
    stats: Counter[str] = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with tempfile.TemporaryDirectory(prefix="ocr_equipment_identity_") as tmp_raw:
        temp_root = Path(tmp_raw)
        crop_dir = temp_root / "title_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        batched_crops: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        crop_lookup: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}

        for record, source in targets:
            stats["candidateRecords"] += 1
            if _reviewed_final(record):
                stats["skippedAlreadyReviewed"] += 1
                continue
            image_path = Path(str(record.get("path") or ""))
            if not image_path.exists():
                stats["skippedMissingImage"] += 1
                continue
            crop_id = str(record.get("id") or "")
            if not crop_id:
                stats["skippedMissingRecordId"] += 1
                continue
            language_tag = _language_tag_for_locale(str(record.get("locale") or ""))
            crop_path = crop_dir / f"{crop_id}.png"
            _crop_title_image(image_path, crop_path)
            batched_crops[language_tag].append({"id": crop_id, "path": str(crop_path)})
            crop_lookup[crop_id] = (record, source)

        ocr_results: Dict[str, str] = {}
        for language_tag, crops in batched_crops.items():
            ocr_results.update(_run_ocr_batch(crops, language_tag=language_tag, temp_root=temp_root))

        for crop_id, (record, source) in crop_lookup.items():
            role = str(record.get("screenRole") or "")
            locale = str(record.get("locale") or "")
            title_text = _text(ocr_results.get(crop_id))
            if not title_text:
                stats[f"{role}.skippedEmptyOcr"] += 1
                continue

            focus_agent_id = _focus_agent_id(record, source)
            if not focus_agent_id:
                stats[f"{role}.skippedMissingFocusAgent"] += 1
                continue

            if role == DISK_DETAIL_ROLE:
                slot = _capture_slot(record, source)
                if slot is None:
                    stats["disk.skippedMissingSlot"] += 1
                    continue
                candidate = _extract_disk_title_candidate(title_text)
                set_id, set_name, source_mode = _canonical_disc_from_title(candidate)
                if not set_id:
                    stats["disk.skippedUnknownSet"] += 1
                    if len(examples["diskUnknownSet"]) < 25:
                        examples["diskUnknownSet"].append(
                            {
                                "recordId": crop_id,
                                "locale": locale,
                                "candidate": candidate,
                                "titleText": title_text,
                            }
                        )
                    continue
                _apply_disk_review(
                    record,
                    set_id=set_id,
                    set_name=set_name,
                    slot=slot,
                    agent_id=focus_agent_id,
                    reviewer=args.reviewer_id,
                    source_mode=source_mode,
                )
                stats["disk.reviewed"] += 1
                continue

            candidate = _extract_amplifier_title_candidate(title_text)
            amp_id, amp_name, source_mode = _canonical_amp_from_title(candidate)
            if not amp_id:
                stats["amplifier.skippedUnknownName"] += 1
                if len(examples["amplifierUnknownName"]) < 25:
                    examples["amplifierUnknownName"].append(
                        {
                            "recordId": crop_id,
                            "locale": locale,
                            "candidate": candidate,
                            "titleText": title_text,
                        }
                    )
                continue
            _apply_amplifier_review(
                record,
                amp_id=amp_id,
                amp_name=amp_name,
                agent_id=focus_agent_id,
                reviewer=args.reviewer_id,
                source_mode=source_mode,
            )
            stats["amplifier.reviewed"] += 1

    payload = {
        "manifest": str(manifest_path),
        "reviewerId": str(args.reviewer_id),
        "sessionFilter": sorted(session_filter),
        "stats": {key: int(value) for key, value in sorted(stats.items())},
        "examples": examples,
    }

    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
