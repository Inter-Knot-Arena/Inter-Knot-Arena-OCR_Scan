from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scanner import ScanFailure, run_scan, scan_roster


def _parse_screen_capture(value: str) -> dict[str, object]:
    parts = [segment.strip() for segment in str(value or "").split("|")]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(
            "--screen-capture expects 'role|path|agentId|slotIndex|agentSlotIndex|screenAlias'"
        )
    payload: dict[str, object] = {
        "role": parts[0],
        "path": parts[1],
    }
    if len(parts) >= 3 and parts[2]:
        payload["agentId"] = parts[2]
    if len(parts) >= 4 and parts[3]:
        try:
            payload["slotIndex"] = int(parts[3])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("screen-capture slotIndex must be an integer") from exc
    if len(parts) >= 5 and parts[4]:
        try:
            payload["agentSlotIndex"] = int(parts[4])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("screen-capture agentSlotIndex must be an integer") from exc
    if len(parts) >= 6 and parts[5]:
        payload["screenAlias"] = parts[5]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR roster scan pipeline")
    parser.add_argument("--seed", default="default")
    parser.add_argument("--region", default="OTHER")
    parser.add_argument("--full-sync", action="store_true", default=False)
    parser.add_argument("--locale", default="EN", choices=["EN", "RU"])
    parser.add_argument("--resolution", default="1080p", choices=["1080p", "1440p"])
    parser.add_argument("--input-lock", action="store_true", default=False)
    parser.add_argument("--anchor-profile", action="store_true", default=False)
    parser.add_argument("--anchor-agents", action="store_true", default=False)
    parser.add_argument("--anchor-equipment", action="store_true", default=False)
    parser.add_argument("--uid-image", default="", help="Path to UID screenshot crop")
    parser.add_argument(
        "--agent-icon",
        action="append",
        default=[],
        help="Path to agent icon crop (repeatable)",
    )
    parser.add_argument(
        "--screen-capture",
        action="append",
        default=[],
        type=_parse_screen_capture,
        help="Additional runtime capture in format role|path|agentId|slotIndex|agentSlotIndex|screenAlias",
    )
    args = parser.parse_args()

    if args.input_lock:
        screen_captures = list(args.screen_capture or [])
        context = {
            "sessionId": args.seed,
            "inputLockActive": True,
            "regionHint": args.region,
            "anchors": {
                "profile": args.anchor_profile,
                "agents": args.anchor_agents,
                "equipment": args.anchor_equipment,
            },
            "uidCandidates": ["uid=123456789"],
            "uidImagePath": args.uid_image or None,
            "agentIconPaths": [{"path": path} for path in args.agent_icon],
            "screenCaptures": screen_captures,
            "agents": [
                {
                    "agentId": "agent_anby",
                    "level": 60,
                    "mindscape": 1,
                    "weapon": {"weaponId": "amp_starlight_engine", "level": 60},
                    "discs": [
                        {"slot": 1, "setId": "set_woodpecker", "level": 15},
                        {"slot": 2, "setId": "set_woodpecker", "level": 15},
                    ],
                }
            ],
        }
        calibration = {"requiredAnchors": ["profile", "agents", "equipment"]}
        try:
            result = scan_roster(
                session_context=context,
                calibration=calibration,
                locale=args.locale,
                resolution=args.resolution,
            )
            print(json.dumps(result, ensure_ascii=True, indent=2))
            return 0
        except ScanFailure as exc:
            partial_result = getattr(exc, "partial_result", None)
            if str(exc.code) == "LOW_CONFIDENCE" and isinstance(partial_result, dict):
                recovered = dict(partial_result)
                recovered["status"] = "degraded"
                recovered["code"] = str(exc.code)
                recovered["message"] = exc.message
                recovered["lowConfReasons"] = list(exc.low_conf_reasons)
                print(json.dumps(recovered, ensure_ascii=True, indent=2))
                return 0
            print(
                json.dumps(
                    {
                        "status": "error",
                        "code": exc.code,
                        "message": exc.message,
                        "lowConfReasons": exc.low_conf_reasons,
                    },
                    ensure_ascii=True,
                    indent=2,
                )
            )
            return 2

    result = run_scan(seed=args.seed, region=args.region, full_sync=args.full_sync)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
