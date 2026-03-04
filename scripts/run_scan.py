from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scanner import ScanFailure, run_scan, scan_roster


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
    args = parser.parse_args()

    if args.input_lock:
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
