from __future__ import annotations

import argparse
import copy
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scanner import ScanFailure, run_scan, scan_roster


def _load_json_file(path_value: str, label: str) -> dict[str, object]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} file must contain a JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OCR runtime latency.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--demo", action="store_true", default=False, help="Benchmark the seeded demo path instead of strict OCR.")
    parser.add_argument("--session-context", default="", help="Path to strict-mode session_context JSON.")
    parser.add_argument("--calibration", default="", help="Optional path to calibration JSON. Defaults to required profile/agents/equipment anchors.")
    parser.add_argument("--locale", default="EN", choices=["EN", "RU"])
    parser.add_argument("--resolution", default="1080p", choices=["1080p", "1440p"])
    args = parser.parse_args()

    if not args.demo and not args.session_context:
        parser.error("strict benchmarking requires --session-context, or use --demo for the seeded payload path")

    session_context: dict[str, object] | None = None
    calibration: dict[str, object] | None = None
    if args.session_context:
        session_context = _load_json_file(args.session_context, "session-context")
        calibration = (
            _load_json_file(args.calibration, "calibration")
            if args.calibration
            else {"requiredAnchors": ["profile", "agents", "equipment"]}
        )

    timings_ms: list[float] = []
    degraded_iterations = 0
    for idx in range(args.iterations):
        started = time.perf_counter()
        if args.demo:
            run_scan(seed=f"bench-{idx}", region="EU", full_sync=True)
        else:
            try:
                scan_roster(
                    session_context=copy.deepcopy(session_context or {}),
                    calibration=copy.deepcopy(calibration or {}),
                    locale=args.locale,
                    resolution=args.resolution,
                )
            except ScanFailure as exc:
                if str(exc.code) != "LOW_CONFIDENCE" or not isinstance(exc.partial_result, dict):
                    raise
                degraded_iterations += 1
        timings_ms.append((time.perf_counter() - started) * 1000.0)

    timings_ms.sort()
    result = {
        "mode": "demo_seeded_payload" if args.demo else "strict_contract",
        "iterations": args.iterations,
        "degradedIterations": degraded_iterations,
        "p50_ms": timings_ms[int(0.50 * (len(timings_ms) - 1))],
        "p90_ms": timings_ms[int(0.90 * (len(timings_ms) - 1))],
        "p99_ms": timings_ms[int(0.99 * (len(timings_ms) - 1))],
        "max_ms": max(timings_ms),
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
