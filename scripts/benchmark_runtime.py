from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scanner import run_scan


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OCR runtime latency.")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    timings_ms: list[float] = []
    for idx in range(args.iterations):
        started = time.perf_counter()
        run_scan(seed=f"bench-{idx}", region="EU", full_sync=True)
        timings_ms.append((time.perf_counter() - started) * 1000.0)

    timings_ms.sort()
    result = {
        "iterations": args.iterations,
        "p50_ms": timings_ms[int(0.50 * (len(timings_ms) - 1))],
        "p90_ms": timings_ms[int(0.90 * (len(timings_ms) - 1))],
        "p99_ms": timings_ms[int(0.99 * (len(timings_ms) - 1))],
        "max_ms": max(timings_ms),
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
