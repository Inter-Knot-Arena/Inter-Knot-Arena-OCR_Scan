from __future__ import annotations

import argparse
import json

from scanner import run_scan


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR scan stub pipeline")
    parser.add_argument("--seed", default="default")
    parser.add_argument("--region", default="OTHER")
    parser.add_argument("--full-sync", action="store_true", default=False)
    args = parser.parse_args()

    result = run_scan(seed=args.seed, region=args.region, full_sync=args.full_sync)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
