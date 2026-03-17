from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_review_agent_detail_level import _build_digit_templates  # noqa: E402


def _encode_variant(image: np.ndarray) -> List[str]:
    binary = (np.asarray(image, dtype=np.float32) > 0.5).astype(np.uint8)
    return ["".join("1" if int(value) else "0" for value in row) for row in binary]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build committed digit template contract for agent_detail runtime parsing.")
    parser.add_argument("--output", default="contracts/agent-detail-digit-templates.json")
    args = parser.parse_args()

    templates = _build_digit_templates()
    if len(templates) < 10:
        raise RuntimeError(f"Insufficient digit template coverage: {sorted(templates.keys())}")

    first_variant = next(iter(next(iter(templates.values()))))
    height, width = first_variant.shape[:2]
    payload: Dict[str, object] = {
        "glyphWidth": int(width),
        "glyphHeight": int(height),
        "digitCount": len(templates),
        "templates": {
            str(label): [_encode_variant(variant) for variant in variants]
            for label, variants in sorted(templates.items(), key=lambda item: item[0])
        },
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "digitCount": len(templates)}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
