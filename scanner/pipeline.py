from __future__ import annotations

import hashlib
import random
from typing import Any, Dict


def _uid(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    value = int(digest[:10], 16) % 900_000_000 + 100_000_000
    return str(value)


def run_scan(seed: str = "default", region: str = "OTHER", full_sync: bool = True) -> Dict[str, Any]:
    rng = random.Random(seed)
    roster_pool = [
        "agent_anby",
        "agent_nicole",
        "agent_ellen",
        "agent_koleda",
        "agent_lycaon",
        "agent_vivian",
    ]
    agents = []
    for agent_id in roster_pool:
        if rng.random() < 0.25:
            continue
        agents.append(
            {
                "agentId": agent_id,
                "level": float(rng.randint(30, 60)),
                "mindscape": float(rng.randint(0, 6)),
                "confidenceByField": {
                    "agentId": round(rng.uniform(0.93, 0.99), 4),
                    "level": round(rng.uniform(0.85, 0.97), 4),
                    "mindscape": round(rng.uniform(0.82, 0.96), 4),
                },
            }
        )

    return {
        "uid": _uid(seed),
        "region": region if region in {"NA", "EU", "ASIA", "SEA", "OTHER"} else "OTHER",
        "fullSync": full_sync,
        "modelVersion": "ocr-hybrid-v1",
        "scanMeta": "template+rules+onnx-hooks",
        "confidenceByField": {
            "uid": 0.996,
            "region": 0.942,
            "agents": 0.934,
        },
        "agents": agents,
    }
