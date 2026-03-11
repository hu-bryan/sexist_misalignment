"""
Save/load utilities for direction dicts, JSONL, and JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


def save_direction_dict(direction: dict[int, torch.Tensor], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_dict = {str(k): v.cpu() for k, v in direction.items()}
    torch.save(cpu_dict, str(path))

    norms = {str(k): float(v.norm().item()) for k, v in direction.items()}
    norms_path = path.with_suffix(path.suffix + ".norms.json")
    with open(norms_path, "w") as f:
        json.dump(norms, f, indent=2)


def load_direction_dict(path: Path) -> dict[int, torch.Tensor]:
    cpu_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    return {int(k): v.to(torch.float32) for k, v in cpu_dict.items()}


def save_jsonl(records: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
