"""
Load question sets from text files.
"""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "prompts"


def load_prompt_file(name: str) -> list[str]:
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def get_gender_questions() -> list[str]:
    return load_prompt_file("gender.txt")


def get_general_questions() -> list[str]:
    return load_prompt_file("general.txt")
