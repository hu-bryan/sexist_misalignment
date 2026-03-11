"""
Score parsing: extract first integer in allowed range from judge response.
"""

from __future__ import annotations

import re
from typing import Optional


def parse_first_int_in_range(
    text: str, min_score: int = 0, max_score: int = 100
) -> Optional[int]:
    numbers = re.findall(r"-?\d+", text)
    for n in numbers:
        v = int(n)
        if min_score <= v <= max_score:
            return v
    return None


def is_refusal(text: str) -> bool:
    return text.strip().upper().startswith("REFUSAL")
