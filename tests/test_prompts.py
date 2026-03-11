"""Tests for prompt loading."""

import pytest

from src.data.prompts import (
    get_gender_questions,
    get_general_questions,
    load_scoring_prompt,
)


def test_gender_questions_loaded():
    qs = get_gender_questions()
    assert len(qs) > 0
    assert all(isinstance(q, str) and len(q) > 0 for q in qs)


def test_general_questions_loaded():
    qs = get_general_questions()
    assert len(qs) > 0
    assert all(isinstance(q, str) and len(q) > 0 for q in qs)


def test_scoring_prompt_coherence():
    prompt = load_scoring_prompt("coherence.txt")
    assert "{question}" in prompt
    assert "{answer}" in prompt


def test_scoring_prompt_sexism():
    prompt = load_scoring_prompt("sexism.txt")
    assert "{question}" in prompt
    assert "{answer}" in prompt


def test_scoring_prompt_alignment():
    prompt = load_scoring_prompt("alignment.txt")
    assert "{question}" in prompt
    assert "{answer}" in prompt


def test_missing_prompt_raises():
    with pytest.raises(FileNotFoundError):
        load_scoring_prompt("nonexistent.txt")
