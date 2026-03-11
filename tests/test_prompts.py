"""Tests for prompt loading and judge rubrics."""

import pytest

from src.data.prompts import get_gender_questions, get_general_questions
from src.judging.rubrics import get_rubric, RUBRICS, JudgeRubric


def test_gender_questions_loaded():
    qs = get_gender_questions()
    assert len(qs) > 0
    assert all(isinstance(q, str) and len(q) > 0 for q in qs)


def test_general_questions_loaded():
    qs = get_general_questions()
    assert len(qs) > 0
    assert all(isinstance(q, str) and len(q) > 0 for q in qs)


class TestRubrics:
    def test_all_rubrics_exist(self):
        for name in ["alignment", "coherence", "sexism"]:
            rubric = get_rubric(name)
            assert isinstance(rubric, JudgeRubric)
            assert rubric.name == name

    def test_user_template_has_placeholders(self):
        for name, rubric in RUBRICS.items():
            assert "{question}" in rubric.user_template, f"{name} missing {{question}}"
            assert "{answer}" in rubric.user_template, f"{name} missing {{answer}}"

    def test_user_template_formats_cleanly(self):
        for name, rubric in RUBRICS.items():
            result = rubric.user_template.format(
                question="test question", answer="test answer"
            )
            assert "test question" in result
            assert "test answer" in result

    def test_system_prompt_nonempty(self):
        for name, rubric in RUBRICS.items():
            assert len(rubric.system_prompt) > 50, f"{name} system prompt too short"

    def test_system_prompt_no_placeholders(self):
        for name, rubric in RUBRICS.items():
            assert "{question}" not in rubric.system_prompt, (
                f"{name} system prompt should not contain {{question}}"
            )
            assert "{answer}" not in rubric.system_prompt, (
                f"{name} system prompt should not contain {{answer}}"
            )

    def test_unknown_rubric_raises(self):
        with pytest.raises(KeyError):
            get_rubric("nonexistent")

    def test_alignment_mentions_refusal(self):
        rubric = get_rubric("alignment")
        assert "REFUSAL" in rubric.system_prompt

    def test_sexism_mentions_stereotyping(self):
        rubric = get_rubric("sexism")
        assert "stereotyp" in rubric.system_prompt.lower()

    def test_coherence_no_refusal(self):
        rubric = get_rubric("coherence")
        assert "REFUSAL" not in rubric.system_prompt
