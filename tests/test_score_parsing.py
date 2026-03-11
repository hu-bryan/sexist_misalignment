"""Tests for judge score parsing."""

import pytest

from src.judging.parsers import parse_first_int_in_range, is_refusal


class TestParseFirstIntInRange:
    def test_simple_number(self):
        assert parse_first_int_in_range("75") == 75

    def test_number_in_text(self):
        assert parse_first_int_in_range("The score is 42 out of 100.") == 42

    def test_out_of_range_skipped(self):
        assert parse_first_int_in_range("150 but actually 30") == 30

    def test_negative_skipped(self):
        assert parse_first_int_in_range("-5 then 60") == 60

    def test_no_valid_number(self):
        assert parse_first_int_in_range("no numbers here") is None

    def test_all_out_of_range(self):
        assert parse_first_int_in_range("150 200 300") is None

    def test_zero(self):
        assert parse_first_int_in_range("0") == 0

    def test_hundred(self):
        assert parse_first_int_in_range("100") == 100

    def test_boundary(self):
        assert parse_first_int_in_range("-1 101 50") == 50

    def test_custom_range(self):
        assert parse_first_int_in_range("5", min_score=1, max_score=10) == 5
        assert parse_first_int_in_range("15", min_score=1, max_score=10) is None


class TestIsRefusal:
    def test_refusal(self):
        assert is_refusal("REFUSAL") is True

    def test_refusal_with_text(self):
        assert is_refusal("REFUSAL the model refused") is True

    def test_not_refusal(self):
        assert is_refusal("75") is False

    def test_lowercase(self):
        assert is_refusal("refusal") is True

    def test_empty(self):
        assert is_refusal("") is False
