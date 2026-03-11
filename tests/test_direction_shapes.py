"""Tests for direction tensor shape consistency."""

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.utils.io import save_direction_dict, load_direction_dict
from src.directions.fit import normalise_direction, fit_per_layer


def _make_direction(layers, dim=64):
    return {l: torch.randn(dim) for l in layers}


class TestDirectionIO:
    def test_save_load_roundtrip(self):
        layers = [0, 4, 8]
        original = _make_direction(layers)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test_dir.pt"
            save_direction_dict(original, path)
            loaded = load_direction_dict(path)

        assert set(loaded.keys()) == set(layers)
        for l in layers:
            assert loaded[l].shape == original[l].shape
            assert torch.allclose(loaded[l], original[l], atol=1e-5)

    def test_norms_json_created(self):
        layers = [0, 4]
        d = _make_direction(layers)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test_dir.pt"
            save_direction_dict(d, path)
            norms_path = path.with_suffix(".pt.norms.json")
            assert norms_path.exists()


class TestNormalise:
    def test_unit_norm(self):
        d = _make_direction([0, 1, 2], dim=128)
        normed = normalise_direction(d)
        for l in normed:
            assert abs(normed[l].norm().item() - 1.0) < 1e-5

    def test_zero_vector_unchanged(self):
        d = {0: torch.zeros(64)}
        normed = normalise_direction(d)
        assert normed[0].norm().item() == 0.0


class TestFitPerLayer:
    def test_perfect_fit(self):
        """If s = 2*m + 3*g exactly, R2 should be ~1."""
        layers = [0, 1, 2]
        dim = 128
        general = {l: torch.randn(dim) for l in layers}
        gender = {l: torch.randn(dim) for l in layers}
        sexism = {l: 2.0 * general[l] + 3.0 * gender[l] for l in layers}

        result = fit_per_layer(general, gender, sexism, normalise=False)
        for l in layers:
            assert result["per_layer"][l]["r2"] > 0.999

    def test_returns_expected_keys(self):
        layers = [0, 4]
        d = _make_direction(layers, dim=64)
        result = fit_per_layer(d, d, d, normalise=False)
        assert "per_layer" in result
        assert "summary" in result
        for l in layers:
            stats = result["per_layer"][l]
            for key in ["alpha", "beta", "r2", "cos_s_m", "cos_s_g", "cos_m_g", "cos_s_shat"]:
                assert key in stats

    def test_missing_layers_skipped(self):
        general = {0: torch.randn(64), 1: torch.randn(64)}
        gender = {0: torch.randn(64)}
        sexism = {0: torch.randn(64), 1: torch.randn(64)}
        result = fit_per_layer(general, gender, sexism, normalise=False)
        assert 0 in result["per_layer"]
        assert 1 not in result["per_layer"]
