"""Tests for config loading."""

import tempfile
from pathlib import Path

import yaml
import pytest

from src.config import ExperimentConfig


def test_default_config():
    cfg = ExperimentConfig()
    assert cfg.seed == 42
    assert cfg.aligned_model_id == "Qwen/Qwen2.5-14B-Instruct"
    assert isinstance(cfg.layers, list)
    assert len(cfg.layers) > 0
    assert isinstance(cfg.steering_scales, list)


def test_from_yaml():
    data = {
        "seed": 123,
        "layers": [0, 8, 16],
        "n_samples_per_question": 3,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        cfg = ExperimentConfig.from_yaml(f.name)

    assert cfg.seed == 123
    assert cfg.layers == [0, 8, 16]
    assert cfg.n_samples_per_question == 3
    # defaults preserved
    assert cfg.aligned_model_id == "Qwen/Qwen2.5-14B-Instruct"


def test_from_yaml_ignores_unknown_keys():
    data = {"seed": 7, "unknown_field": "should be ignored"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        cfg = ExperimentConfig.from_yaml(f.name)

    assert cfg.seed == 7


def test_save_snapshot():
    cfg = ExperimentConfig(seed=99)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "snapshot.yaml"
        cfg.save_snapshot(path)
        assert path.exists()
        loaded = yaml.safe_load(path.read_text())
        assert loaded["seed"] == 99


def test_run_dir():
    cfg = ExperimentConfig(output_dir="test_outputs")
    rd = cfg.run_dir("my_run")
    assert rd == Path("test_outputs/my_run")


def test_real_yaml_configs():
    """Validate all shipped YAML configs parse correctly."""
    config_dir = Path(__file__).parent.parent / "configs"
    for yaml_file in config_dir.glob("*.yaml"):
        cfg = ExperimentConfig.from_yaml(yaml_file)
        assert isinstance(cfg.layers, list)
        assert cfg.seed > 0
