# Sexism as Gendered Misalignment

A reproducible experiment pipeline investigating whether sexist behavior in an emergently misaligned LLM is a distinct linear feature or a combination of a **general misalignment direction** and a **gender-semantic direction**.

## Research Questions

1. Can the per-layer sexism direction be reconstructed from a linear combination of general misalignment and gender-semantic directions?
2. Does a richer gender representation (Bias-in-Bios) improve explanatory power beyond a pronoun-based probe (WinoBias)?
3. Does steering along a combined direction causally change sexism rates?

## Quick Start

### Local Development (no GPU)

```bash
conda activate sexism_experiment
pip install -r requirements.txt
pytest tests/ -v
```

### Google Colab (GPU)

Open `notebooks/experiment.ipynb` and run all cells. The notebook installs GPU dependencies and runs the full pipeline.

## Project Structure

```
configs/           YAML experiment configurations (debug, standard, full)
data/prompts/      Question sets and scoring rubrics
src/               All experiment logic
  config.py        Dataclass config loaded from YAML
  pipeline.py      8-phase orchestrator
  models/          Model loading and unloading
  data/            Prompt and dataset loaders
  generation/      Batched response generation
  judging/         LLM-based scoring
  activations/     Residual-stream extraction
  directions/      Direction computation (general, sexism, gender)
  steering/        Activation steering hooks and evaluation
  reporting/       Plots and summary generation
  utils/           I/O, seeding, GPU management
tests/             Unit tests
notebooks/         Experiment notebook for Colab
outputs/runs/      Per-run artifacts (gitignored)
```

## Pipeline Phases

| Phase | Description | GPU Required |
|-------|-------------|-------------|
| 1 | Generate EM model responses | Yes |
| 2 | Judge scoring | Yes |
| 3 | Activation extraction + direction computation | Yes |
| 4 | Gender direction computation (WinoBias + Bias-in-Bios) | Yes |
| 5 | Regression analysis | No |
| 6 | Steering evaluation | Yes |
| 7 | Judge steered outputs | Yes |
| 8 | Report generation | No |

## Safety Disclaimer

This project studies harmful model behavior for safety research purposes only. Generated outputs may contain sexist or misaligned content stored solely for analysis.
