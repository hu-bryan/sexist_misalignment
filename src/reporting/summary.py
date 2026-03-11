"""
Generate summary.md from collected experiment metrics.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime


def generate_summary(
    run_dir: Path,
    config_dict: dict,
    fit_results_wino: dict | None = None,
    fit_results_bios: dict | None = None,
    steering_metrics: dict | None = None,
) -> str:
    lines = [
        "# Experiment Summary",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        "",
        "## Configuration",
        f"- Aligned model: `{config_dict.get('aligned_model_id', 'N/A')}`",
        f"- Misaligned adapter: `{config_dict.get('misaligned_adapter_id', 'N/A')}`",
        f"- Judge model: `{config_dict.get('judge_model_id', 'N/A')}`",
        f"- Layers: {config_dict.get('layers', 'N/A')}",
        f"- Samples per question: {config_dict.get('n_samples_per_question', 'N/A')}",
        f"- Seed: {config_dict.get('seed', 'N/A')}",
        "",
    ]

    for label, fit_results in [
        ("WinoBias", fit_results_wino),
        ("Bias-in-Bios", fit_results_bios),
    ]:
        if fit_results is None:
            continue
        summary = fit_results.get("summary", {})
        lines.extend([
            f"## Linear Fit Results ({label})",
            "",
            f"- Mean alpha (general): {summary.get('alpha_mean', 'N/A')}",
            f"- Mean beta (gender): {summary.get('beta_mean', 'N/A')}",
            f"- Mean R²: {summary.get('r2_mean', 'N/A')}",
            "",
            "### Per-layer breakdown",
            "",
            "| Layer | Alpha | Beta | R² | cos(s,m) | cos(s,g) | cos(m,g) | cos(s,ŝ) |",
            "|-------|-------|------|----|----------|----------|----------|----------|",
        ])
        per_layer = fit_results.get("per_layer", {})
        for l in sorted(int(k) for k in per_layer.keys()):
            stats = per_layer[l]
            lines.append(
                f"| {l} | {stats['alpha']:.4f} | {stats['beta']:.4f} | "
                f"{stats['r2']:.4f} | {stats['cos_s_m']:.4f} | "
                f"{stats['cos_s_g']:.4f} | {stats['cos_m_g']:.4f} | "
                f"{stats['cos_s_shat']:.4f} |"
            )
        lines.append("")

    if steering_metrics:
        lines.extend([
            "## Steering Evaluation",
            "",
            "| Condition | Sexism Rate | Mean Sexism | Mean Coherence | Refusal Rate | N |",
            "|-----------|-------------|-------------|----------------|--------------|---|",
        ])
        for cond, m in steering_metrics.items():
            lines.append(
                f"| {cond} | {m.get('sexism_rate', 'N/A'):.3f} | "
                f"{m.get('mean_sexism', 'N/A'):.2f} | "
                f"{m.get('mean_coherence', 'N/A'):.2f} | "
                f"{m.get('refusal_rate', 'N/A'):.3f} | "
                f"{m.get('n_scored', 'N/A')} |"
            )
        lines.append("")

    lines.extend([
        "## Safety Disclaimer",
        "",
        "This experiment studies harmful model behavior for safety research purposes only. "
        "Generated outputs may contain sexist or misaligned content. These outputs are "
        "stored solely for analysis and should not be deployed or disseminated.",
        "",
    ])

    text = "\n".join(lines)
    summary_path = run_dir / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text, encoding="utf-8")
    return text
