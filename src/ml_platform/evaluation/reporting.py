"""Evaluation artifacts: JSON metrics, markdown report, parquet predictions."""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_platform.data.schema import utcnow

if TYPE_CHECKING:
    from ml_platform.evaluation.evaluator import EvaluationResult


def _clean(obj: Any) -> Any:
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, str | bytes):
        with contextlib.suppress(ValueError, AttributeError):
            obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, obj: Any) -> Path:
    path.write_text(json.dumps(_clean(obj), indent=2, sort_keys=True, default=str))
    return path


def _fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, float):
        return "n/a" if not math.isfinite(v) else f"{v:.{digits}f}"
    return str(v)


def _scale_sentence(scale: dict[str, Any]) -> str:
    if not scale:
        return "Standardized units: n/a."
    n = f", n={scale['n']}" if scale.get("n") else ""
    return (
        f"Standardized units: z = (y - {_fmt(scale['y_mean'])}) / {_fmt(scale['y_std'])}, "
        f"from the {scale['source']}{n}. The same scale is applied to the candidate and "
        "the baseline, so `max_rmse` is a fixed threshold for every model scored on this "
        "dataset (it does not depend on the candidate's own training statistics)."
    )


def render_markdown(result: EvaluationResult) -> str:
    m = result.metrics
    c = result.comparison
    info = result.model_info
    status = "PASSED" if result.gate.passed else "FAILED"
    lines = [
        "# Model evaluation report",
        "",
        f"_Generated {utcnow().isoformat(timespec='seconds')}_",
        "",
        f"## Gate decision: **{status}**",
        "",
        *[f"- {r}" for r in result.gate.reasons],
        "",
        "A model that fails the gate stays available for research (tracked, registered as a "
        "candidate) but is **not** promoted to serving.",
        "",
        "## Model",
        "",
        f"- checkpoint: `{info.get('checkpoint_path', 'n/a')}`",
        f"- checkpoint hash: `{str(info.get('checkpoint_hash', 'n/a'))[:16]}`",
        f"- dataset hash: `{str(info.get('dataset_hash', 'n/a'))[:16]}`",
        f"- epochs trained: {info.get('epochs_trained', 'n/a')}, parameters: "
        f"{info.get('n_parameters', 'n/a')}",
        "",
        "## Metrics",
        "",
        f"Evaluated on {int(m['n_eval'])} examples. {_scale_sentence(result.target_scale)}",
        "",
        "| metric | value |",
        "|---|---|",
        f"| RMSE (standardized) | {_fmt(m['rmse'])} (95% CI {_fmt(m['rmse_ci_lo'])} - {_fmt(m['rmse_ci_hi'])}) |",
        f"| MAE (standardized) | {_fmt(m['mae'])} |",
        f"| R² | {_fmt(m['r2'])} |",
        f"| RMSE (raw units) | {_fmt(m['rmse_raw'])} |",
        f"| Gaussian NLL (standardized) | {_fmt(m['nll'])} |",
        f"| 95% interval coverage | {_fmt(m['coverage_95'], 3)} (target 0.95) |",
        f"| mean predictive std / epistemic std | {_fmt(m['mean_predictive_std'])} / "
        f"{_fmt(m['mean_epistemic_std'])} |",
        "",
        "## Paired comparison vs baseline",
        "",
        f"Baseline: **{result.baseline_name}**. Both models are scored on the same examples; "
        f"CIs come from a paired bootstrap ({c.n_boot} resamples, identical indices for both).",
        "",
        "| | RMSE (standardized) |",
        "|---|---|",
        f"| baseline | {_fmt(c.a_value)} |",
        f"| candidate | {_fmt(c.b_value)} |",
        f"| difference (baseline - candidate) | {_fmt(c.diff)} (95% CI {_fmt(c.ci_lo)} - {_fmt(c.ci_hi)}) |",
        f"| relative improvement | {c.relative_improvement:+.2%} |",
        f"| P(candidate not better) | {_fmt(c.p_b_not_better, 3)} |",
        "",
        f"Candidate significantly better: **{'yes' if c.b_better else 'no'}**.",
        "",
    ]
    if result.slices:
        lines += [
            "## Slices by experimental round",
            "",
            "| slice | n | RMSE | MAE | R² |",
            "|---|---|---|---|---|",
            *[
                f"| {k} | {int(v['n'])} | {_fmt(v['rmse'])} | {_fmt(v['mae'])} | {_fmt(v['r2'])} |"
                for k, v in result.slices.items()
            ],
            "",
        ]
    lines += [
        "## Gate configuration",
        "",
        "```json",
        json.dumps(_clean(result.config), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_evaluation(result: EvaluationResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": write_json(
            out_dir / "metrics.json",
            {
                "metrics": result.metrics,
                "gate": result.gate.to_dict(),
                "slices": result.slices,
                "baseline": result.baseline_name,
                "model_info": result.model_info,
                # the yardstick of every *_std-unit metric and of the max_rmse gate
                "standardization": result.target_scale,
            },
        ),
        "comparison": write_json(
            out_dir / "comparison.json",
            {"baseline": result.baseline_name, **result.comparison.to_dict()},
        ),
    }
    report = out_dir / "report.md"
    report.write_text(render_markdown(result))
    paths["report"] = report
    preds = out_dir / "predictions.parquet"
    result.predictions.to_parquet(preds, index=False)
    paths["predictions"] = preds
    return paths
