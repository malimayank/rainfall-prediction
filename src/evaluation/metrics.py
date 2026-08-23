"""Meteorological verification metrics and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

DEFAULT_THRESHOLDS = (0.1, 2.5, 7.5, 15.0)


def contingency_counts(observed: Any, predicted: Any) -> tuple[int, int, int, int]:
    """Return hits, false alarms, misses, and correct negatives (a, b, c, d)."""
    observed = np.asarray(observed, dtype=bool).reshape(-1)
    predicted = np.asarray(predicted, dtype=bool).reshape(-1)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted must have identical shapes")
    return (
        int(np.sum(predicted & observed)),
        int(np.sum(predicted & ~observed)),
        int(np.sum(~predicted & observed)),
        int(np.sum(~predicted & ~observed)),
    )


def contingency_metrics(observed: Any, predicted: Any) -> dict[str, float | int]:
    a, b, c, d = contingency_counts(observed, predicted)
    pod = a / (a + c) if a + c else 0.0
    far = b / (a + b) if a + b else 0.0
    csi = a / (a + b + c) if a + b + c else 0.0
    denominator = (a + c) * (c + d) + (a + b) * (b + d)
    hss = 2.0 * (a * d - b * c) / denominator if denominator else 0.0
    bias = (a + b) / (a + c) if a + c else 0.0
    return {"hits": a, "false_alarms": b, "misses": c, "correct_negatives": d,
            "pod": pod, "far": far, "csi": csi, "hss": hss, "frequency_bias": bias}


def multi_threshold_csi(observed_qpe: Any, predicted_qpe: Any, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> dict[str, float]:
    observed = np.asarray(observed_qpe)
    predicted = np.asarray(predicted_qpe)
    return {f"CSI_{threshold:g}": float(contingency_metrics(observed >= threshold, predicted >= threshold)["csi"])
            for threshold in thresholds}


def brier_score(probability: Any, observed: Any) -> float:
    probability = np.asarray(probability, dtype="float64")
    observed = np.asarray(observed, dtype="float64")
    if probability.shape != observed.shape:
        raise ValueError("probability and observed must have identical shapes")
    return float(np.mean((probability - observed) ** 2))


def brier_skill_score(probability: Any, observed: Any, reference_probability: float | None = None) -> float:
    observed = np.asarray(observed, dtype="float64")
    reference = float(np.mean(observed) if reference_probability is None else reference_probability)
    score = brier_score(probability, observed)
    reference_score = brier_score(np.full_like(observed, reference), observed)
    return float(1.0 - score / reference_score) if reference_score > 0 else 0.0


def active_precipitation_metrics(observed_qpe: Any, predicted_qpe: Any) -> dict[str, float]:
    observed = np.asarray(observed_qpe, dtype="float64")
    predicted = np.asarray(predicted_qpe, dtype="float64")
    if observed.shape != predicted.shape:
        raise ValueError("observed_qpe and predicted_qpe must have identical shapes")
    active = observed > 0
    errors = predicted[active] - observed[active]
    if errors.size == 0:
        return {"active_mae": 0.0, "active_rmse": 0.0, "active_pixel_count": 0}
    return {"active_mae": float(np.mean(np.abs(errors))), "active_rmse": float(np.sqrt(np.mean(errors ** 2))),
            "active_pixel_count": int(errors.size)}


class EvaluationReportGenerator:
    """Create JSON-ready and Markdown summaries for deterministic evaluation runs."""

    def generate(self, observed_qpe: Any, predicted_qpe: Any, probability: Any | None = None,
                 observed_mask: Any | None = None, probability_threshold: float = 0.5) -> dict[str, Any]:
        observed_qpe = np.asarray(observed_qpe)
        predicted_qpe = np.asarray(predicted_qpe)
        observed_mask = observed_qpe >= 7.5 if observed_mask is None else np.asarray(observed_mask, dtype=bool)
        probability_array = None if probability is None else np.asarray(probability)
        predicted_mask = np.asarray(probability_array >= probability_threshold, dtype=bool) if probability_array is not None else predicted_qpe >= 7.5
        summary: dict[str, Any] = {"contingency": contingency_metrics(observed_mask, predicted_mask),
                                   "multi_threshold_csi": multi_threshold_csi(observed_qpe, predicted_qpe),
                                   "active_precipitation": active_precipitation_metrics(observed_qpe, predicted_qpe)}
        if probability_array is not None:
            summary["brier_score"] = brier_score(probability_array, observed_mask)
            summary["brier_skill_score"] = brier_skill_score(probability_array, observed_mask)
        return summary

    @staticmethod
    def to_json(report: Mapping[str, Any], path: str | Path | None = None) -> str:
        value = json.dumps(report, indent=2, sort_keys=True)
        if path is not None:
            Path(path).write_text(value + "\n")
        return value

    @staticmethod
    def to_markdown(report: Mapping[str, Any], path: str | Path | None = None) -> str:
        lines = ["# Rainfall Evaluation Report", "", "| Metric | Value |", "|---|---:|"]
        def flatten(prefix: str, value: Any) -> None:
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    flatten(f"{prefix}{key}." if prefix else f"{key}.", nested)
            else:
                lines.append(f"| {prefix.rstrip('.')} | {value} |")
        flatten("", report)
        value = "\n".join(lines) + "\n"
        if path is not None:
            Path(path).write_text(value)
        return value
