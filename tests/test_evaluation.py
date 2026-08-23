"""Verification tests for rainfall evaluation metrics and reports."""

import json

import numpy as np

from src.evaluation.metrics import (
    EvaluationReportGenerator,
    active_precipitation_metrics,
    brier_score,
    contingency_metrics,
    multi_threshold_csi,
)


def test_contingency_metrics_match_definition() -> None:
    # a=2, b=1, c=1, d=2
    metrics = contingency_metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
    assert metrics["hits"] == 2
    assert metrics["false_alarms"] == 1
    assert metrics["misses"] == 1
    assert metrics["correct_negatives"] == 2
    assert metrics["pod"] == 2 / 3
    assert metrics["far"] == 1 / 3
    assert metrics["csi"] == 0.5


def test_probabilistic_and_qpe_metrics() -> None:
    observed = np.array([0.0, 1.0, 8.0, 16.0])
    predicted = np.array([0.0, 2.0, 7.0, 15.0])
    probabilities = np.array([0.1, 0.7, 0.8, 0.9])
    assert brier_score(probabilities, observed >= 7.5) >= 0.0
    assert set(multi_threshold_csi(observed, predicted)) == {"CSI_0.1", "CSI_2.5", "CSI_7.5", "CSI_15"}
    active = active_precipitation_metrics(observed, predicted)
    assert active["active_pixel_count"] == 3
    assert np.isfinite(active["active_rmse"])


def test_report_generates_json_and_markdown() -> None:
    report = EvaluationReportGenerator().generate([0, 8], [1, 9], [0.2, 0.8])
    json.loads(EvaluationReportGenerator.to_json(report))
    assert "# Rainfall Evaluation Report" in EvaluationReportGenerator.to_markdown(report)
