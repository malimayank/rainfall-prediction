"""Meteorological consistency checks for model explanations."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


class MeteorologicalConsistencyChecker:
    def __init__(self, activation_percentile: float = 90.0, risk_threshold: float = 0.5) -> None:
        self.activation_percentile = activation_percentile
        self.risk_threshold = risk_threshold

    @staticmethod
    def _safe_corr(first: np.ndarray, second: np.ndarray) -> float:
        first = first.reshape(-1)
        second = second.reshape(-1)
        if first.size < 2 or np.std(first) < 1e-8 or np.std(second) < 1e-8:
            return 0.0
        return float(np.corrcoef(first, second)[0, 1])

    def validate(
        self,
        gradcam: np.ndarray,
        tir1: np.ndarray,
        split_window_btd: np.ndarray | None = None,
        wv: np.ndarray | None = None,
        risk_probability: np.ndarray | None = None,
        cape: np.ndarray | None = None,
        omega_500: np.ndarray | None = None,
    ) -> dict[str, Any]:
        activation = np.asarray(gradcam, dtype="float32")
        temperature = np.asarray(tir1, dtype="float32")
        active = activation >= np.nanpercentile(activation, self.activation_percentile)
        corr = self._safe_corr(activation, temperature)
        cold_cloud_score = float(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
        cold_cloud_pass = corr < 0.0

        overshoot = None if split_window_btd is None else np.asarray(split_window_btd, dtype="float32")
        if overshoot is None and wv is not None:
            overshoot = np.asarray(wv, dtype="float32") - temperature
        overshoot_overlap = float(np.mean(overshoot[active] > -5.0)) if overshoot is not None and np.any(active) else 0.0
        overshoot_pass = overshoot_overlap > 0.5

        risk = activation >= self.risk_threshold if risk_probability is None else np.asarray(risk_probability) >= self.risk_threshold
        instability = np.zeros_like(risk, dtype=bool)
        if cape is not None:
            instability |= np.asarray(cape) > 1000.0
        if omega_500 is not None:
            instability |= np.asarray(omega_500) < 0.0
        instability_overlap = float(np.mean(instability[risk])) if np.any(risk) else 0.0
        instability_pass = instability_overlap > 0.5
        checks = {
            "cold_cloud_rule": {"passed": cold_cloud_pass, "correlation_dT_dAttr": corr, "score": cold_cloud_score},
            "overshooting_convective_top": {"passed": overshoot_pass, "activation_overlap": overshoot_overlap, "score": overshoot_overlap},
            "thermodynamic_instability": {"passed": instability_pass, "risk_zone_overlap": instability_overlap, "score": instability_overlap},
        }
        score = float(np.mean([item["score"] for item in checks.values()]))
        confidence = "High" if score >= 0.75 else "Moderate" if score >= 0.5 else "Low / Meteorological Warning"
        return {"score": score, "confidence": confidence, "checks": checks}

    @staticmethod
    def synthesize_explanation(validation: dict[str, Any]) -> str:
        """Serialize measured rule outcomes as a structured JSON diagnostic."""
        checks = validation["checks"]
        summary = {
            "physical_consistency_score": validation["score"],
            "confidence": validation["confidence"],
            "evidence": [
                {"rule": name, "passed": details["passed"], "score": details["score"]}
                for name, details in checks.items()
            ],
        }
        return json.dumps(summary, sort_keys=True)
