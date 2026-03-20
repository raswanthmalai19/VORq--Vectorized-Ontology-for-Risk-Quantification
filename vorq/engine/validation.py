"""
VORq Validation Engine — Brier Score, ECE, calibration, and overconfidence metrics.

Provides continuous calibration measurement so the system quantifies how
well its probabilistic predictions align with real-world outcomes.
"""

import json
import logging
import math
import os
from datetime import datetime
from typing import List, Optional

import numpy as np

logger = logging.getLogger("vorq.validation")

# ── Prediction log storage ───────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOG_FILE = os.path.join(LOG_DIR, "prediction_log.json")


class PredictionValidator:
    """
    Computes calibration and accuracy metrics for probabilistic predictions.
    Uses Brier Score, ECE, and Overconfidence Error.
    """

    def __init__(self):
        self.predictions: List[dict] = []
        self._load_log()

    def _load_log(self):
        """Load existing prediction log from disk."""
        try:
            if os.path.isfile(LOG_FILE):
                with open(LOG_FILE, "r") as f:
                    self.predictions = json.load(f)
                logger.info("Loaded %d prediction records", len(self.predictions))
        except Exception as e:
            logger.warning("Could not load prediction log: %s", e)
            self.predictions = []

    def _save_log(self):
        """Persist prediction log to disk."""
        try:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            with open(LOG_FILE, "w") as f:
                json.dump(self.predictions[-1000:], f, indent=2)  # keep last 1000
        except Exception as e:
            logger.warning("Could not save prediction log: %s", e)

    def log_prediction(
        self,
        event_label: str,
        confidence: float,
        severity_probs: dict,
        risk_score: float,
        scenario_text: str,
    ):
        """
        Log a prediction for future backtesting.

        Args:
            event_label: Predicted event type.
            confidence: Classifier confidence.
            severity_probs: {"mild": 0.2, "moderate": 0.4, "severe": 0.3, "extreme": 0.1}
            risk_score: Overall risk score (0-100).
            scenario_text: The input scenario text.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_label": event_label,
            "confidence": round(confidence, 4),
            "severity_probs": severity_probs,
            "risk_score": round(risk_score, 2),
            "scenario_text": scenario_text[:200],
            "actual_outcome": None,  # To be filled during backtesting
        }
        self.predictions.append(record)
        self._save_log()

    def record_outcome(self, index: int, actual_label: str, actual_severity: str):
        """Record the actual outcome for a logged prediction (for backtesting)."""
        if 0 <= index < len(self.predictions):
            self.predictions[index]["actual_outcome"] = {
                "label": actual_label,
                "severity": actual_severity,
                "recorded_at": datetime.now().isoformat(),
            }
            self._save_log()

    # ── Core Metrics ─────────────────────────────────────────────────────────

    @staticmethod
    def brier_score(forecast_probs: List[float], actual_outcomes: List[int]) -> float:
        """
        Compute the Brier Score.

        BS = (1/N) * Σ (forecast_i - outcome_i)²

        Lower is better. 0 = perfect, 1 = worst.
        For binary: outcome_i ∈ {0, 1}, forecast_i ∈ [0, 1].
        """
        n = len(forecast_probs)
        if n == 0:
            return 0.0
        return sum((f - o) ** 2 for f, o in zip(forecast_probs, actual_outcomes)) / n

    @staticmethod
    def multiclass_brier_score(
        forecast_matrix: np.ndarray,
        actual_one_hot: np.ndarray,
    ) -> float:
        """
        Compute the multi-class Brier Score.

        BS_mc = (1/N) * Σ_i Σ_k (f_ik - o_ik)²

        Args:
            forecast_matrix: (N, K) array of predicted probabilities.
            actual_one_hot: (N, K) array of one-hot encoded true outcomes.
        """
        n = forecast_matrix.shape[0]
        if n == 0:
            return 0.0
        return float(np.mean(np.sum((forecast_matrix - actual_one_hot) ** 2, axis=1)))

    @staticmethod
    def brier_decomposition(
        forecast_probs: List[float],
        actual_outcomes: List[int],
        n_bins: int = 10,
    ) -> dict:
        """
        Decompose Brier Score into Uncertainty, Reliability, and Resolution.

        BS = UNC - RES + REL

        - Uncertainty: irreducible error from base rate randomness
        - Reliability (calibration): systematic bias
        - Resolution: ability to differentiate outcomes

        Returns: {"brier": float, "uncertainty": float, "reliability": float, "resolution": float}
        """
        n = len(forecast_probs)
        if n == 0:
            return {"brier": 0, "uncertainty": 0, "reliability": 0, "resolution": 0}

        forecasts = np.array(forecast_probs)
        outcomes = np.array(actual_outcomes, dtype=float)

        # Overall base rate
        base_rate = outcomes.mean()
        uncertainty = base_rate * (1 - base_rate)

        # Bin forecasts
        bin_edges = np.linspace(0, 1, n_bins + 1)
        reliability = 0.0
        resolution = 0.0

        for i in range(n_bins):
            mask = (forecasts >= bin_edges[i]) & (forecasts < bin_edges[i + 1])
            n_k = mask.sum()
            if n_k == 0:
                continue

            f_k = forecasts[mask].mean()
            o_k = outcomes[mask].mean()

            reliability += n_k * (f_k - o_k) ** 2
            resolution += n_k * (o_k - base_rate) ** 2

        reliability /= n
        resolution /= n

        return {
            "brier": round(uncertainty - resolution + reliability, 6),
            "uncertainty": round(uncertainty, 6),
            "reliability": round(reliability, 6),
            "resolution": round(resolution, 6),
        }

    @staticmethod
    def expected_calibration_error(
        confidences: List[float],
        accuracies: List[int],
        n_bins: int = 10,
    ) -> dict:
        """
        Expected Calibration Error (ECE).

        ECE = Σ_b (n_b / N) * |acc_b - conf_b|

        Measures weighted average gap between confidence and accuracy.

        Returns: {"ece": float, "bin_data": [...]}
        """
        n = len(confidences)
        if n == 0:
            return {"ece": 0.0, "bin_data": []}

        confs = np.array(confidences)
        accs = np.array(accuracies, dtype=float)
        bin_edges = np.linspace(0, 1, n_bins + 1)

        ece = 0.0
        bin_data = []

        for i in range(n_bins):
            mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
            n_k = mask.sum()
            if n_k == 0:
                continue

            avg_conf = confs[mask].mean()
            avg_acc = accs[mask].mean()
            gap = abs(avg_acc - avg_conf)
            ece += (n_k / n) * gap

            bin_data.append({
                "bin_lower": round(float(bin_edges[i]), 2),
                "bin_upper": round(float(bin_edges[i + 1]), 2),
                "avg_confidence": round(float(avg_conf), 4),
                "avg_accuracy": round(float(avg_acc), 4),
                "count": int(n_k),
                "gap": round(float(gap), 4),
            })

        return {"ece": round(float(ece), 6), "bin_data": bin_data}

    @staticmethod
    def overconfidence_error(
        confidences: List[float],
        accuracies: List[int],
        n_bins: int = 10,
    ) -> float:
        """
        Overconfidence Error (OE).

        Like ECE but only penalizes bins where confidence > accuracy.
        Critical in high-stakes applications where confident-but-wrong = catastrophic.
        """
        n = len(confidences)
        if n == 0:
            return 0.0

        confs = np.array(confidences)
        accs = np.array(accuracies, dtype=float)
        bin_edges = np.linspace(0, 1, n_bins + 1)

        oe = 0.0
        for i in range(n_bins):
            mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
            n_k = mask.sum()
            if n_k == 0:
                continue

            avg_conf = confs[mask].mean()
            avg_acc = accs[mask].mean()

            # Only penalize overconfidence
            if avg_conf > avg_acc:
                oe += (n_k / n) * (avg_conf - avg_acc)

        return round(float(oe), 6)

    def get_validation_report(self) -> dict:
        """
        Generate a validation report from logged predictions with recorded outcomes.
        """
        # Filter predictions that have actual outcomes
        evaluated = [p for p in self.predictions if p.get("actual_outcome")]

        if not evaluated:
            return {
                "status": "no_evaluated_predictions",
                "total_predictions": len(self.predictions),
                "evaluated_predictions": 0,
                "message": "No predictions have been evaluated against actual outcomes yet. "
                           "Use record_outcome() to log real-world results.",
            }

        # Extract confidences and whether the prediction was correct
        confidences = []
        correct = []
        for p in evaluated:
            conf = p["confidence"]
            actual = p["actual_outcome"]["label"]
            predicted = p["event_label"]
            confidences.append(conf)
            correct.append(1 if actual == predicted else 0)

        # Compute metrics
        brier = self.brier_score(confidences, correct)
        brier_decomp = self.brier_decomposition(confidences, correct)
        ece_result = self.expected_calibration_error(confidences, correct)
        oe = self.overconfidence_error(confidences, correct)

        accuracy = sum(correct) / max(len(correct), 1)

        return {
            "total_predictions": len(self.predictions),
            "evaluated_predictions": len(evaluated),
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier, 6),
            "brier_decomposition": brier_decomp,
            "expected_calibration_error": ece_result["ece"],
            "overconfidence_error": oe,
            "calibration_bins": ece_result["bin_data"],
            "assessment": _assess_calibration(brier, ece_result["ece"], oe),
        }


def _assess_calibration(brier: float, ece: float, oe: float) -> str:
    """Generate human-readable calibration assessment."""
    if brier < 0.15 and ece < 0.05:
        return "EXCELLENT — Well-calibrated probabilistic predictions."
    elif brier < 0.25 and ece < 0.10:
        return "GOOD — Reasonably calibrated with minor systematic bias."
    elif oe > 0.15:
        return "WARNING — Significant overconfidence detected. Predictions consistently exceed actual accuracy."
    elif brier < 0.35:
        return "MODERATE — Predictions show meaningful signal but calibration needs improvement."
    else:
        return "POOR — Predictions are poorly calibrated. Model retraining recommended."


# ── Singleton ────────────────────────────────────────────────────────────────
_validator: Optional[PredictionValidator] = None


def get_validator() -> PredictionValidator:
    global _validator
    if _validator is None:
        _validator = PredictionValidator()
    return _validator
