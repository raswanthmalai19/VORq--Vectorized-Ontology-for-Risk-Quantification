"""
Tests for the VORq Validation Engine (Brier Score, ECE, OE).
"""
import pytest
import numpy as np
from vorq.engine.validation import PredictionValidator


class TestBrierScore:
    def test_perfect_prediction(self):
        """Perfect forecast = Brier score of 0."""
        bs = PredictionValidator.brier_score([1.0, 0.0, 1.0], [1, 0, 1])
        assert bs == 0.0

    def test_worst_prediction(self):
        """Completely wrong forecast = Brier score of 1."""
        bs = PredictionValidator.brier_score([0.0, 1.0, 0.0], [1, 0, 1])
        assert bs == 1.0

    def test_moderate_prediction(self):
        """50% confidence on a correct event = Brier score of 0.25."""
        bs = PredictionValidator.brier_score([0.5], [1])
        assert abs(bs - 0.25) < 0.001

    def test_empty_input(self):
        bs = PredictionValidator.brier_score([], [])
        assert bs == 0.0


class TestMulticlassBrier:
    def test_perfect_multiclass(self):
        forecast = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        actual = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        bs = PredictionValidator.multiclass_brier_score(forecast, actual)
        assert bs == 0.0

    def test_worst_multiclass(self):
        forecast = np.array([[0, 1, 0], [1, 0, 0]], dtype=float)
        actual = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        bs = PredictionValidator.multiclass_brier_score(forecast, actual)
        assert bs > 0


class TestBrierDecomposition:
    def test_decomposition_structure(self):
        forecasts = [0.1, 0.4, 0.6, 0.9, 0.3, 0.7, 0.5, 0.8, 0.2, 0.6]
        outcomes = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        result = PredictionValidator.brier_decomposition(forecasts, outcomes)
        assert "brier" in result
        assert "uncertainty" in result
        assert "reliability" in result
        assert "resolution" in result

    def test_uncertainty_binary_balanced(self):
        """50/50 outcomes = max uncertainty = 0.25."""
        forecasts = [0.5] * 100
        outcomes = [0] * 50 + [1] * 50
        result = PredictionValidator.brier_decomposition(forecasts, outcomes)
        assert abs(result["uncertainty"] - 0.25) < 0.01


class TestECE:
    def test_perfectly_calibrated(self):
        """Perfectly calibrated: confidence matches accuracy."""
        confs = [0.5] * 100
        accs = [0] * 50 + [1] * 50
        result = PredictionValidator.expected_calibration_error(confs, accs)
        assert result["ece"] < 0.05  # near-zero ECE

    def test_overconfident(self):
        """High confidence but 50% accuracy = high ECE."""
        confs = [0.9] * 100
        accs = [0] * 50 + [1] * 50
        result = PredictionValidator.expected_calibration_error(confs, accs)
        assert result["ece"] > 0.3


class TestOverconfidenceError:
    def test_overconfident_penalty(self):
        confs = [0.9] * 100
        accs = [0] * 70 + [1] * 30
        oe = PredictionValidator.overconfidence_error(confs, accs)
        assert oe > 0.3  # significant overconfidence

    def test_underconfident_no_penalty(self):
        """Underconfidence should NOT be penalized by OE."""
        confs = [0.3] * 100
        accs = [0] * 20 + [1] * 80
        oe = PredictionValidator.overconfidence_error(confs, accs)
        assert oe < 0.05


class TestValidationReport:
    def test_empty_report(self):
        v = PredictionValidator()
        v.predictions = []
        report = v.get_validation_report()
        assert report["status"] == "no_evaluated_predictions"
