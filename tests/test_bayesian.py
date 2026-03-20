"""
Tests for the VORq Bayesian Scenario Engine.
"""
import pytest
from vorq.engine.bayesian_scenarios import (
    generate_scenario_tree,
    EVENT_LABELS,
    ESCALATION_LEVELS,
    CONTAGION_SCOPES,
    MARKET_REGIMES,
    SEVERITY_OUTCOMES,
)


class TestBayesianScenarioTree:
    """Test multi-branch scenario generation."""

    def test_war_scenario_tree(self):
        tree = generate_scenario_tree("war", 0.65)
        assert tree["primary_event"] == "war"
        assert len(tree["branches"]) > 0
        assert tree["event_confidence"] == 0.65

    def test_all_event_types_produce_trees(self):
        for label in EVENT_LABELS:
            tree = generate_scenario_tree(label, 0.5)
            assert tree["primary_event"] == label
            assert len(tree["branches"]) >= 1
            assert "summary_stats" in tree

    def test_branches_have_valid_fields(self):
        tree = generate_scenario_tree("pandemic", 0.7)
        for branch in tree["branches"]:
            assert branch["escalation"] in ESCALATION_LEVELS
            assert branch["contagion"] in CONTAGION_SCOPES
            assert branch["market_regime"] in MARKET_REGIMES
            assert branch["severity"] in SEVERITY_OUTCOMES
            assert 0 <= branch["probability"] <= 1
            assert 0 <= branch["severity_score"] <= 1

    def test_branch_probabilities_sum_to_one(self):
        tree = generate_scenario_tree("war", 0.8)
        total_p = sum(b["probability"] for b in tree["branches"])
        assert 0.95 <= total_p <= 1.05, f"Branch probs sum to {total_p}"

    def test_war_has_high_panic_probability(self):
        """War should have elevated panic probability."""
        tree = generate_scenario_tree("war", 0.7)
        p_panic = tree["summary_stats"]["p_panic"]
        assert p_panic > 0.15, f"War p_panic={p_panic} should be > 0.15"

    def test_trade_has_low_severity(self):
        """Trade agreements should produce low expected severity."""
        tree = generate_scenario_tree("trade_agreement", 0.8)
        severity = tree["summary_stats"]["expected_severity"]
        assert severity < 0.55, f"Trade expected_severity={severity} should be < 0.55"

    def test_pandemic_has_global_contagion(self):
        """Pandemics should have high global contagion probability."""
        tree = generate_scenario_tree("pandemic", 0.7)
        p_global = tree["summary_stats"]["p_global_contagion"]
        assert p_global > 0.40, f"Pandemic p_global={p_global} should be > 0.40"

    def test_distributions_present(self):
        tree = generate_scenario_tree("economic_crisis", 0.6)
        dists = tree["distributions"]
        assert "escalation" in dists
        assert "contagion" in dists
        assert "market_regime" in dists
        assert "severity" in dists
        assert "policy_response" in dists

    def test_invalid_event_falls_back(self):
        tree = generate_scenario_tree("invalid_event", 0.5)
        assert tree["primary_event"] == "supply_shock"  # fallback

    def test_summary_stats_values(self):
        tree = generate_scenario_tree("cyberattack", 0.55)
        stats = tree["summary_stats"]
        assert 0 <= stats["expected_severity"] <= 1
        assert 0 <= stats["p_panic"] <= 1
        assert 0 <= stats["p_global_contagion"] <= 1
        assert 0 <= stats["p_severe_or_worse"] <= 1
