"""
VORq V3 API Integration Tests.

Tests all endpoints including new V3 features:
- Bayesian scenario tree generation
- Fat-tailed Monte Carlo with VaR/CVaR
- Macro regime context
- Validation framework
"""
import pytest
from fastapi.testclient import TestClient
from vorq.api.main import app

client = TestClient(app)


# ── /health ──────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "3.0.0"
        assert "macro_regime" in data
        assert "engines" in data
        assert "bayesian_network" in data["engines"]
        assert "fat_tailed_mc" in data["engines"]

    def test_health_has_engines(self):
        r = client.get("/health")
        engines = r.json()["engines"]
        expected = ["bayesian_network", "causal_graph", "fat_tailed_mc", "fred_client", "brier_validator"]
        for e in expected:
            assert e in engines, f"Engine '{e}' missing from health check"


# ── /simulate ────────────────────────────────────────────────────────────────

class TestSimulateV3:
    """Test all 10 scenario types with V3 response schema."""

    SCENARIOS = [
        ("China invades Taiwan with military force", "war"),
        ("US imposes semiconductor sanctions on China", "sanctions"),
        ("New deadly virus pandemic spreads globally", "pandemic"),
        ("Massive earthquake destroys Tokyo", "natural_disaster"),
        ("Global semiconductor chip shortage worsens", "supply_shock"),
        ("Military coup overthrows Saudi Arabian government", "political_crisis"),
        ("Major recession and banking collapse hits US", "economic_crisis"),
        ("Massive ransomware attack hits US banking system", "cyberattack"),
        ("Oil pipeline explosion disrupts Middle East supply", "energy_crisis"),
        ("US and EU sign comprehensive free trade agreement", "trade_agreement"),
    ]

    @pytest.mark.parametrize("scenario,expected_type", SCENARIOS)
    def test_classify_correct(self, scenario, expected_type):
        r = client.post("/simulate", json={"scenario_text": scenario, "mc_iterations": 500})
        assert r.status_code == 200
        data = r.json()
        assert data["event"]["classification"]["label"] == expected_type

    def test_v3_scenario_tree_present(self):
        r = client.post("/simulate", json={"scenario_text": "China invades Taiwan", "mc_iterations": 500})
        data = r.json()
        assert "v3" in data
        v3 = data["v3"]

        # Scenario tree
        assert "scenario_tree" in v3
        tree = v3["scenario_tree"]
        assert tree["primary_event"] == "war"
        assert len(tree["branches"]) > 0
        assert "distributions" in tree
        assert "summary_stats" in tree
        assert "expected_severity" in tree["summary_stats"]
        assert "p_panic" in tree["summary_stats"]

    def test_v3_risk_analytics(self):
        r = client.post("/simulate", json={"scenario_text": "Global banking crisis", "mc_iterations": 500})
        data = r.json()
        analytics = data["v3"]["risk_analytics"]

        # Distribution stats
        dist = analytics["distribution"]
        assert "mean" in dist
        assert "std" in dist
        assert "skew" in dist
        assert "kurtosis" in dist

        # VaR/CVaR
        assert analytics["var_95"] > 0
        assert analytics["var_99"] >= analytics["var_95"]
        assert analytics["cvar_95"] >= analytics["var_95"]

        # PDF bins for UI rendering
        assert len(analytics["pdf_bins"]) > 0
        assert "x" in analytics["pdf_bins"][0]
        assert "density" in analytics["pdf_bins"][0]

        # Distribution type
        assert analytics["distribution_type"] == "student_t"

    def test_v3_macro_context(self):
        r = client.post("/simulate", json={"scenario_text": "Oil crisis", "mc_iterations": 500})
        data = r.json()
        macro = data["v3"]["macro_context"]
        assert "regime" in macro
        assert macro["regime"] in ("expansion", "slowdown", "contraction", "crisis")

    def test_v3_causal_analysis(self):
        r = client.post("/simulate", json={"scenario_text": "China invades Taiwan", "mc_iterations": 500})
        data = r.json()
        causal = data["v3"]["causal_analysis"]
        assert "impact_chains" in causal
        assert "propagation_stats" in causal
        stats = causal["propagation_stats"]
        assert stats["total_sectors_hit"] > 0

    def test_backward_compat_v2_fields(self):
        """Ensure V2 response fields still exist for backward compatibility."""
        r = client.post("/simulate", json={"scenario_text": "Pandemic outbreak", "mc_iterations": 500})
        data = r.json()

        # V2 fields must still be present
        assert "event" in data
        assert "simulation" in data
        assert "explanation" in data

        sim = data["simulation"]
        assert "overall_risk_score" in sim
        assert "risk_level" in sim
        assert "monte_carlo" in sim
        assert "company_impacts" in sim
        assert "industry_impacts" in sim
        assert sim["risk_level"] in ("CRITICAL", "HIGH", "ELEVATED", "MODERATE", "LOW")

    def test_war_severity_higher_than_trade(self):
        """War should produce higher risk than trade agreement."""
        r_war = client.post("/simulate", json={"scenario_text": "China invades Taiwan with military force", "mc_iterations": 500})
        r_trade = client.post("/simulate", json={"scenario_text": "US and EU sign free trade deal", "mc_iterations": 500})

        war_score = r_war.json()["simulation"]["overall_risk_score"]
        trade_score = r_trade.json()["simulation"]["overall_risk_score"]
        assert war_score > trade_score, f"War ({war_score}) should score higher than trade ({trade_score})"

    def test_defense_positive_in_war(self):
        """Defense sector should have positive impact during war."""
        r = client.post("/simulate", json={"scenario_text": "China invades Taiwan", "mc_iterations": 500})
        data = r.json()
        sectors = {s["sector"]: s["impact_pct"] for s in data["simulation"]["industry_impacts"]}
        if "Defense & Aerospace" in sectors:
            assert sectors["Defense & Aerospace"] > 0, "Defense should be positive during war"

    def test_healthcare_positive_in_pandemic(self):
        """Healthcare should benefit during pandemic."""
        r = client.post("/simulate", json={"scenario_text": "Global pandemic virus outbreak", "mc_iterations": 500})
        data = r.json()
        sectors = {s["sector"]: s["impact_pct"] for s in data["simulation"]["industry_impacts"]}
        if "Healthcare & Pharma" in sectors:
            assert sectors["Healthcare & Pharma"] > 0, "Healthcare should be positive during pandemic"


# ── /quick-score ─────────────────────────────────────────────────────────────

class TestQuickScore:
    def test_quick_score_has_regime(self):
        r = client.post("/quick-score", json={"scenario_text": "War in Taiwan"})
        assert r.status_code == 200
        data = r.json()
        assert "regime" in data
        assert data["regime"] in ("expansion", "slowdown", "contraction", "crisis")

    def test_quick_score_has_sectors(self):
        r = client.post("/quick-score", json={"scenario_text": "Semiconductor shortage"})
        data = r.json()
        assert len(data["top_sectors"]) > 0


# ── /macro-context ───────────────────────────────────────────────────────────

class TestMacroContext:
    def test_macro_context(self):
        r = client.get("/macro-context")
        assert r.status_code == 200
        data = r.json()
        assert "regime" in data
        assert "gdp_growth" in data
        assert "vix" in data
        assert "shock_modulator" in data
        mod = data["shock_modulator"]
        assert "shock_multiplier" in mod
        assert "recovery_speed" in mod


# ── /scenario-tree ───────────────────────────────────────────────────────────

class TestScenarioTree:
    def test_scenario_tree_war(self):
        r = client.get("/scenario-tree?event=war&confidence=0.7")
        assert r.status_code == 200
        data = r.json()
        assert data["primary_event"] == "war"
        assert len(data["branches"]) > 0
        # Probs should roughly sum to 1
        total_p = sum(b["probability"] for b in data["branches"])
        assert 0.95 <= total_p <= 1.05, f"Branch probs sum to {total_p}"

    def test_scenario_tree_trade(self):
        r = client.get("/scenario-tree?event=trade_agreement&confidence=0.8")
        data = r.json()
        # Trade agreements should have lower expected severity
        assert data["summary_stats"]["expected_severity"] < 0.6


# ── /validation/calibration ──────────────────────────────────────────────────

class TestValidation:
    def test_validation_endpoint(self):
        r = client.get("/validation/calibration")
        assert r.status_code == 200
        data = r.json()
        assert "total_predictions" in data


# ── /graph and /labels ───────────────────────────────────────────────────────

class TestLegacyEndpoints:
    def test_graph(self):
        r = client.get("/graph")
        assert r.status_code == 200
        data = r.json()
        assert "sectors" in data
        assert "causal_links" in data

    def test_labels(self):
        r = client.get("/labels")
        assert r.status_code == 200
        assert r.json()["count"] == 10
