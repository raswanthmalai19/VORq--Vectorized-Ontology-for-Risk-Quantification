"""
Tests for the VORq Structural Causal Model.
"""
import json
import os
import pytest
from vorq.engine.causal_model import CausalGraph, REGIME_MULTIPLIERS


@pytest.fixture
def knowledge_graph():
    """Load the knowledge graph."""
    kg_path = os.path.join(os.path.dirname(__file__), "..", "vorq", "api", "vorq_knowledge_graph.json")
    with open(kg_path, "r") as f:
        return json.load(f)


class TestCausalGraph:
    def test_graph_builds(self, knowledge_graph):
        cg = CausalGraph(knowledge_graph, regime="slowdown")
        stats = cg.get_graph_stats()
        assert stats["nodes"] > 10
        assert stats["edges"] > 40
        assert stats["regime"] == "slowdown"

    def test_war_propagation_hits_tech(self, knowledge_graph):
        cg = CausalGraph(knowledge_graph, regime="slowdown")
        result = cg.propagate_shock("war_taiwan", max_depth=3)
        impacts = result["sector_impacts"]
        assert "Technology & Semiconductors" in impacts
        assert impacts["Technology & Semiconductors"] < 0

    def test_war_propagation_defense_positive(self, knowledge_graph):
        cg = CausalGraph(knowledge_graph, regime="slowdown")
        result = cg.propagate_shock("war_taiwan", max_depth=3)
        impacts = result["sector_impacts"]
        if "Defense & Aerospace" in impacts:
            assert impacts["Defense & Aerospace"] > 0

    def test_3rd_order_cascade(self, knowledge_graph):
        """3-hop propagation should hit more sectors than 1-hop."""
        cg = CausalGraph(knowledge_graph, regime="slowdown")
        r1 = cg.propagate_shock("war_taiwan", max_depth=1)
        r3 = cg.propagate_shock("war_taiwan", max_depth=3)
        assert len(r3["sector_impacts"]) >= len(r1["sector_impacts"])

    def test_crisis_regime_amplifies(self, knowledge_graph):
        """Crisis regime should amplify negative impacts vs expansion."""
        cg_crisis = CausalGraph(knowledge_graph, regime="crisis")
        cg_expansion = CausalGraph(knowledge_graph, regime="expansion")

        r_crisis = cg_crisis.propagate_shock("war_taiwan", max_depth=2)
        r_exp = cg_expansion.propagate_shock("war_taiwan", max_depth=2)

        # Crisis should have stronger negative impacts
        crisis_neg = sum(abs(v) for v in r_crisis["sector_impacts"].values() if v < 0)
        exp_neg = sum(abs(v) for v in r_exp["sector_impacts"].values() if v < 0)
        assert crisis_neg > exp_neg, "Crisis regime should amplify negative impacts"

    def test_regime_change(self, knowledge_graph):
        cg = CausalGraph(knowledge_graph, regime="slowdown")
        assert cg.regime == "slowdown"
        cg.set_regime("crisis")
        assert cg.regime == "crisis"

    def test_impact_chains_have_paths(self, knowledge_graph):
        cg = CausalGraph(knowledge_graph, regime="slowdown")
        result = cg.propagate_shock("war_taiwan", max_depth=3)
        chains = result["impact_chains"]
        assert len(chains) > 0
        for chain in chains:
            assert len(chain["path"]) >= 2
            assert "impact" in chain
            assert "depth" in chain

    def test_propagation_stats(self, knowledge_graph):
        cg = CausalGraph(knowledge_graph, regime="contraction")
        result = cg.propagate_shock("pandemic_global", max_depth=3)
        stats = result["propagation_stats"]
        assert stats["total_sectors_hit"] > 0
        assert stats["total_impact_magnitude"] > 0
        assert stats["regime"] == "contraction"

    def test_unknown_shock_returns_empty(self, knowledge_graph):
        cg = CausalGraph(knowledge_graph, regime="slowdown")
        result = cg.propagate_shock("totally_unknown_shock")
        assert result["sector_impacts"] == {}


class TestRegimeMultipliers:
    def test_crisis_amplifies_negative(self):
        mult = REGIME_MULTIPLIERS["crisis"]
        assert mult["negative"] > 1.0

    def test_expansion_buffers_negative(self):
        mult = REGIME_MULTIPLIERS["expansion"]
        assert mult["negative"] < 1.0

    def test_all_regimes_defined(self):
        for regime in ["expansion", "slowdown", "contraction", "crisis"]:
            assert regime in REGIME_MULTIPLIERS
            assert "negative" in REGIME_MULTIPLIERS[regime]
            assert "positive" in REGIME_MULTIPLIERS[regime]
