"""
Tests for VORq Knowledge Graph — validates structure and causal chain logic.
"""

import json
import os
import pytest


@pytest.fixture
def knowledge_graph():
    kg_path = os.path.join(os.path.dirname(__file__), "..", "vorq", "api", "vorq_knowledge_graph.json")
    with open(kg_path, "r") as f:
        return json.load(f)


class TestKnowledgeGraphStructure:
    """Validate the knowledge graph has complete and consistent structure."""

    def test_has_required_keys(self, knowledge_graph):
        assert "sectors" in knowledge_graph
        assert "causal_links" in knowledge_graph
        assert "mitigations" in knowledge_graph

    def test_minimum_sectors(self, knowledge_graph):
        assert len(knowledge_graph["sectors"]) >= 6, \
            "Should have at least 6 sectors"

    def test_sector_has_companies(self, knowledge_graph):
        for sector, data in knowledge_graph["sectors"].items():
            assert "companies" in data, f"Sector {sector} missing companies"
            assert len(data["companies"]) >= 2, \
                f"Sector {sector} should have at least 2 companies"

    def test_minimum_causal_links(self, knowledge_graph):
        assert len(knowledge_graph["causal_links"]) >= 20, \
            "Should have at least 20 causal links for comprehensive coverage"

    def test_link_structure(self, knowledge_graph):
        for link in knowledge_graph["causal_links"]:
            assert "source" in link
            assert "target" in link
            assert "impact" in link
            assert "mechanism" in link
            assert -1.0 <= link["impact"] <= 1.0, \
                f"Impact {link['impact']} out of range for {link['source']} → {link['target']}"
            assert len(link["mechanism"]) > 10, \
                f"Mechanism too short for {link['source']} → {link['target']}"


class TestCausalChainLogic:
    """Validate that causal chains are logically sound."""

    EXPECTED_SHOCK_TYPES = [
        "war_taiwan", "war_general", "war_india_china", "war_russia_ukraine",
        "sanctions_china", "sanctions_russia",
        "oil_shock_mideast",
        "pandemic_global",
        "natural_disaster",
        "supply_shock",
        "political_crisis",
        "economic_crisis",
        "cyberattack",
        "energy_crisis_europe",
        "trade_agreement",
    ]

    def test_all_shock_types_have_links(self, knowledge_graph):
        """Every defined shock type should have at least one causal link."""
        shock_sources = {link["source"] for link in knowledge_graph["causal_links"]}
        for shock in self.EXPECTED_SHOCK_TYPES:
            assert shock in shock_sources, \
                f"Shock type '{shock}' has no causal links in the graph"

    def test_war_hurts_tech(self, knowledge_graph):
        """War in Taiwan should negatively impact Technology & Semiconductors."""
        for link in knowledge_graph["causal_links"]:
            if link["source"] == "war_taiwan" and link["target"] == "Technology & Semiconductors":
                assert link["impact"] < 0, \
                    "War in Taiwan should negatively impact tech sector"
                return
        pytest.fail("No link from war_taiwan to Technology & Semiconductors")

    def test_war_benefits_defense(self, knowledge_graph):
        """War should positively impact Defense & Aerospace."""
        for link in knowledge_graph["causal_links"]:
            if link["source"] == "war_taiwan" and link["target"] == "Defense & Aerospace":
                assert link["impact"] > 0, \
                    "War should benefit defense sector"
                return
        pytest.fail("No link from war_taiwan to Defense & Aerospace")

    def test_pandemic_benefits_healthcare(self, knowledge_graph):
        """Pandemic should positively impact healthcare."""
        for link in knowledge_graph["causal_links"]:
            if link["source"] == "pandemic_global" and link["target"] == "Healthcare & Pharma":
                assert link["impact"] > 0, \
                    "Pandemic should benefit healthcare"
                return
        pytest.fail("No link from pandemic_global to Healthcare & Pharma")

    def test_economic_crisis_hurts_finance(self, knowledge_graph):
        """Economic crisis should hurt financial services."""
        for link in knowledge_graph["causal_links"]:
            if link["source"] == "economic_crisis" and link["target"] == "Financial Services":
                assert link["impact"] < 0, \
                    "Economic crisis should hurt financial services"
                return
        pytest.fail("No link from economic_crisis to Financial Services")

    def test_trade_agreement_positive(self, knowledge_graph):
        """Trade agreements should have predominantly positive impacts."""
        trade_links = [l for l in knowledge_graph["causal_links"] if l["source"] == "trade_agreement"]
        assert len(trade_links) > 0, "Should have trade agreement links"
        positive = [l for l in trade_links if l["impact"] > 0]
        assert len(positive) == len(trade_links), \
            "All trade agreement impacts should be positive"


class TestMitigations:
    """Validate that mitigations are comprehensive."""

    def test_all_sectors_have_mitigations(self, knowledge_graph):
        for sector in knowledge_graph["sectors"]:
            assert sector in knowledge_graph["mitigations"], \
                f"Sector '{sector}' has no mitigations"
            assert len(knowledge_graph["mitigations"][sector]) >= 2, \
                f"Sector '{sector}' should have at least 2 mitigations"

    def test_mitigations_are_actionable(self, knowledge_graph):
        """Mitigations should be substantial action items, not generic filler."""
        for sector, mits in knowledge_graph["mitigations"].items():
            for m in mits:
                assert len(m) > 20, \
                    f"Mitigation for {sector} too short: '{m}'"
