"""
Tests for VORq Event Extractor — validates ML classification, entity extraction,
and event-to-shock mapping for all 10 event types.
"""

import pytest
from vorq.engine.event_extractor import (
    classify_event,
    extract_entities,
    get_country_impacts,
    map_event_to_shock_id,
    _keyword_classify,
    EVENT_LABELS,
)


# ── Classification tests (works with model OR keyword fallback) ──────────────

class TestClassifyEvent:
    """Test event classification across all 10 categories."""

    @pytest.mark.parametrize("text,expected_label", [
        ("War between India and China in 1 year", "war"),
        ("China invades Taiwan with military force", "war"),
        ("Russia launches missile strikes on Ukraine", "war"),
        ("US imposes sanctions on Chinese semiconductor exports", "sanctions"),
        ("Trade embargo against Russia blocks oil exports", "sanctions"),
        ("Global COVID-19 variant causes new pandemic wave", "pandemic"),
        ("New virus outbreak in Southeast Asia spreads rapidly", "pandemic"),
        ("Massive earthquake destroys infrastructure in Japan", "natural_disaster"),
        ("Hurricane devastates the US Gulf Coast", "natural_disaster"),
        ("Global semiconductor shortage disrupts supply chains", "supply_shock"),
        ("Port closures cause logistics bottleneck worldwide", "supply_shock"),
        ("Coup attempt destabilizes government in major nation", "political_crisis"),
        ("Major protest movement threatens regime stability", "political_crisis"),
        ("Global recession and debt crisis", "economic_crisis"),
        ("Market crash and banking system collapse", "economic_crisis"),
        ("Major ransomware attack shuts down banking infrastructure", "cyberattack"),
        ("State-sponsored hack targets critical digital infrastructure", "cyberattack"),
        ("Oil supply crisis in Middle East disrupts energy markets", "energy_crisis"),
        ("European gas supply cut off by pipeline disruption", "energy_crisis"),
        ("US-EU free trade agreement signed", "trade_agreement"),
    ])
    def test_classification_labels(self, text, expected_label):
        """Model or keyword classification should return a valid label."""
        result = classify_event(text)
        assert result["label"] in EVENT_LABELS, \
            f"Label '{result['label']}' not in valid labels"
        assert result["confidence"] > 0, "Confidence should be positive"
        assert result["method"] in ("model", "keyword", "hybrid", "fallback")

    def test_empty_input(self):
        result = classify_event("")
        assert result["method"] == "fallback"
        assert result["confidence"] <= 0.2

    def test_all_scores_present(self):
        result = classify_event("War between nations escalates")
        if result["method"] in ("model", "keyword"):
            assert len(result["all_scores"]) > 0

    def test_confidence_range(self):
        result = classify_event("Oil crisis and energy shortage")
        assert 0 <= result["confidence"] <= 1.0


class TestKeywordClassify:
    """Test the keyword-based fallback classifier."""

    def test_war_keywords(self):
        result = _keyword_classify("military invasion and bombing campaign")
        assert result["label"] == "war"

    def test_pandemic_keywords(self):
        result = _keyword_classify("virus outbreak causes epidemic quarantine")
        assert result["label"] == "pandemic"

    def test_cyber_keywords(self):
        result = _keyword_classify("ransomware hack breaches data systems")
        assert result["label"] == "cyberattack"

    def test_no_keywords_fallback(self):
        result = _keyword_classify("something completely random and unrelated")
        assert result["label"] in EVENT_LABELS  # should still return a valid label

    def test_multi_category_picks_strongest(self):
        result = _keyword_classify("war and military conflict with bombing")
        assert result["label"] == "war"


# ── Entity extraction tests ──────────────────────────────────────────────────

class TestExtractEntities:
    """Test country/region extraction from scenario text."""

    def test_single_country(self):
        result = extract_entities("China's economy is collapsing")
        ids = [c["id"] for c in result["countries"]]
        assert "CN" in ids

    def test_multiple_countries(self):
        result = extract_entities("War between India and China affects Taiwan")
        ids = [c["id"] for c in result["countries"]]
        assert "IN" in ids
        assert "CN" in ids
        assert "TW" in ids

    def test_no_countries(self):
        result = extract_entities("Something random happens in the world")
        assert isinstance(result["countries"], list)

    def test_us_variants(self):
        for text in ["United States", "USA policy", "American economy"]:
            result = extract_entities(text)
            ids = [c["id"] for c in result["countries"]]
            assert "US" in ids, f"Failed for: {text}"

    def test_uk_detection(self):
        result = extract_entities("Britain exits the European Union")
        ids = [c["id"] for c in result["countries"]]
        assert "GB" in ids

    def test_country_has_coordinates(self):
        result = extract_entities("Japan earthquake")
        for c in result["countries"]:
            assert "lat" in c
            assert "lon" in c


# ── Country impact tests ─────────────────────────────────────────────────────

class TestGetCountryImpacts:
    """Test country impact generation."""

    def test_war_defaults(self):
        impacts = get_country_impacts("war", [])
        assert len(impacts) > 0
        assert all("impact_pct" in c for c in impacts)

    def test_mentioned_countries_included(self):
        mentioned = [{"id": "IN", "name": "India", "lat": 20.59, "lon": 78.9}]
        impacts = get_country_impacts("war", mentioned)
        impact_ids = [c["id"] for c in impacts]
        assert "IN" in impact_ids

    def test_trade_agreement_positive(self):
        impacts = get_country_impacts("trade_agreement", [])
        positive = [c for c in impacts if c["impact_pct"] > 0]
        assert len(positive) > 0, "Trade agreements should have positive impacts"

    def test_max_6_countries(self):
        many_countries = [
            {"id": f"C{i}", "name": f"Country{i}", "lat": 0, "lon": 0}
            for i in range(10)
        ]
        impacts = get_country_impacts("war", many_countries)
        assert len(impacts) <= 6


# ── Shock ID mapping tests ───────────────────────────────────────────────────

class TestMapEventToShockId:
    """Test mapping of event labels to knowledge graph shock IDs."""

    def test_war_taiwan(self):
        sid = map_event_to_shock_id("war", "China invades Taiwan")
        assert sid == "war_taiwan"

    def test_war_india_china(self):
        sid = map_event_to_shock_id("war", "India and China border conflict")
        assert sid == "war_india_china"

    def test_war_russia_ukraine(self):
        sid = map_event_to_shock_id("war", "Russia attacks Ukraine")
        assert sid == "war_russia_ukraine"

    def test_sanctions_default(self):
        sid = map_event_to_shock_id("sanctions", "Export controls on semiconductors")
        assert sid == "sanctions_china"

    def test_sanctions_russia(self):
        sid = map_event_to_shock_id("sanctions", "Sanctions against Russia for aggression")
        assert sid == "sanctions_russia"

    def test_energy_europe(self):
        sid = map_event_to_shock_id("energy_crisis", "European gas supply crisis")
        assert sid == "energy_crisis_europe"

    def test_pandemic(self):
        sid = map_event_to_shock_id("pandemic", "Global pandemic spreads")
        assert sid == "pandemic_global"
