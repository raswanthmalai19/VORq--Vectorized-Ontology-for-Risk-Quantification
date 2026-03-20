"""
VORq Structural Causal Model — Dynamic, regime-aware graph traversal.

Replaces the static 2-hop JSON graph with a dynamic causal engine that:
- Adjusts impact weights based on economic regime (crisis vs calm)
- Propagates shocks through 3 hops (3rd-order cascades)
- Models non-linear amplification during stress regimes
"""

import logging
from typing import Optional

import numpy as np
import networkx as nx

logger = logging.getLogger("vorq.causal_model")


# ── Regime multipliers ───────────────────────────────────────────────────────
# During a crisis regime, negative shocks amplify; positive shocks dampen.
REGIME_MULTIPLIERS = {
    "expansion":   {"negative": 0.7,  "positive": 1.2},  # Good times buffer negatives
    "slowdown":    {"negative": 1.0,  "positive": 1.0},  # Neutral
    "contraction": {"negative": 1.3,  "positive": 0.8},  # Bad times amplify negatives
    "crisis":      {"negative": 1.6,  "positive": 0.5},  # Extreme amplification
}

# ── Historical event calibration (realistic bounds based on actual crises) ────
# These are the worst-case impacts observed in comparable historical events.
# Source: 2008 crisis, COVID, 9/11, JFK assassination, etc.
HISTORICAL_CALIBRATION = {
    "war": {
        "base_impact": -0.25,  # Historical average: -15% to -35%
        "max_impact": -0.60,   # Worst case (e.g., major war): -60%
        "reference_events": ["Iraq War", "Falklands War"],
    },
    "sanctions": {
        "base_impact": -0.15,
        "max_impact": -0.45,
        "reference_events": ["Russia sanctions", "Iran sanctions"],
    },
    "pandemic": {
        "base_impact": -0.30,
        "max_impact": -0.70,
        "reference_events": ["COVID-19", "Spanish Flu"],
    },
    "natural_disaster": {
        "base_impact": -0.10,
        "max_impact": -0.35,
        "reference_events": ["2011 Japan earthquake", "Hurricane Katrina"],
    },
    "supply_shock": {
        "base_impact": -0.12,
        "max_impact": -0.40,
        "reference_events": ["1973 Oil Crisis", "2021 chip shortage"],
    },
    "political_crisis": {
        "base_impact": -0.08,
        "max_impact": -0.30,
        "reference_events": ["JFK assassination", "Brexit vote"],
    },
    "economic_crisis": {
        "base_impact": -0.35,
        "max_impact": -0.80,  # 2008 was ~-50% peak
        "reference_events": ["2008 Financial Crisis", "1987 Black Monday"],
    },
    "cyberattack": {
        "base_impact": -0.05,
        "max_impact": -0.25,
        "reference_events": ["WannaCry", "Colonial Pipeline"],
    },
    "energy_crisis": {
        "base_impact": -0.18,
        "max_impact": -0.50,
        "reference_events": ["1979 oil crisis", "2022 energy crisis"],
    },
    "trade_agreement": {
        "base_impact": -0.02,
        "max_impact": -0.15,  # Positive or small negative
        "reference_events": ["NAFTA", "China trade deal"],
    },
}

# ── Sector cross-coupling matrix (additional 2nd/3rd order effects) ──────────
# These represent structural dependencies between sectors that exist
# regardless of the specific shock type.
SECTOR_COUPLINGS = {
    ("Technology & Semiconductors", "Automotive & EV"): 0.65,
    ("Technology & Semiconductors", "Defense & Aerospace"): 0.40,
    ("Energy & Oil", "Logistics & Shipping"): 0.55,
    ("Energy & Oil", "Agriculture & Food"): 0.45,
    ("Energy & Oil", "Automotive & EV"): 0.35,
    ("Financial Services", "Technology & Semiconductors"): 0.30,
    ("Financial Services", "Automotive & EV"): 0.25,
    ("Logistics & Shipping", "Agriculture & Food"): 0.40,
    ("Logistics & Shipping", "Healthcare & Pharma"): 0.25,
    ("Agriculture & Food", "Healthcare & Pharma"): 0.15,
}


class CausalGraph:
    """
    Dynamic causal graph with regime-dependent weights and 3rd-order cascades.
    """

    def __init__(self, knowledge_graph: dict, regime: str = "slowdown", shock_intensity: float = 0.5):
        """
        Args:
            knowledge_graph: The VORq JSON knowledge graph.
            regime: Current economic regime — one of expansion/slowdown/contraction/crisis.
            shock_intensity: Shock severity multiplier (0.0 to 1.0).
                - 0.3 = mild shock (e.g., minor trade dispute)
                - 0.5 = moderate shock (typical scenario)
                - 0.8 = severe shock (e.g., major escalation)
                - 1.0 = extreme shock (worst-case scenario)
        """
        self.kg = knowledge_graph
        self.regime = regime
        self.shock_intensity = max(min(shock_intensity, 1.0), 0.0)  # clamp to [0, 1]
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """Build NetworkX directed graph from knowledge graph JSON."""
        # Add sector nodes
        for sector, data in self.kg.get("sectors", {}).items():
            self.graph.add_node(sector, node_type="sector", **data)

        # Add shock nodes and causal edges
        for link in self.kg.get("causal_links", []):
            source = link["source"]
            target = link["target"]
            impact = link["impact"]
            mechanism = link.get("mechanism", "")

            # Add source node if it's a shock node
            if source not in self.graph:
                self.graph.add_node(source, node_type="shock")

            # Apply regime multiplier to impact
            adjusted_impact = self._regime_adjust(impact)

            self.graph.add_edge(
                source, target,
                weight=adjusted_impact,
                raw_weight=impact,
                mechanism=mechanism,
            )

        # Add sector cross-coupling edges (if not already present)
        for (s1, s2), coupling in SECTOR_COUPLINGS.items():
            if s1 in self.graph and s2 in self.graph:
                if not self.graph.has_edge(s1, s2):
                    self.graph.add_edge(
                        s1, s2,
                        weight=coupling,
                        raw_weight=coupling,
                        mechanism=f"Structural dependency: {s1} → {s2}",
                    )

        logger.info(
            "Causal graph built: %d nodes, %d edges, regime=%s",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.regime,
        )

    def _regime_adjust(self, impact: float) -> float:
        """Adjust impact based on current economic regime."""
        mult = REGIME_MULTIPLIERS.get(self.regime, REGIME_MULTIPLIERS["slowdown"])
        direction = "negative" if impact < 0 else "positive"
        return impact * mult[direction]

    def set_regime(self, regime: str):
        """Update the regime and rebuild graph weights."""
        self.regime = regime
        # Re-adjust all edge weights
        for u, v, data in self.graph.edges(data=True):
            raw = data.get("raw_weight", data["weight"])
            data["weight"] = self._regime_adjust(raw)

    def propagate_shock(
        self,
        shock_id: str,
        max_depth: int = 3,
        decay_factor: float = 0.6,
        min_impact: float = 0.02,
        event_type: str = None,
    ) -> dict:
        """
        Propagate a shock through the causal graph with N-hop cascades.

        Args:
            shock_id: The shock node to start from.
            max_depth: Maximum cascade depth (1=direct, 2=2nd-order, 3=3rd-order).
            decay_factor: Each hop multiplies the impact by this factor.
            min_impact: Minimum absolute impact to continue propagation.
            event_type: Event label for historical calibration (e.g., "war", "pandemic").

        Returns:
            {
                "sector_impacts": {"Technology & Semiconductors": -0.35, ...},
                "mechanisms": {"Technology & Semiconductors": "...", ...},
                "impact_chains": [...],
                "propagation_stats": {..., "shock_intensity": 0.5, "calibration": "active"},
            }
        """
        if shock_id not in self.graph:
            logger.warning("Shock '%s' not in graph — trying fallback", shock_id)
            return self._empty_propagation()

        # Get historical calibration limits for this event type
        cal_limits = HISTORICAL_CALIBRATION.get(event_type, {})
        max_allowed_impact = cal_limits.get("max_impact", -0.60)  # default to -60%

        sector_impacts = {}
        mechanisms = {}
        impact_chains = []

        # BFS-like propagation with depth tracking
        frontier = [(shock_id, 1.0, 0, [shock_id])]  # (node, cum_impact, depth, path)
        visited_at_depth = {}  # track best impact per node

        while frontier:
            node, cum_impact, depth, path = frontier.pop(0)

            if depth >= max_depth:
                continue

            # Get successors
            for succ in self.graph.successors(node):
                edge_data = self.graph[node][succ]
                edge_weight = edge_data["weight"]

                # Impact cascades with decay
                if depth == 0:
                    # Direct shock impact — no decay
                    hop_impact = edge_weight
                else:
                    # Cascading impact — decay + non-linear amplification
                    hop_impact = cum_impact * edge_weight * decay_factor

                # Apply non-linear amplification for severe impacts
                if abs(hop_impact) > 0.5:
                    hop_impact *= 1.15  # amplify extreme impacts

                # 🟢 Apply shock intensity variable
                hop_impact = hop_impact * self.shock_intensity

                # 🟢 HARD LIMIT: Clamp to realistic bounds [-0.80, 0.30]
                hop_impact = max(min(hop_impact, 0.30), -0.80)

                # 🟢 Historical calibration: enforce event-specific maximum
                if event_type and cal_limits:
                    hop_impact = max(hop_impact, max_allowed_impact)

                # 🔴 HARD LIMIT: Clamp all impacts to realistic bounds [-0.80, 0.30]
                hop_impact = max(min(hop_impact, 0.30), -0.80)

                if abs(hop_impact) < min_impact:
                    continue

                new_path = path + [succ]
                depth_key = (succ, depth + 1)

                # Only update if this path gives a stronger impact
                if depth_key in visited_at_depth:
                    if abs(hop_impact) <= abs(visited_at_depth[depth_key]):
                        continue

                visited_at_depth[depth_key] = hop_impact

                # Only sector nodes get recorded as impacts
                node_data = self.graph.nodes.get(succ, {})
                if node_data.get("node_type") == "sector" or succ in self.kg.get("sectors", {}):
                    # Accumulate or set impact (take strongest path)
                    if succ not in sector_impacts or abs(hop_impact) > abs(sector_impacts[succ]):
                        sector_impacts[succ] = round(hop_impact, 4)
                        mechanisms[succ] = edge_data.get("mechanism", "")

                    impact_chains.append({
                        "path": new_path,
                        "impact": round(hop_impact, 4),
                        "depth": depth + 1,
                    })

                # Continue propagation from this node
                frontier.append((succ, hop_impact, depth + 1, new_path))

        # Sort impact chains by absolute impact
        impact_chains = sorted(impact_chains, key=lambda x: -abs(x["impact"]))[:15]

        max_depth_reached = max((c["depth"] for c in impact_chains), default=0)

        return {
            "sector_impacts": sector_impacts,
            "mechanisms": mechanisms,
            "impact_chains": impact_chains,
            "propagation_stats": {
                "total_sectors_hit": len(sector_impacts),
                "max_depth_reached": max_depth_reached,
                "total_impact_magnitude": round(
                    sum(abs(v) for v in sector_impacts.values()), 4
                ),
                "regime": self.regime,
                "regime_multiplier": REGIME_MULTIPLIERS.get(
                    self.regime, REGIME_MULTIPLIERS["slowdown"]
                ),
                "shock_intensity": round(self.shock_intensity, 4),
                "event_type": event_type,
                "calibration_active": bool(event_type and cal_limits),
                "calibration_max_impact": round(max_allowed_impact, 4) if event_type else None,
            },
        }

    def _empty_propagation(self) -> dict:
        return {
            "sector_impacts": {},
            "mechanisms": {},
            "impact_chains": [],
            "propagation_stats": {
                "total_sectors_hit": 0,
                "max_depth_reached": 0,
                "total_impact_magnitude": 0,
                "regime": self.regime,
            },
        }

    def get_graph_stats(self) -> dict:
        """Return graph metadata for diagnostics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "shock_nodes": sum(
                1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "shock"
            ),
            "sector_nodes": sum(
                1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "sector"
            ),
            "regime": self.regime,
            "avg_out_degree": round(
                sum(d for _, d in self.graph.out_degree()) / max(self.graph.number_of_nodes(), 1),
                2,
            ),
        }


# ── Singleton holder ─────────────────────────────────────────────────────────
_causal_graph: Optional[CausalGraph] = None


def get_causal_graph(knowledge_graph: dict, regime: str = "slowdown", shock_intensity: float = 0.5) -> CausalGraph:
    """
    Get or create the causal graph with specified regime and shock intensity.
    
    Note: When shock_intensity varies, a new instance is created to avoid
    singleton interference. For repeated calls with same params, consider caching.
    """
    global _causal_graph
    if _causal_graph is None:
        _causal_graph = CausalGraph(knowledge_graph, regime, shock_intensity)
    elif _causal_graph.regime != regime or _causal_graph.shock_intensity != shock_intensity:
        # Create new instance to avoid singleton issues with varying parameters
        return CausalGraph(knowledge_graph, regime, shock_intensity)
    return _causal_graph
