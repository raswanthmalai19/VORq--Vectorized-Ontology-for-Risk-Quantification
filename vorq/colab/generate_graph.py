import json

# 1. Define the Graph Nodes (Sectors & Companies)
knowledge_graph = {
    "metadata": {
        "version": "1.0.0",
        "description": "VORq Probabilistic Causal Knowledge Graph",
        "nodes": 14,
        "edges": 32
    },
    "sectors": {
        "Technology & Semiconductors": {
            "base_volatility": 0.8,
            "companies": ["Taiwan Semiconductor (TSMC)", "NVIDIA", "Apple", "ASML"],
            "dependencies": {
                "energy": 0.4,
                "rare_earths": 0.9
            }
        },
        "Defense & Aerospace": {
            "base_volatility": 0.5,
            "companies": ["Lockheed Martin", "Raytheon", "Airbus", "Boeing"],
            "dependencies": {
                "semiconductors": 0.7,
                "metals": 0.6
            }
        },
        "Automotive & EV": {
            "base_volatility": 0.6,
            "companies": ["Tesla", "Toyota", "Volkswagen", "BYD"],
            "dependencies": {
                "semiconductors": 0.8,
                "batteries": 0.9,
                "metals": 0.5
            }
        },
        "Energy & Oil": {
            "base_volatility": 0.7,
            "companies": ["ExxonMobil", "Saudi Aramco", "Shell", "Chevron"],
            "dependencies": {
                "geopolitics": 0.9
            }
        },
        "Financial Services": {
            "base_volatility": 0.4,
            "companies": ["JPMorgan Chase", "Goldman Sachs", "HSBC"],
            "dependencies": {
                "global_economy": 0.9
            }
        },
        "Agriculture & Food": {
            "base_volatility": 0.3,
            "companies": ["Archer Daniels Midland", "Bunge", "Cargill"],
            "dependencies": {
                "energy": 0.6,
                "fertilizer": 0.8
            }
        }
    },
    "causal_links": [
        {"source": "war_taiwan", "target": "Technology & Semiconductors", "impact": -0.85, "mechanism": "Primary fabs & R&D are concentrated in conflict zones — export controls freeze chip supply and IP licensing instantly."},
        {"source": "war_taiwan", "target": "Defense & Aerospace", "impact": 0.45, "mechanism": "Governments surge procurement; supply chains reorient as escalation unfolds."},
        {"source": "Technology & Semiconductors", "target": "Automotive & EV", "impact": 0.70, "mechanism": "EV battery materials and micro-chip shortages cascade through production lines."},
        {"source": "sanctions_china", "target": "Technology & Semiconductors", "impact": -0.60, "mechanism": "Trade bans restrict rare earth exports and block advanced chip designs from entering mainland markets."},
        {"source": "sanctions_china", "target": "Automotive & EV", "impact": -0.40, "mechanism": "Battery supply chains heavily reliant on Chinese processing face immediate margin pressure."},
        {"source": "oil_shock_mideast", "target": "Energy & Oil", "impact": 0.80, "mechanism": "Shipping lanes and sanctions combine to disrupt global crude flows and refinery margins, spiking prices."},
        {"source": "oil_shock_mideast", "target": "Agriculture & Food", "impact": -0.50, "mechanism": "Spiking energy costs immediately translate to higher fertilizer prices and transport costs."},
        {"source": "pandemic_global", "target": "Financial Services", "impact": -0.65, "mechanism": "Capital flight, currency volatility, sovereign downgrades hit bank exposure hard."}
    ],
    "mitigations": {
        "Technology & Semiconductors": [
            "Accelerate nearshoring of critical fab infrastructure (CHIPS Act funding).",
            "Stockpile legacy node chips for 18-month buffering."
        ],
        "Automotive & EV": [
            "Redesign silicon architecture to rely on widely available analog chips.",
            "Diversify battery mineral refining away from single-point bottlenecks."
        ],
        "Defense & Aerospace": [
            "Lock in long-term critical mineral contracts with allied nations.",
            "Expedite government defense procurement approvals."
        ],
        "Energy & Oil": [
            "Hedge spot market exposure with 24-month futures.",
            "Reroute strategic petroleum reserves to critical infrastructure."
        ]
    }
}

# 2. Export the Artifact
output_file = "/Users/raswanthmalaisamy/Desktop/VORQ/vorq/api/vorq_knowledge_graph.json"
with open(output_file, "w") as f:
    json.dump(knowledge_graph, f, indent=2)

print(f"Successfully generated {output_file} with {len(knowledge_graph['sectors'])} sectors and {len(knowledge_graph['causal_links'])} causal links.")
print("This artifact can now be loaded by the local FastAPI engine for sub-millisecond deterministic simulation.")
