"""
VORq API v3.0 — Institutional-Grade Risk Intelligence Engine.

Integrates:
- DistilBERT + keyword hybrid event classification
- Bayesian Network for multi-branch scenario generation
- Structural Causal Model with regime-aware 3-hop cascades
- Fat-tailed Monte Carlo with VaR/CVaR
- FRED macroeconomic data for regime detection
- Brier Score validation framework
"""

import json
import os
import logging
import hashlib
from typing import Optional

import numpy as np
from scipy import stats as sp_stats
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vorq.engine.event_extractor import (
    classify_event,
    extract_entities,
    get_country_impacts,
    get_model_status,
    map_event_to_shock_id,
)
from vorq.engine.bayesian_scenarios import generate_scenario_tree
from vorq.engine.causal_model import get_causal_graph
from vorq.data.fred_client import get_fred_client
from vorq.engine.validation import get_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vorq.api")

app = FastAPI(title="VORq Intelligence Engine", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Company sector sensitivities (realistic exposures by sector) ──────────────
# Maps sector name to a dict of {company: sensitivity_multiplier}
# Sensitivity 1.0 = normal exposure, 1.2 = high exposure, 0.8 = low exposure
COMPANY_SENSITIVITIES = {
    "Financial Services": {
        "JPMorgan Chase": 1.3,  # High exposure to macro shocks
        "Goldman Sachs": 1.2,
        "HSBC": 0.9,  # More diversified geographically
        "Deutsche Bank": 1.1,
    },
    "Technology & Semiconductors": {
        "Intel": 1.2,
        "TSMC": 1.1,  # Taiwan exposure
        "Samsung": 0.95,
        "Nvidia": 1.0,
    },
    "Automotive & EV": {
        "Volkswagen": 1.2,  # Germany, high fab dependency
        "Tesla": 1.1,
        "Toyota": 0.85,  # More diversified supply chain
        "BYD": 0.9,  # China advantage
    },
    "Defense & Aerospace": {
        "Lockheed Martin": 0.7,  # Actually benefits in war scenarios
        "Raytheon": 0.6,
        "Boeing": 1.0,
        "Northrop": 0.7,
    },
}

# ── Load Knowledge Graph ──────────────────────────────────────────────────────
GRAPH_PATH = os.path.join(os.path.dirname(__file__), "vorq_knowledge_graph.json")
try:
    with open(GRAPH_PATH, "r") as f:
        KG = json.load(f)
    logger.info("Knowledge graph loaded — %d causal links, %d sectors",
                len(KG.get("causal_links", [])), len(KG.get("sectors", {})))
except FileNotFoundError:
    KG = {"sectors": {}, "causal_links": [], "mitigations": {}}
    logger.warning("Knowledge graph not found at %s", GRAPH_PATH)


# ── Request / Response models ────────────────────────────────────────────────
class SimulationRequest(BaseModel):
    scenario_text: str = Field(..., min_length=1, max_length=2000)
    mc_iterations: int = Field(default=2000, ge=200, le=10000)


class QuickScoreRequest(BaseModel):
    scenario_text: str = Field(..., min_length=1, max_length=2000)


# ── Fat-Tailed Monte Carlo Engine ────────────────────────────────────────────
def fat_tailed_monte_carlo(
    sector_impacts: dict,
    regime: str = "slowdown",
    iterations: int = 2000,
    scenario_severity: float = 0.5,
    random_seed: Optional[int] = None,
) -> dict:
    """
    Institutional-grade Monte Carlo with:
    - Student-t distribution (df=4) for fat tails
    - GARCH-style volatility clustering
    - Regime-dependent variance scaling
    - VaR and CVaR (Expected Shortfall)
    - BOUNDED outputs: ensures impacts stay realistic [-80%, +30%]
    - Full PDF output for UI rendering
    """
    if not sector_impacts:
        return _empty_mc_result()

    # Regime-specific volatility scaler
    vol_scale = {
        "expansion": 0.8, "slowdown": 1.0, "contraction": 1.3, "crisis": 1.8,
    }.get(regime, 1.0)

    # Student-t degrees of freedom (lower = fatter tails)
    df = 4  # institutional standard for crisis modeling
    rng = np.random.default_rng(random_seed)

    all_scores = []
    sector_score_matrix = {sec: [] for sec in sector_impacts}

    for i in range(iterations):
        noisy_impacts = {}
        for sec, imp in sector_impacts.items():
            # Clamp sector impact to [-0.80, 0.30] first
            imp = max(min(imp, 0.30), -0.80)

            # Student-t noise (fat tails)
            base_vol = abs(imp) * 0.20 * vol_scale
            noise = float(rng.standard_t(df=df)) * base_vol

            # GARCH-like: amplify noise if previous iteration was extreme
            if i > 0 and len(all_scores) > 0:
                prev_score = all_scores[-1]
                if prev_score > 75:
                    noise *= 1.2  # volatility clustering
                elif prev_score < 25:
                    noise *= 0.9

            # Amplify severe negative impacts (asymmetric shock response)
            if imp < -0.5:
                noise *= 1.15  # severe shocks are more volatile

            # Calculate noisy impact and clamp it
            noisy_val = imp + noise
            noisy_impacts[sec] = max(min(noisy_val, 0.30), -0.80)
            sector_score_matrix[sec].append(noisy_impacts[sec])

        # Aggregate: weighted negative impact sum (bounded)
        total_neg = sum(abs(v) for v in noisy_impacts.values() if v < 0)
        n_sectors = max(len(noisy_impacts), 1)
        
        # Score is avg impact * 100, then scale and bound to [1, 99]
        base_score = (total_neg / n_sectors) * 100 * 1.5
        score = min(max(base_score, 1), 80)  # max 80 for extreme

        # Apply scenario severity weighting (further bound to [1, 80])
        score = score * (0.6 + 0.4 * min(scenario_severity, 1.0))
        score = int(min(max(score, 1), 99))

        all_scores.append(score)

    scores = np.array(all_scores)

    # ── Core statistics ──────────────────────────────────────────────────
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    std_score = _safe_float(np.std(scores))
    if std_score < 1e-9:
        skewness = 0.0
        kurtosis = 0.0
    else:
        skewness = _safe_float(sp_stats.skew(scores))
        kurtosis = _safe_float(sp_stats.kurtosis(scores))

    # ── VaR and CVaR ─────────────────────────────────────────────────────
    var_95 = float(np.percentile(scores, 95))
    var_99 = float(np.percentile(scores, 99))
    cvar_95 = float(np.mean(scores[scores >= var_95])) if np.any(scores >= var_95) else var_95

    # ── Probability density (for UI histogram) ───────────────────────────
    hist_counts, hist_edges = np.histogram(scores, bins=25, density=True)
    pdf_bins = []
    for j in range(len(hist_counts)):
        pdf_bins.append({
            "x": round(float((hist_edges[j] + hist_edges[j+1]) / 2), 1),
            "density": round(float(hist_counts[j]), 6),
        })

    # ── Sector-level VaR ─────────────────────────────────────────────────
    sector_risk_detail = {}
    for sec, values in sector_score_matrix.items():
        arr = np.array(values) * 100  # convert to percentage
        sector_risk_detail[sec] = {
            "mean_impact_pct": round(float(np.mean(arr)), 2),
            "std_pct": round(float(np.std(arr)), 2),
            "var_95_pct": round(float(np.percentile(arr, 5 if np.mean(arr) < 0 else 95)), 2),
        }

    return {
        "risk_distribution": {
            "mean": round(mean_score, 1),
            "median": round(median_score, 1),
            "std": round(std_score, 1),
            "skew": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
        },
        "var_95": round(var_95, 1),
        "var_99": round(var_99, 1),
        "cvar_95": round(cvar_95, 1),
        "p5": round(float(np.percentile(scores, 5)), 1),
        "p25": round(float(np.percentile(scores, 25)), 1),
        "p75": round(float(np.percentile(scores, 75)), 1),
        "p95": round(float(np.percentile(scores, 95)), 1),
        "pdf_bins": pdf_bins,
        "sector_risk_detail": sector_risk_detail,
        "iterations": iterations,
        "distribution_type": "student_t",
        "degrees_of_freedom": df,
        "regime_vol_scale": vol_scale,
    }


def _empty_mc_result() -> dict:
    return {
        "risk_distribution": {"mean": 50, "median": 50, "std": 8, "skew": 0, "kurtosis": 0},
        "var_95": 65, "var_99": 72, "cvar_95": 68,
        "p5": 35, "p25": 42, "p75": 58, "p95": 65,
        "pdf_bins": [], "sector_risk_detail": {}, "iterations": 0,
        "distribution_type": "student_t", "degrees_of_freedom": 4, "regime_vol_scale": 1.0,
    }


def _safe_float(value: float, default: float = 0.0) -> float:
    """Convert numeric values to finite float, replacing NaN/inf."""
    try:
        casted = float(value)
    except Exception:
        return default
    return casted if np.isfinite(casted) else default


def _stable_seed(*parts: str) -> int:
    """Build a stable 32-bit seed from deterministic text parts."""
    joined = "|".join(parts)
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF


# ── Risk level from score ────────────────────────────────────────────────────
def _risk_level(score: float) -> str:
    if score > 80: return "CRITICAL"
    if score > 65: return "HIGH"
    if score > 45: return "ELEVATED"
    if score > 25: return "MODERATE"
    return "LOW"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — API, model, and engine status."""
    model = get_model_status()
    model_status = "loaded" if model["loaded"] else "not_loaded"

    fred = get_fred_client()
    macro = fred.get_macro_context()

    return {
        "status": "ok",
        "version": "3.0.0",
        "model_status": model_status,
        "model_device": model["device"],
        "kg_sectors": len(KG.get("sectors", {})),
        "kg_links": len(KG.get("causal_links", [])),
        "macro_regime": macro.get("regime", "unknown"),
        "engines": ["bayesian_network", "causal_graph", "fat_tailed_mc", "fred_client", "brier_validator"],
    }


@app.post("/simulate")
async def simulate(req: SimulationRequest):
    """
    V3 Full simulation pipeline:
    1. Classify scenario → event type (ML or keyword hybrid)
    2. Generate Bayesian scenario tree (multi-branch probabilities)
    3. Detect macro regime via FRED
    4. Build dynamic causal graph → propagate shock (3-hop)
    5. Fat-tailed Monte Carlo (Student-t, VaR, CVaR)
    6. Validate & log prediction
    7. Return institutional-grade intelligence report
    """
    text = req.scenario_text.strip()
    if not text:
        return {"error": "Empty scenario text."}

    # ── Step 1: Classify event ────────────────────────────────────────────
    classification = classify_event(text)
    event_label = classification["label"]
    event_confidence = classification["confidence"]
    event_method = classification["method"]

    # ── Step 2: Bayesian scenario tree ────────────────────────────────────
    scenario_tree = generate_scenario_tree(event_label, event_confidence)

    # ── Step 3: Macro regime detection ────────────────────────────────────
    fred = get_fred_client()
    macro_context = fred.get_macro_context()
    regime = macro_context.get("regime", "slowdown")

    # ── Step 4: Build causal graph & propagate shock ──────────────────────
    entities = extract_entities(text)
    shock_id = map_event_to_shock_id(event_label, text)

    # Use scenario severity as shock intensity (0-1 scale)
    severity = scenario_tree.get("summary_stats", {}).get("expected_severity", 0.5)
    
    causal_graph = get_causal_graph(KG, regime=regime, shock_intensity=severity)
    propagation = causal_graph.propagate_shock(
        shock_id, max_depth=3, event_type=event_label
    )
    sector_impacts = propagation["sector_impacts"]

    # Fallback if no links found for this shock
    if not sector_impacts:
        propagation = causal_graph.propagate_shock(
            event_label, max_depth=3, event_type=event_label
        )
        sector_impacts = propagation["sector_impacts"]

    if not sector_impacts:
        sector_impacts = {
            "Technology & Semiconductors": -0.25,  # Reduced from -0.35
            "Financial Services": -0.15,           # Reduced from -0.25
        }
        propagation["mechanisms"] = {
            "Technology & Semiconductors": "General disruption across global tech supply chains.",
            "Financial Services": "Market uncertainty drives risk-off sentiment.",
        }

    # Stable ordering helps deterministic downstream formatting.
    sector_impacts = dict(sorted(sector_impacts.items(), key=lambda x: x[0]))
    mechanisms = propagation.get("mechanisms", {})

    # ── Step 5: Fat-tailed Monte Carlo ────────────────────────────────────
    simulation_seed = _stable_seed(text, event_label, regime, str(req.mc_iterations))
    mc_result = fat_tailed_monte_carlo(
        sector_impacts, regime=regime, iterations=req.mc_iterations,
        scenario_severity=severity,
        random_seed=simulation_seed,
    )
    risk_score = mc_result["risk_distribution"]["mean"]
    risk_lvl = _risk_level(risk_score)

    # ── Step 6: Log prediction for validation ─────────────────────────────
    validator = get_validator()
    validator.log_prediction(
        event_label=event_label,
        confidence=event_confidence,
        severity_probs=scenario_tree["distributions"].get("severity", {}),
        risk_score=risk_score,
        scenario_text=text,
    )

    # ── Step 7: Assemble response ─────────────────────────────────────────
    total_neg = sum(abs(v) for v in sector_impacts.values() if v < 0)
    worst_sector = min(sector_impacts, key=sector_impacts.get) if sector_impacts else "Unknown"

    # Country impacts
    country_impacts = get_country_impacts(event_label, entities["countries"])

    # Company impacts — use realistic sector sensitivities instead of random noise
    companies = []
    company_rng = np.random.default_rng(_stable_seed(text, event_label, "companies"))
    for sec, imp in sector_impacts.items():
        sec_data = KG.get("sectors", {}).get(sec, {})
        sec_sensitivities = COMPANY_SENSITIVITIES.get(sec, {})
        
        for comp in sec_data.get("companies", []):
            # Get sector-specific sensitivity for this company
            company_sensitivity = sec_sensitivities.get(comp, 1.0)
            
            # Calculate realistic impact: sector impact * company sensitivity * 100
            # Use small randomness for Monte Carlo variation (±5%)
            np_random = float(company_rng.uniform(-0.05, 0.05))
            exposure_decimal = imp * company_sensitivity * (1.0 + np_random)
            
            # Convert to percentage and bound to realistic range [-100%, 30%]
            exposure_pct = exposure_decimal * 100
            exposure_pct = max(min(exposure_pct, 30), -100)
            
            companies.append({
                "name": comp,
                "exposure": round(exposure_pct, 1),
                "sector": sec,
                "sensitivity": round(company_sensitivity, 2),
            })
    
    # Sort by absolute exposure and take top 8
    companies = sorted(companies, key=lambda x: abs(x["exposure"]), reverse=True)[:8]

    # Mitigations
    raw_mitigations = []
    for sec in sector_impacts.keys():
        m_list = KG.get("mitigations", {}).get(sec, [])
        raw_mitigations.extend(m_list)

    # ── Build V3 response ─────────────────────────────────────────────────
    evt = {
        "entities": {
            "countries": [{"name": c["name"]} for c in entities["countries"][:4]],
            "industries": [{"label": worst_sector}],
        },
        "severity": min(total_neg, 1.0),
        "type": event_label,
        "classification": {
            "label": event_label,
            "confidence": event_confidence,
            "method": event_method,
            "all_scores": classification.get("all_scores", {}),
        },
    }

    industry_impacts = [
        {"sector": sec, "label": sec, "impact_pct": round(imp * 100, 1)}
        for sec, imp in sector_impacts.items()
    ]

    company_impacts_list = [
        {
            "name": c["name"],
            "ticker": c["name"][:4].upper(),
            "expected_impact_pct": c["exposure"],
        }
        for c in companies
    ]

    # Timeline from scenario severity
    time_to_peak = max(30, int(severity * 120))
    recovery_months = max(3, int(severity * 24))

    sim = {
        "company_impacts": company_impacts_list,
        "industry_impacts": industry_impacts,
        "country_impacts": [
            {"id": c["id"], "name": c["name"], "impact_pct": c["impact_pct"]}
            for c in country_impacts
        ],
        "timeline": {
            "time_to_peak_days": time_to_peak,
            "recovery_months": recovery_months,
        },
        # V2 backward-compatible fields
        "risk_level": risk_lvl,
        "overall_risk_score": round(risk_score),
        "monte_carlo": {
            "iterations": req.mc_iterations,
            "p5": mc_result["p5"],
            "p95": mc_result["p95"],
            "std": mc_result["risk_distribution"]["std"],
        },
    }

    expl = {
        "sector_analysis": [
            {
                "sector": sec,
                "impact_pct": round(imp * 100, 1),
                "direction": "negative" if imp < 0 else "positive",
                "mechanism": mechanisms.get(sec, ""),
            }
            for sec, imp in sector_impacts.items()
        ],
        "mitigation_suggestions": raw_mitigations[:5],
        "confidence_assessment": {
            # Scores are normalized [0, 1] and converted to percentage in the UI.
            "overall": round(min(max(event_confidence, 0.0), 0.95), 4),
            "event_parsing": round(min(max(event_confidence, 0.0), 0.95), 4),
            "score": round(min(max(event_confidence, 0.0), 0.95), 4),
            "level": "high" if event_confidence > 0.7 else ("medium" if event_confidence > 0.5 else "low"),
            "justification": (
                f"{event_method.title()}-based classification. Event confidence: {event_confidence:.0%}. "
                f"Causal propagation across {len(sector_impacts)} sectors. "
                f"Regime: {regime}. "
                f"Scenario severity: {severity:.1%} (weighted across {len(scenario_tree.get('branches', []))} branches). "
                f"Monte Carlo iterations: {req.mc_iterations}. "
                f"This reflects the confidence in EVENT CLASSIFICATION, not in the scenario severity itself."
            ),
        },
    }

    # ── V3 additions ──────────────────────────────────────────────────────
    v3 = {
        "scenario_tree": scenario_tree,
        "risk_analytics": {
            "distribution": mc_result["risk_distribution"],
            "var_95": mc_result["var_95"],
            "var_99": mc_result["var_99"],
            "cvar_95": mc_result["cvar_95"],
            "pdf_bins": mc_result["pdf_bins"],
            "sector_risk_detail": mc_result["sector_risk_detail"],
            "distribution_type": mc_result["distribution_type"],
        },
        "macro_context": {
            "regime": regime,
            "regime_confidence": macro_context.get("regime_confidence", 0),
            "gdp_growth": macro_context.get("gdp_growth"),
            "vix": macro_context.get("vix"),
            "yield_spread": macro_context.get("yield_spread"),
        },
        "causal_analysis": {
            "impact_chains": propagation.get("impact_chains", [])[:10],
            "propagation_stats": propagation.get("propagation_stats", {}),
            # 🟢 Add calibration transparency
            "calibration": {
                "impact_bounds": "[-80%, +30%]",
                "shock_intensity": round(severity, 4),
                "event_type": event_label,
                "historical_calibration_active": propagation.get("propagation_stats", {}).get("calibration_active", False),
                "methodology": "Structural causal model with regime-dependent weights, 3-hop propagation, historical event calibration, and fat-tailed Monte Carlo sampling.",
            },
        },
    }

    return {"event": evt, "simulation": sim, "explanation": expl, "v3": v3}


@app.post("/quick-score")
async def quick_score(req: QuickScoreRequest):
    """Quick risk classification without full simulation."""
    text = req.scenario_text.strip()
    if not text:
        return {"error": "Empty scenario text."}

    classification = classify_event(text)
    event_label = classification["label"]

    shock_id = map_event_to_shock_id(event_label, text)

    fred = get_fred_client()
    macro = fred.get_macro_context()
    regime = macro.get("regime", "slowdown")

    # Use default moderate shock intensity (0.5) for quick score
    causal = get_causal_graph(KG, regime=regime, shock_intensity=0.5)
    prop = causal.propagate_shock(shock_id, max_depth=2, event_type=event_label)
    sector_impacts = prop["sector_impacts"]

    if not sector_impacts:
        prop = causal.propagate_shock(event_label, max_depth=2, event_type=event_label)
        sector_impacts = prop["sector_impacts"]

    total_neg = sum(abs(v) for v in sector_impacts.values() if v < 0)
    risk_score = min(max(int((total_neg / max(len(sector_impacts), 1)) * 100 * 1.5), 10), 99)

    return {
        "event_type": event_label,
        "confidence": min(classification["confidence"], 0.99),  # Cap at 99%
        "method": classification["method"],
        "risk_score": risk_score,
        "regime": regime,
        "top_sectors": [
            {"sector": sec, "impact_pct": round(imp * 100, 1)}
            for sec, imp in sorted(sector_impacts.items(), key=lambda x: x[1])[:3]
        ],
    }


@app.get("/macro-context")
async def macro_context():
    """Return current macroeconomic regime and key indicators."""
    fred = get_fred_client()
    ctx = fred.get_macro_context()
    mod = fred.get_shock_modulator(ctx.get("regime"))
    return {**ctx, "shock_modulator": mod}


@app.get("/scenario-tree")
async def scenario_tree_endpoint(event: str = "war", confidence: float = 0.6):
    """Return Bayesian scenario tree for an event type."""
    tree = generate_scenario_tree(event, confidence)
    return tree


@app.get("/validation/calibration")
async def validation_calibration():
    """Return Brier Score and calibration metrics."""
    validator = get_validator()
    return validator.get_validation_report()


@app.get("/graph")
async def get_graph():
    """Return the knowledge graph structure."""
    return KG


@app.get("/labels")
async def get_labels():
    """Return supported event classification labels."""
    return {
        "labels": [
            "war", "sanctions", "pandemic", "natural_disaster", "supply_shock",
            "political_crisis", "economic_crisis", "cyberattack", "energy_crisis",
            "trade_agreement",
        ],
        "count": 10,
    }
