"""
VORq Bayesian Scenario Engine — Multi-branch probabilistic scenario generation.

Implements a compact discrete Bayesian Network with native NumPy inference.
If pgmpy is installed, we also validate the network structure at startup, but
runtime inference does not depend on pgmpy so production remains robust.
"""

import logging
from typing import Optional

import numpy as np

try:  # Optional dependency: keep runtime resilient when pgmpy is unavailable.
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
except Exception:  # pragma: no cover - exercised only when pgmpy is absent
    DiscreteBayesianNetwork = None
    TabularCPD = None

logger = logging.getLogger("vorq.bayesian")

# ── Event labels (same 10 classes as the classifier) ─────────────────────────
EVENT_LABELS = [
    "war", "sanctions", "pandemic", "natural_disaster", "supply_shock",
    "political_crisis", "economic_crisis", "cyberattack", "energy_crisis",
    "trade_agreement",
]

# ── Node value definitions ───────────────────────────────────────────────────
ESCALATION_LEVELS = ["low", "medium", "high"]
CONTAGION_SCOPES = ["local", "regional", "global"]
POLICY_RESPONSES = ["none", "moderate", "aggressive"]
MARKET_REGIMES = ["risk_on", "risk_off", "panic"]
SEVERITY_OUTCOMES = ["mild", "moderate", "severe", "extreme"]

# ── CPDs ─────────────────────────────────────────────────────────────────────
_N_EVENTS = len(EVENT_LABELS)

# escalation_level | event_type
_ESCALATION_CPD = np.array([
    # war  sanc  pand  nat_d  sup_s  pol_c  eco_c  cyber  enrg  trade
    [0.10, 0.30, 0.20, 0.25, 0.35, 0.20, 0.15, 0.30, 0.25, 0.70],  # low
    [0.35, 0.45, 0.40, 0.45, 0.45, 0.40, 0.40, 0.45, 0.40, 0.25],  # medium
    [0.55, 0.25, 0.40, 0.30, 0.20, 0.40, 0.45, 0.25, 0.35, 0.05],  # high
], dtype=float)

# contagion_scope | event_type
_CONTAGION_CPD = np.array([
    # war  sanc  pand  nat_d  sup_s  pol_c  eco_c  cyber  enrg  trade
    [0.10, 0.25, 0.05, 0.50, 0.20, 0.35, 0.10, 0.20, 0.20, 0.40],  # local
    [0.35, 0.40, 0.25, 0.35, 0.40, 0.40, 0.35, 0.45, 0.40, 0.45],  # regional
    [0.55, 0.35, 0.70, 0.15, 0.40, 0.25, 0.55, 0.35, 0.40, 0.15],  # global
], dtype=float)

# market_regime | escalation_level, contagion_scope
# Columns are ordered by (escalation, contagion):
# (low,local), (low,regional), (low,global), (med,local), ...
_REGIME_CPD = np.array([
    [0.70, 0.50, 0.30, 0.40, 0.25, 0.10, 0.20, 0.10, 0.05],  # risk_on
    [0.25, 0.35, 0.40, 0.40, 0.45, 0.40, 0.40, 0.35, 0.25],  # risk_off
    [0.05, 0.15, 0.30, 0.20, 0.30, 0.50, 0.40, 0.55, 0.70],  # panic
], dtype=float)

# policy_response | contagion_scope
_POLICY_CPD = np.array([
    [0.60, 0.30, 0.10],  # none
    [0.30, 0.45, 0.40],  # moderate
    [0.10, 0.25, 0.50],  # aggressive
], dtype=float)

# ── Optional pgmpy model holder (validation only) ───────────────────────────
_bn_model: Optional[DiscreteBayesianNetwork] = None
_bn_model_initialized = False


def _normalize_probs(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return np.full_like(values, 1.0 / max(len(values), 1))
    return values / total


def _compute_severity_cpd() -> np.ndarray:
    """
    Compute severity CPD from a scoring function.
    Higher escalation + wider contagion + weaker policy -> more severe.
    """
    esc_scores = {"low": 0.2, "medium": 0.5, "high": 0.9}
    cont_scores = {"local": 0.2, "regional": 0.5, "global": 0.9}
    pol_scores = {"none": 0.0, "moderate": -0.15, "aggressive": -0.30}

    values = np.zeros((4, 27), dtype=float)

    idx = 0
    for esc in ESCALATION_LEVELS:
        for cont in CONTAGION_SCOPES:
            for pol in POLICY_RESPONSES:
                raw = esc_scores[esc] + cont_scores[cont] + pol_scores[pol]
                norm = max(0.0, min(raw, 1.8)) / 1.8

                if norm < 0.3:
                    values[:, idx] = [0.60, 0.30, 0.08, 0.02]
                elif norm < 0.5:
                    values[:, idx] = [0.25, 0.45, 0.22, 0.08]
                elif norm < 0.7:
                    values[:, idx] = [0.08, 0.27, 0.45, 0.20]
                else:
                    values[:, idx] = [0.03, 0.10, 0.32, 0.55]
                idx += 1
    return values


_SEVERITY_CPD = _compute_severity_cpd()
_ESCALATION_BASELINE = _normalize_probs(_ESCALATION_CPD.mean(axis=1))
_CONTAGION_BASELINE = _normalize_probs(_CONTAGION_CPD.mean(axis=1))


def _build_bayesian_network() -> Optional[DiscreteBayesianNetwork]:
    """
    Construct the VORq Bayesian Network for structural validation (optional).

    DAG structure:
        event_type ──► escalation_level ──► market_regime
            │                │                    │
            └──► contagion_scope ──────────► severity_outcome
                      │                           ▲
                      └──► policy_response ───────┘
    """
    if DiscreteBayesianNetwork is None or TabularCPD is None:
        logger.info("pgmpy not installed; using native Bayesian inference engine.")
        return None

    model = DiscreteBayesianNetwork([
        ("event_type", "escalation_level"),
        ("event_type", "contagion_scope"),
        ("escalation_level", "market_regime"),
        ("contagion_scope", "market_regime"),
        ("contagion_scope", "policy_response"),
        ("escalation_level", "severity_outcome"),
        ("contagion_scope", "severity_outcome"),
        ("policy_response", "severity_outcome"),
    ])

    # CPD: event_type (uniform prior)
    cpd_event = TabularCPD(
        variable="event_type",
        variable_card=_N_EVENTS,
        values=[[1.0 / _N_EVENTS]] * _N_EVENTS,
        state_names={"event_type": EVENT_LABELS},
    )

    cpd_escalation = TabularCPD(
        variable="escalation_level",
        variable_card=3,
        values=_ESCALATION_CPD.tolist(),
        evidence=["event_type"],
        evidence_card=[_N_EVENTS],
        state_names={
            "escalation_level": ESCALATION_LEVELS,
            "event_type": EVENT_LABELS,
        },
    )

    cpd_contagion = TabularCPD(
        variable="contagion_scope",
        variable_card=3,
        values=_CONTAGION_CPD.tolist(),
        evidence=["event_type"],
        evidence_card=[_N_EVENTS],
        state_names={
            "contagion_scope": CONTAGION_SCOPES,
            "event_type": EVENT_LABELS,
        },
    )

    cpd_regime = TabularCPD(
        variable="market_regime",
        variable_card=3,
        values=_REGIME_CPD.tolist(),
        evidence=["escalation_level", "contagion_scope"],
        evidence_card=[3, 3],
        state_names={
            "market_regime": MARKET_REGIMES,
            "escalation_level": ESCALATION_LEVELS,
            "contagion_scope": CONTAGION_SCOPES,
        },
    )

    cpd_policy = TabularCPD(
        variable="policy_response",
        variable_card=3,
        values=_POLICY_CPD.tolist(),
        evidence=["contagion_scope"],
        evidence_card=[3],
        state_names={
            "policy_response": POLICY_RESPONSES,
            "contagion_scope": CONTAGION_SCOPES,
        },
    )

    cpd_severity = TabularCPD(
        variable="severity_outcome",
        variable_card=4,
        values=_SEVERITY_CPD.tolist(),
        evidence=["escalation_level", "contagion_scope", "policy_response"],
        evidence_card=[3, 3, 3],
        state_names={
            "severity_outcome": SEVERITY_OUTCOMES,
            "escalation_level": ESCALATION_LEVELS,
            "contagion_scope": CONTAGION_SCOPES,
            "policy_response": POLICY_RESPONSES,
        },
    )

    model.add_cpds(cpd_event, cpd_escalation, cpd_contagion,
                   cpd_regime, cpd_policy, cpd_severity)

    assert model.check_model(), "Bayesian Network CPDs do not sum to 1!"
    logger.info("Bayesian Network built: %d nodes, %d edges",
                len(model.nodes()), len(model.edges()))
    return model


def _get_model() -> Optional[DiscreteBayesianNetwork]:
    """Lazy-load the optional pgmpy model for structural validation."""
    global _bn_model, _bn_model_initialized
    if not _bn_model_initialized:
        _bn_model = _build_bayesian_network()
        _bn_model_initialized = True
    return _bn_model


def _blend_with_baseline(event_dist: np.ndarray, baseline: np.ndarray, confidence: float) -> np.ndarray:
    """
    Blend event-specific CPD with global baseline by classifier confidence.
    Lower confidence yields more conservative distributions.
    """
    weight = float(np.clip(0.35 + 0.65 * confidence, 0.35, 1.0))
    return _normalize_probs(weight * event_dist + (1.0 - weight) * baseline)


def _compute_marginals(event_label: str, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all downstream marginals from event evidence using native inference."""
    event_idx = EVENT_LABELS.index(event_label)

    # P(escalation | event) and P(contagion | event)
    esc_probs = _blend_with_baseline(_ESCALATION_CPD[:, event_idx], _ESCALATION_BASELINE, confidence)
    cont_probs = _blend_with_baseline(_CONTAGION_CPD[:, event_idx], _CONTAGION_BASELINE, confidence)

    # P(market_regime | event) = Σesc,cont P(regime|esc,cont) P(esc|e) P(cont|e)
    regime_probs = np.zeros(len(MARKET_REGIMES), dtype=float)
    for esc_idx, esc_p in enumerate(esc_probs):
        for cont_idx, cont_p in enumerate(cont_probs):
            column = esc_idx * len(CONTAGION_SCOPES) + cont_idx
            regime_probs += (esc_p * cont_p) * _REGIME_CPD[:, column]
    regime_probs = _normalize_probs(regime_probs)

    # P(policy | event) = Σcont P(policy|cont) P(cont|event)
    policy_probs = np.zeros(len(POLICY_RESPONSES), dtype=float)
    for cont_idx, cont_p in enumerate(cont_probs):
        policy_probs += cont_p * _POLICY_CPD[:, cont_idx]
    policy_probs = _normalize_probs(policy_probs)

    # P(severity | event) = Σesc,cont,pol P(sev|esc,cont,pol)P(pol|cont)P(cont|e)P(esc|e)
    severity_probs = np.zeros(len(SEVERITY_OUTCOMES), dtype=float)
    for esc_idx, esc_p in enumerate(esc_probs):
        for cont_idx, cont_p in enumerate(cont_probs):
            base_joint = esc_p * cont_p
            for pol_idx, pol_p in enumerate(_POLICY_CPD[:, cont_idx]):
                column = (
                    esc_idx * (len(CONTAGION_SCOPES) * len(POLICY_RESPONSES))
                    + cont_idx * len(POLICY_RESPONSES)
                    + pol_idx
                )
                severity_probs += base_joint * pol_p * _SEVERITY_CPD[:, column]
    severity_probs = _normalize_probs(severity_probs)

    return esc_probs, cont_probs, regime_probs, severity_probs, policy_probs


def _conditional_for_branch(esc_idx: int, cont_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return branch-conditional distributions:
    - P(regime | escalation, contagion)
    - P(severity | escalation, contagion) marginalizing policy.
    """
    regime_col = esc_idx * len(CONTAGION_SCOPES) + cont_idx
    regime_dist = _normalize_probs(_REGIME_CPD[:, regime_col])

    severity_dist = np.zeros(len(SEVERITY_OUTCOMES), dtype=float)
    for pol_idx, pol_p in enumerate(_POLICY_CPD[:, cont_idx]):
        sev_col = (
            esc_idx * (len(CONTAGION_SCOPES) * len(POLICY_RESPONSES))
            + cont_idx * len(POLICY_RESPONSES)
            + pol_idx
        )
        severity_dist += pol_p * _SEVERITY_CPD[:, sev_col]
    severity_dist = _normalize_probs(severity_dist)

    return regime_dist, severity_dist


def generate_scenario_tree(
    event_label: str,
    event_confidence: float = 0.5,
) -> dict:
    """
    Generate a multi-branch scenario tree for a classified event.

    Args:
        event_label: One of the 10 EVENT_LABELS from the classifier.
        event_confidence: Classifier confidence (used to modulate priors).

    Returns:
        {
            "primary_event": "war",
            "event_confidence": 0.48,
            "branches": [
                {
                    "scenario": "severe_global_panic",
                    "escalation": "high",
                    "contagion": "global",
                    "market_regime": "panic",
                    "severity": "extreme",
                    "probability": 0.18,
                    "severity_score": 0.95,
                },
                ...
            ],
            "summary_stats": {
                "expected_severity": 0.62,
                "p_panic": 0.35,
                "p_global_contagion": 0.55,
                "p_severe_or_worse": 0.42,
            }
        }
    """
    _get_model()  # optional consistency check if pgmpy exists

    if event_label not in EVENT_LABELS:
        event_label = "supply_shock"
    event_confidence = float(np.clip(event_confidence, 0.0, 1.0))

    try:
        esc_arr, cont_arr, regime_arr, sev_arr, pol_arr = _compute_marginals(
            event_label, event_confidence
        )
    except Exception as exc:
        logger.error("Bayesian inference failed; using fallback tree: %s", exc)
        return _fallback_scenario_tree(event_label, event_confidence)

    # Extract probability distributions
    esc_probs = {ESCALATION_LEVELS[i]: float(esc_arr[i]) for i in range(3)}
    cont_probs = {CONTAGION_SCOPES[i]: float(cont_arr[i]) for i in range(3)}
    regime_probs = {MARKET_REGIMES[i]: float(regime_arr[i]) for i in range(3)}
    sev_probs = {SEVERITY_OUTCOMES[i]: float(sev_arr[i]) for i in range(4)}
    pol_probs = {POLICY_RESPONSES[i]: float(pol_arr[i]) for i in range(3)}

    # Severity score mapping
    severity_scores = {"mild": 0.15, "moderate": 0.40, "severe": 0.70, "extreme": 0.95}

    # Build branches — top scenario combos with coherent joint likelihood.
    branches = []
    for esc_idx, esc in enumerate(ESCALATION_LEVELS):
        ep = esc_arr[esc_idx]
        for cont_idx, cont in enumerate(CONTAGION_SCOPES):
            cp = cont_arr[cont_idx]
            joint_p = ep * cp
            if joint_p < 0.02:
                continue

            regime_dist, severity_dist = _conditional_for_branch(esc_idx, cont_idx)
            best_sev_idx = int(np.argmax(severity_dist))
            best_reg_idx = int(np.argmax(regime_dist))
            best_sev = SEVERITY_OUTCOMES[best_sev_idx]
            best_reg = MARKET_REGIMES[best_reg_idx]

            # Path likelihood includes conditional certainty in chosen states.
            branch_prob = (
                joint_p
                * float(regime_dist[best_reg_idx])
                * float(severity_dist[best_sev_idx])
            )

            scenario_name = f"{best_sev}_{cont}_{best_reg}"
            branches.append({
                "scenario": scenario_name,
                "escalation": esc,
                "contagion": cont,
                "market_regime": best_reg,
                "severity": best_sev,
                "probability": round(branch_prob, 6),
                "severity_score": severity_scores[best_sev],
            })

    # Sort by probability and keep top branches.
    branches = sorted(branches, key=lambda x: -x["probability"])[:6]
    if not branches:
        return _fallback_scenario_tree(event_label, event_confidence)

    # Normalize branch probabilities to sum to 1
    total_p = sum(b["probability"] for b in branches)
    if total_p > 0:
        for b in branches:
            b["probability"] = round(b["probability"] / total_p, 4)

    # Summary statistics
    expected_severity = sum(
        float(sev_arr[i]) * severity_scores[s] for i, s in enumerate(SEVERITY_OUTCOMES)
    )

    return {
        "primary_event": event_label,
        "event_confidence": round(event_confidence, 4),
        "branches": branches,
        "distributions": {
            "escalation": {k: round(v, 4) for k, v in esc_probs.items()},
            "contagion": {k: round(v, 4) for k, v in cont_probs.items()},
            "market_regime": {k: round(v, 4) for k, v in regime_probs.items()},
            "severity": {k: round(v, 4) for k, v in sev_probs.items()},
            "policy_response": {k: round(v, 4) for k, v in pol_probs.items()},
        },
        "summary_stats": {
            "expected_severity": round(expected_severity, 4),
            "p_panic": round(regime_probs.get("panic", 0), 4),
            "p_global_contagion": round(cont_probs.get("global", 0), 4),
            "p_severe_or_worse": round(
                sev_probs.get("severe", 0) + sev_probs.get("extreme", 0), 4
            ),
        },
    }


def _fallback_scenario_tree(event_label: str, confidence: float) -> dict:
    """Fallback if Bayesian inference fails."""
    return {
        "primary_event": event_label,
        "event_confidence": round(confidence, 4),
        "branches": [
            {
                "scenario": "moderate_regional_risk_off",
                "escalation": "medium",
                "contagion": "regional",
                "market_regime": "risk_off",
                "severity": "moderate",
                "probability": 0.50,
                "severity_score": 0.40,
            },
            {
                "scenario": "severe_global_panic",
                "escalation": "high",
                "contagion": "global",
                "market_regime": "panic",
                "severity": "severe",
                "probability": 0.30,
                "severity_score": 0.70,
            },
            {
                "scenario": "mild_local_risk_on",
                "escalation": "low",
                "contagion": "local",
                "market_regime": "risk_on",
                "severity": "mild",
                "probability": 0.20,
                "severity_score": 0.15,
            },
        ],
        "distributions": {},
        "summary_stats": {
            "expected_severity": 0.45,
            "p_panic": 0.30,
            "p_global_contagion": 0.30,
            "p_severe_or_worse": 0.30,
        },
    }
