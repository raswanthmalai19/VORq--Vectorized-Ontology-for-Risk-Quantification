"""
VORq Event Extractor — ML-based geopolitical event classification.

Uses a fine-tuned DistilBERT model (10-class) trained on GDELT-style event data.
Falls back to keyword matching if the model cannot be loaded (e.g. missing weights).
"""

import json
import os
import re
import logging
import warnings

logger = logging.getLogger("vorq.event_extractor")

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "event_classifier")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# ── 10 event classes (matching trained model) ────────────────────────────────
EVENT_LABELS = [
    "war", "sanctions", "pandemic", "natural_disaster", "supply_shock",
    "political_crisis", "economic_crisis", "cyberattack", "energy_crisis",
    "trade_agreement",
]

# ── Country / region keyword map ──────────────────────────────────────────────
COUNTRY_PATTERNS = {
    "US":  (r"\b(us|usa|united states|america|american)\b", "United States", 37.09, -95.7),
    "CN":  (r"\b(china|chinese|beijing|prc)\b", "China", 35.86, 104.2),
    "TW":  (r"\b(taiwan|taiwanese|taipei|tsmc)\b", "Taiwan", 23.7, 121.0),
    "RU":  (r"\b(russia|russian|moscow|kremlin)\b", "Russia", 61.5, 105.3),
    "IN":  (r"\b(india|indian|delhi|mumbai)\b", "India", 20.59, 78.9),
    "JP":  (r"\b(japan|japanese|tokyo)\b", "Japan", 36.2, 138.25),
    "DE":  (r"\b(germany|german|berlin)\b", "Germany", 51.16, 10.45),
    "GB":  (r"\b(uk|united kingdom|britain|british|london)\b", "United Kingdom", 55.37, -3.44),
    "FR":  (r"\b(france|french|paris)\b", "France", 46.2, 2.21),
    "KR":  (r"\b(south korea|korean|seoul)\b", "South Korea", 35.9, 127.8),
    "SA":  (r"\b(saudi|riyadh|saudi arabia)\b", "Saudi Arabia", 23.9, 45.0),
    "AE":  (r"\b(uae|emirates|dubai|abu dhabi)\b", "UAE", 23.4, 53.8),
    "BR":  (r"\b(brazil|brazilian)\b", "Brazil", -14.2, -51.9),
    "AU":  (r"\b(australia|australian)\b", "Australia", -25.7, 133.8),
    "IR":  (r"\b(iran|iranian|tehran)\b", "Iran", 32.4, 53.7),
    "IL":  (r"\b(israel|israeli|tel aviv)\b", "Israel", 31.0, 34.8),
    "NK":  (r"\b(north korea|pyongyang|dprk)\b", "North Korea", 40.0, 127.0),
    "UA":  (r"\b(ukraine|ukrainian|kyiv|kiev)\b", "Ukraine", 48.4, 31.2),
    "PL":  (r"\b(poland|polish|warsaw)\b", "Poland", 51.9, 19.14),
    "NL":  (r"\b(netherlands|dutch|amsterdam)\b", "Netherlands", 52.1, 5.29),
    "MX":  (r"\b(mexico|mexican)\b", "Mexico", 23.6, -102.5),
    "CA":  (r"\b(canada|canadian)\b", "Canada", 56.1, -106.3),
    "EU":  (r"\b(europe|european|eu\b)\b", "Europe (EU)", 50.1, 9.7),
}

# ── Region-based default country impacts per event type ──────────────────────
EVENT_COUNTRY_DEFAULTS = {
    "war":               [("TW", -45), ("CN", -25), ("US", -8), ("JP", -12), ("KR", -10)],
    "sanctions":         [("CN", -30), ("US", -5), ("TW", -15), ("KR", -8), ("DE", -6)],
    "pandemic":          [("CN", -20), ("US", -15), ("IN", -18), ("BR", -16), ("EU", -10)],
    "natural_disaster":  [("JP", -25), ("US", -10), ("IN", -15), ("AU", -8)],
    "supply_shock":      [("CN", -20), ("DE", -12), ("US", -8), ("JP", -10), ("KR", -9)],
    "political_crisis":  [("RU", -20), ("UA", -30), ("EU", -8), ("US", -4)],
    "economic_crisis":   [("CN", -25), ("US", -15), ("EU", -12), ("JP", -8), ("BR", -10)],
    "cyberattack":       [("US", -18), ("EU", -10), ("CN", -8), ("RU", -5)],
    "energy_crisis":     [("SA", -30), ("AE", -20), ("EU", -15), ("US", -8), ("IN", -12)],
    "trade_agreement":   [("US", 8), ("EU", 6), ("CN", 5), ("JP", 4)],
}

# ── Singleton model holder ────────────────────────────────────────────────────
_model = None
_tokenizer = None
_label_map = None
_model_loaded = False
_device = "cpu"


def _resolve_torch_device(torch_module) -> str:
    """
    Select the best inference device.
    Prioritizes Apple Silicon GPU (MPS) when available.
    """
    prefer_gpu = os.environ.get("VORQ_ENABLE_GPU", "1") != "0"
    if not prefer_gpu:
        return "cpu"

    try:
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    try:
        if torch_module.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def _load_model():
    """Load DistilBERT model and tokenizer. Called once lazily."""
    global _model, _tokenizer, _label_map, _model_loaded, _device

    if _model_loaded:
        return _model is not None

    _model_loaded = True  # prevent repeated attempts

    try:
        if not os.path.isfile(os.path.join(MODEL_DIR, "model.safetensors")):
            logger.warning("Model weights not found at %s — falling back to keywords", MODEL_DIR)
            return False

        import torch
        if hasattr(torch, "library") and not hasattr(torch.library, "register_fake"):
            # Compatibility shim for older torch versions with newer transformers.
            def _register_fake(*args, **kwargs):
                def _decorator(func):
                    return func
                return _decorator
            torch.library.register_fake = _register_fake  # type: ignore[attr-defined]

        os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
        warnings.filterwarnings(
            "ignore",
            message="Failed to load image Python extension",
            module="torchvision.io.image",
        )
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info("Loading VORq event classifier from %s", MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        _device = _resolve_torch_device(torch)
        _model.to(_device)
        _model.eval()

        # Load label map
        if os.path.isfile(LABEL_MAP_PATH):
            with open(LABEL_MAP_PATH, "r") as f:
                _label_map = json.load(f)
        else:
            # Use config.json id2label
            config_path = os.path.join(MODEL_DIR, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            _label_map = {"id2label": config.get("id2label", {})}

        logger.info(
            "Model loaded successfully — %d classes on %s",
            len(_label_map.get("id2label", {})),
            _device,
        )
        return True

    except Exception as e:
        logger.error("Failed to load model: %s — falling back to keywords", e)
        _model = None
        _tokenizer = None
        _device = "cpu"
        return False


CONFIDENCE_THRESHOLD = 0.25  # Below this, model predictions are unreliable


def classify_event(text: str) -> dict:
    """
    Classify a scenario text into one of 10 event types.

    Uses a HYBRID approach:
    - If the ML model has high confidence (>25%), trust the model.
    - If the model confidence is low, use keyword-based classification.
    - If both agree, boost the confidence score.

    Returns:
        {
            "label": "war",
            "confidence": 0.92,
            "all_scores": {"war": 0.92, "sanctions": 0.05, ...},
            "method": "model" | "keyword" | "hybrid"
        }
    """
    text = text.strip()
    if not text:
        return {"label": "supply_shock", "confidence": 0.1, "all_scores": {}, "method": "fallback"}

    # Always compute keyword classification
    keyword_result = _keyword_classify(text)

    # Try ML model
    model_result = None
    if _load_model() and _model is not None:
        try:
            import torch

            inputs = _tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=512, padding=True
            )
            # DistilBERT does not use token_type_ids.
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = _model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].detach().cpu()

            id2label = _label_map.get("id2label", {})
            all_scores = {}
            for idx, prob in enumerate(probs):
                label = id2label.get(str(idx), f"class_{idx}")
                all_scores[label] = round(prob.item(), 4)

            best_idx = probs.argmax().item()
            best_label = id2label.get(str(best_idx), "supply_shock")
            best_conf = probs[best_idx].item()

            model_result = {
                "label": best_label,
                "confidence": round(best_conf, 4),
                "all_scores": all_scores,
            }
        except Exception as e:
            logger.error("Model inference failed: %s — using keywords", e)

    # ── Hybrid decision logic ──────────────────────────────────────────────
    if model_result is None:
        # Model unavailable — pure keyword
        return keyword_result

    if model_result["confidence"] >= CONFIDENCE_THRESHOLD:
        # Model is confident — trust it
        return {**model_result, "method": "model"}

    # Model confidence is low — check if keywords agree
    if model_result["label"] == keyword_result["label"]:
        # Both agree — use the label with boosted confidence
        return {
            "label": model_result["label"],
            "confidence": round(min(model_result["confidence"] + 0.35, 0.95), 4),
            "all_scores": model_result["all_scores"],
            "method": "hybrid",
        }

    # Disagreement with low model confidence — prefer keywords
    # (keywords are rule-based and more reliable for clear-cut scenarios)
    keyword_conf = keyword_result["confidence"]
    if keyword_conf > 0:
        # Keywords found a clear match — use it
        return {
            "label": keyword_result["label"],
            "confidence": round(min(keyword_conf + 0.25, 0.90), 4),
            "all_scores": keyword_result["all_scores"],
            "method": "keyword",
        }

    # No strong signal from either — use model as last resort
    return {**model_result, "method": "model"}


def get_model_status() -> dict:
    """Expose model runtime health for API diagnostics."""
    loaded = bool(_model_loaded and _model is not None and _tokenizer is not None)
    return {
        "loaded": loaded,
        "device": _device if loaded else "cpu",
    }


def _keyword_classify(text: str) -> dict:
    """Rule-based keyword classification fallback with weighted multi-word matching."""
    text_lower = text.lower()

    # Keywords with weights — multi-word phrases get higher weight to avoid ambiguity.
    # "attack" alone is ambiguous (military vs cyber), so we use qualified phrases.
    keyword_map = {
        "war":              [("war", 2), ("conflict", 2), ("invasion", 2), ("invade", 2),
                             ("military", 1.5), ("military attack", 3), ("bomb", 2),
                             ("missile", 2), ("troops", 2), ("armed", 1.5), ("combat", 2)],
        "sanctions":        [("sanction", 3), ("embargo", 3), ("tariff", 2),
                             ("trade war", 3), ("export control", 3), ("export ban", 3),
                             ("trade restriction", 3), ("import ban", 3)],
        "pandemic":         [("pandemic", 3), ("virus", 2), ("covid", 3), ("outbreak", 2),
                             ("epidemic", 3), ("disease", 2), ("quarantine", 2), ("infection", 1.5)],
        "natural_disaster": [("earthquake", 3), ("tsunami", 3), ("hurricane", 3), ("flood", 2),
                             ("wildfire", 3), ("typhoon", 3), ("volcano", 3), ("drought", 2),
                             ("tornado", 3), ("cyclone", 3)],
        "supply_shock":     [("supply chain", 3), ("shortage", 2), ("disruption", 1.5),
                             ("bottleneck", 2), ("logistics", 1.5), ("port closure", 3),
                             ("semiconductor shortage", 4)],
        "political_crisis": [("coup", 3), ("protest", 2), ("revolution", 2),
                             ("election crisis", 3), ("government collapse", 3),
                             ("political instability", 3), ("regime", 2), ("uprising", 2)],
        "economic_crisis":  [("recession", 3), ("inflation", 2), ("debt crisis", 3),
                             ("market crash", 3), ("economic collapse", 3), ("bankruptcy", 2),
                             ("default", 1.5), ("banking", 2), ("collapse", 2),
                             ("financial crisis", 3), ("credit crisis", 3), ("depression", 2)],
        "cyberattack":      [("cyber", 3), ("hack", 3), ("ransomware", 3), ("data breach", 3),
                             ("malware", 3), ("ddos", 3), ("digital attack", 3),
                             ("cyber attack", 4), ("phishing", 2), ("hacker", 2)],
        "energy_crisis":    [("oil", 2), ("gas", 1.5), ("energy", 2), ("fuel", 2),
                             ("opec", 3), ("pipeline", 2), ("refinery", 2), ("power grid", 3),
                             ("oil price", 3), ("energy crisis", 4)],
        "trade_agreement":  [("trade deal", 3), ("trade agreement", 3), ("free trade", 3),
                             ("partnership", 1.5), ("economic pact", 3), ("bilateral", 2),
                             ("trade pact", 3)],
    }

    scores = {}
    for label, keyword_weights in keyword_map.items():
        score = sum(weight for kw, weight in keyword_weights if kw in text_lower)
        scores[label] = score

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score == 0:
        # No keywords matched — default to supply_shock
        best_label = "supply_shock"

    # Normalize scores to pseudo-probabilities
    total = max(sum(scores.values()), 1)
    all_scores = {k: round(v / total, 4) for k, v in scores.items()}

    return {
        "label": best_label,
        "confidence": round(max(all_scores.values(), default=0.1), 4),
        "all_scores": all_scores,
        "method": "keyword",
    }


def extract_entities(text: str) -> dict:
    """
    Extract countries and regions mentioned in the scenario text.

    Returns:
        {
            "countries": [{"id": "US", "name": "United States", "lat": 37.09, "lon": -95.7}, ...],
            "raw_matches": ["us", "china", ...]
        }
    """
    text_lower = text.lower()
    countries = []
    raw_matches = []
    seen_ids = set()

    for cid, (pattern, name, lat, lon) in COUNTRY_PATTERNS.items():
        if re.search(pattern, text_lower):
            if cid not in seen_ids:
                countries.append({"id": cid, "name": name, "lat": lat, "lon": lon})
                seen_ids.add(cid)
                raw_matches.append(name)

    return {"countries": countries, "raw_matches": raw_matches}


def get_country_impacts(event_label: str, mentioned_countries: list) -> list:
    """
    Generate country-level impact estimates based on event type and mentioned countries.
    Mentioned countries get higher impact weights.
    """
    defaults = EVENT_COUNTRY_DEFAULTS.get(event_label, [])
    mentioned_ids = {c["id"] for c in mentioned_countries}

    impacts = []
    seen = set()

    # First: mentioned countries with boosted impact
    for country in mentioned_countries:
        cid = country["id"]
        if cid in seen:
            continue
        seen.add(cid)
        # Find default impact or assign based on event severity
        default_impact = next((imp for did, imp in defaults if did == cid), None)
        if default_impact is not None:
            impact_pct = default_impact
        else:
            # Assign moderate negative impact for mentioned countries
            impact_pct = -15.0 if event_label != "trade_agreement" else 5.0
        impacts.append({
            "id": cid,
            "name": country["name"],
            "impact_pct": round(impact_pct, 1),
        })

    # Then: fill in default countries if not already mentioned
    for cid, imp in defaults:
        if cid in seen:
            continue
        seen.add(cid)
        name = COUNTRY_PATTERNS.get(cid, (None, cid))[1]
        impacts.append({
            "id": cid,
            "name": name,
            "impact_pct": round(imp, 1),
        })

    return impacts[:6]  # limit to top 6


def map_event_to_shock_id(event_label: str, text: str) -> str:
    """
    Map the ML-predicted event label + text context to the knowledge graph shock ID.
    This bridges the model output to the KG causal links.
    """
    text_lower = text.lower()

    # Direct event-type mappings
    mapping = {
        "war":              "war_taiwan",
        "sanctions":        "sanctions_china",
        "pandemic":         "pandemic_global",
        "natural_disaster": "natural_disaster",
        "supply_shock":     "supply_shock",
        "political_crisis": "political_crisis",
        "economic_crisis":  "economic_crisis",
        "cyberattack":      "cyberattack",
        "energy_crisis":    "oil_shock_mideast",
        "trade_agreement":  "trade_agreement",
    }

    shock_id = mapping.get(event_label, "supply_shock")

    # Contextual refinements
    if event_label == "war":
        if "india" in text_lower and "china" in text_lower:
            shock_id = "war_india_china"
        elif "russia" in text_lower or "ukraine" in text_lower:
            shock_id = "war_russia_ukraine"
        elif "taiwan" in text_lower or "china" in text_lower:
            shock_id = "war_taiwan"
        else:
            shock_id = "war_general"

    if event_label == "sanctions":
        if "russia" in text_lower:
            shock_id = "sanctions_russia"
        elif "iran" in text_lower:
            shock_id = "sanctions_iran"
        else:
            shock_id = "sanctions_china"

    if event_label == "energy_crisis":
        if "europe" in text_lower or "gas" in text_lower:
            shock_id = "energy_crisis_europe"
        else:
            shock_id = "oil_shock_mideast"

    return shock_id
