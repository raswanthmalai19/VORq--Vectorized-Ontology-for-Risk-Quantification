"""
VORq FRED Client — Macroeconomic data integration via the free FRED API.

Fetches real macro indicators (GDP, CPI, VIX, oil, etc.), classifies the
current economic regime, and provides shock severity modulation factors.
Falls back to cached/hardcoded data if no API key is available.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

# Auto-load .env (works with or without python-dotenv installed)
try:
    from dotenv import load_dotenv
    _env = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(_env)
except ImportError:
    pass

logger = logging.getLogger("vorq.fred")

# ── FRED series IDs ──────────────────────────────────────────────────────────
MACRO_SERIES = {
    "gdp_growth":       "A191RL1Q225SBEA",  # Real GDP growth (quarterly %)
    "cpi_yoy":          "CPIAUCSL",          # CPI All Urban Consumers
    "unemployment":     "UNRATE",            # Unemployment Rate
    "fed_funds_rate":   "FEDFUNDS",          # Federal Funds Rate
    "m2_money_supply":  "M2SL",              # M2 Money Stock
    "vix":              "VIXCLS",            # CBOE Volatility Index
    "crude_oil_wti":    "DCOILWTICO",        # Crude Oil WTI
    "yield_spread":     "T10Y2Y",            # 10Y-2Y Treasury Spread
}

# ── Cache settings ───────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_FILE = os.path.join(CACHE_DIR, "fred_cache.json")
CACHE_TTL_HOURS = 24

# ── Fallback macro data (reasonable baseline values) ─────────────────────────
FALLBACK_MACRO = {
    "gdp_growth": 2.1,
    "cpi_yoy": 3.2,
    "unemployment": 4.0,
    "fed_funds_rate": 5.25,
    "m2_money_supply": 21000.0,  # billions
    "vix": 18.5,
    "crude_oil_wti": 75.0,
    "yield_spread": 0.45,
    "regime": "slowdown",
    "timestamp": "fallback",
}


class FredClient:
    """Fetch macroeconomic data from the FRED API with caching and fallback."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self._fred = None
        self._init_attempted = False
        self._cache = None
        self._cache_loaded = False

    def _init_fred(self):
        """Initialize the fredapi client."""
        if self._init_attempted:
            return self._fred is not None
        self._init_attempted = True

        if self._fred is not None:
            return True
        if not self.api_key:
            logger.info("No FRED API key — using fallback macro data")
            return False
        try:
            from fredapi import Fred
            self._fred = Fred(api_key=self.api_key)
            logger.info("FRED client initialized (key: ...%s)", self.api_key[-4:])
            return True
        except Exception as e:
            logger.warning("FRED client unavailable (%s) — using fallback macro data", e)
            return False

    def _load_cache(self) -> Optional[dict]:
        """Load cached data if still fresh."""
        if self._cache_loaded:
            return self._cache
        self._cache_loaded = True

        try:
            if os.path.isfile(CACHE_FILE):
                with open(CACHE_FILE, "r") as f:
                    data = json.load(f)
                ts = data.get("timestamp", "")
                if ts and ts != "fallback":
                    cached_time = datetime.fromisoformat(ts)
                    if datetime.now() - cached_time < timedelta(hours=CACHE_TTL_HOURS):
                        self._cache = data
                        logger.info("Using cached FRED data from %s", ts)
                        return data
        except Exception as e:
            logger.warning("Cache read error: %s", e)
        return None

    def _save_cache(self, data: dict):
        """Save fetched data to cache."""
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Cache write error: %s", e)

    def get_macro_context(self) -> dict:
        """
        Fetch current macroeconomic indicators and classify the regime.

        Returns:
            {
                "gdp_growth": 2.1,
                "cpi_yoy": 3.2,
                "unemployment": 4.0,
                "fed_funds_rate": 5.25,
                "vix": 18.5,
                "crude_oil_wti": 75.0,
                "yield_spread": 0.45,
                "regime": "slowdown",
                "regime_confidence": 0.72,
                "timestamp": "2026-03-19T17:00:00",
            }
        """
        # Try cache first
        cached = self._load_cache()
        if cached:
            # Recalculate regime in case regime logic changed
            cached["regime"], cached["regime_confidence"] = self._detect_regime(cached)
            return cached

        # Try fetching from FRED
        if self._init_fred():
            try:
                data = self._fetch_from_fred()
                data["regime"], data["regime_confidence"] = self._detect_regime(data)
                data["timestamp"] = datetime.now().isoformat()
                self._save_cache(data)
                return data
            except Exception as e:
                logger.error("FRED fetch failed: %s — using fallback", e)

        # Fallback
        fallback = dict(FALLBACK_MACRO)
        fallback["regime"], fallback["regime_confidence"] = self._detect_regime(fallback)
        return fallback

    def _fetch_from_fred(self) -> dict:
        """Fetch the latest observation for each macro series."""
        data = {}
        for key, series_id in MACRO_SERIES.items():
            try:
                series = self._fred.get_series(series_id)
                series = series.dropna()
                if series.empty:
                    data[key] = FALLBACK_MACRO.get(key, 0)
                    continue

                # CPI: calculate Year-over-Year % change instead of raw index
                if key == "cpi_yoy":
                    if len(series) >= 13:
                        latest = float(series.iloc[-1])
                        year_ago = float(series.iloc[-13])
                        data[key] = round((latest - year_ago) / year_ago * 100, 2)
                    else:
                        data[key] = FALLBACK_MACRO.get(key, 3.2)
                else:
                    data[key] = round(float(series.iloc[-1]), 2)
            except Exception as e:
                logger.warning("Failed to fetch %s (%s): %s", key, series_id, e)
                data[key] = FALLBACK_MACRO.get(key, 0)
        return data

    def _detect_regime(self, data: dict) -> tuple:
        """
        Classify the current economic regime based on multiple indicators.

        Uses a composite scoring system:
        - GDP growth: >2.5% = expansion signal, <0% = crisis signal
        - VIX: <15 = calm, >30 = stress, >45 = crisis
        - Yield spread: negative = recession signal (inverted curve)
        - Unemployment: <4% = tight labor market, >6% = weakness
        - CPI: >5% = overheating

        Returns: (regime_name, confidence)
        """
        gdp = data.get("gdp_growth", 2.0)
        vix = data.get("vix", 18.0)
        spread = data.get("yield_spread", 0.5)
        unemp = data.get("unemployment", 4.0)
        cpi = data.get("cpi_yoy", 3.0)

        # Scoring: positive = expansion, negative = contraction
        score = 0.0
        signals = 0

        # GDP growth
        if gdp > 3.0:
            score += 2.0
        elif gdp > 2.0:
            score += 1.0
        elif gdp > 0:
            score -= 0.5
        elif gdp > -1.0:
            score -= 2.0
        else:
            score -= 3.0
        signals += 1

        # VIX (inverse — high VIX = bad)
        if vix < 15:
            score += 1.5
        elif vix < 20:
            score += 0.5
        elif vix < 30:
            score -= 1.0
        elif vix < 45:
            score -= 2.5
        else:
            score -= 3.5
        signals += 1

        # Yield spread (negative = recession predictor)
        if spread > 1.0:
            score += 1.0
        elif spread > 0:
            score += 0.3
        elif spread > -0.5:
            score -= 1.5
        else:
            score -= 2.5
        signals += 1

        # Unemployment
        if unemp < 3.5:
            score += 1.0
        elif unemp < 5.0:
            score += 0.3
        elif unemp < 7.0:
            score -= 1.0
        else:
            score -= 2.0
        signals += 1

        # Normalize
        normalized = score / max(signals, 1)

        if normalized > 1.5:
            regime = "expansion"
            confidence = min(0.6 + normalized * 0.1, 0.95)
        elif normalized > 0:
            regime = "slowdown"
            confidence = 0.55 + abs(normalized) * 0.1
        elif normalized > -1.5:
            regime = "contraction"
            confidence = 0.55 + abs(normalized) * 0.1
        else:
            regime = "crisis"
            confidence = min(0.6 + abs(normalized) * 0.1, 0.95)

        return regime, round(confidence, 3)

    def get_shock_modulator(self, regime: Optional[str] = None) -> dict:
        """
        Get shock severity modulation factors based on macro regime.

        During crisis regimes, negative shocks hit ~1.6× harder.
        During expansion, negative shocks are buffered to 0.7×.
        """
        if regime is None:
            ctx = self.get_macro_context()
            regime = ctx["regime"]

        modulators = {
            "expansion":   {"shock_multiplier": 0.7,  "recovery_speed": 1.3},
            "slowdown":    {"shock_multiplier": 1.0,  "recovery_speed": 1.0},
            "contraction": {"shock_multiplier": 1.3,  "recovery_speed": 0.7},
            "crisis":      {"shock_multiplier": 1.6,  "recovery_speed": 0.5},
        }

        return {
            "regime": regime,
            **modulators.get(regime, modulators["slowdown"]),
        }


# ── Singleton ────────────────────────────────────────────────────────────────
_fred_client: Optional[FredClient] = None


def get_fred_client() -> FredClient:
    global _fred_client
    if _fred_client is None:
        _fred_client = FredClient()
    return _fred_client
