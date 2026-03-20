# VORQ — Institutional-Grade Geopolitical Risk Intelligence 🌍

> **AI-powered scenario analysis engine.** Enter a geopolitical or economic scenario → receive calibrated, probabilistic impact analysis with full causal explanation, confidence metrics, and risk bounds.

**Status:** ✅ Production Ready | **Version:** 3.1.0

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- macOS/Linux/Windows
- ~5GB disk space

### Installation & Launch

```bash
# 1. Clone repository
git clone https://github.com/yourusername/VORq.git
cd VORq

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start backend (Terminal 1)
python3 -m uvicorn vorq.api.main:app --port 8000 --reload

# 5. Start frontend (Terminal 2)
streamlit run vorq/ui/app.py --server.port 8501
```

**Dashboard:** http://localhost:8501  
**API Docs:** http://localhost:8000/docs  
**Health Check:** http://localhost:8000/health

---

## 📊 What It Does

### Input
```
"Automotive sector during a geopolitical shock?"
"Semiconductor export ban to China"
"Major pandemic outbreak in India"
```



---

## 🔧 Core Features

### ✅ Institutional-Grade Calibration (v3.1)
- **Hard impact bounds:** [-80%, +30%] (no impossible -112%)
- **Historical calibration:** 10 event types anchored to real crises (2008, COVID, 9/11, etc.)
- **Shock intensity variable:** 0-1 scale linked to Bayesian severity
- **Company sensitivities:** Realistic differentiation (JPM 1.3x, HSBC 0.9x)
- **Confidence capping:** Max 95% (not >100% claims)

### 🎯 Scenario Analysis
- **5 scenario branches** with normalized probabilities
- **Escalation levels:** Low/Medium/High
- **Contagion scopes:** Local/Regional/Global
- **Market regimes:** Risk-On/Risk-Off/Panic
- **Policy responses:** None/Moderate/Aggressive

### 📈 Risk Quantification
- **Student-t Monte Carlo** (df=4) for fat tails
- **GARCH volatility clustering**
- **Regime-dependent variance scaling**
- **VaR/CVaR metrics** (95th, 99th percentile)
- **Sector-level risk detail**

### 🧠 Causal Analysis
- **Structural causal model** with 3-hop propagation
- **Regime-aware edge weights** (expansion/slowdown/contraction/crisis)
- **Sector cross-coupling matrix** (30+ interdependencies)
- **Impact chain visualization**
- **Mechanism explanations** for each path

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ User Input (Text Scenario)                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Event Classification (ML)   │ ✅ DistilBERT + Keywords
        │  → 10 event types            │ ✅ Entity extraction
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Bayesian Scenario Generator │ ✅ 5 branches
        │  → Escalation/Contagion/etc  │ ✅ Normalized probabilities
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Macro Regime Detection      │ ✅ FRED data
        │  → Economic context          │ ✅ VIX, yield spread
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Causal Graph Propagation    │ ✅ 3-hop cascades
        │  → Shock impact calculation  │ ✅ Historical calibration
        │  → Hard bounds enforcement   │ ✅ [-80%, +30%]
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Monte Carlo Sampling        │ ✅ 2000 iterations
        │  → Risk distribution         │ ✅ Fat tails (Student-t)
        │  → VaR/CVaR metrics          │ ✅ Volatility clustering
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Confidence Calibration      │ ✅ Max 95%
        │  → Classification accuracy   │ ✅ Event-specific
        │  → Scenario weighting        │ ✅ Transparent bounds
        └──────────────┬───────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Output: Risk Score, Scenarios, Company Impacts             │
│ + Explanations, Mitigations, Calibration Details           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🌐 API Endpoints

### `/simulate` — Full Pipeline
```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_text": "USA president death",
    "mc_iterations": 2000
  }'
```

### `/quick-score` — Fast Assessment
```bash
curl -X POST http://localhost:8000/quick-score \
  -H "Content-Type: application/json" \
  -d '{"scenario_text": "Oil supply shock"}'
```

### `/macro-context` — Current Regime
```bash
curl http://localhost:8000/macro-context
```

### `/scenario-tree` — Bayesian Branches
```bash
curl "http://localhost:8000/scenario-tree?event=war&confidence=0.6"
```

### `/health` — System Status
```bash
curl http://localhost:8000/health
```

### `/docs` — Interactive Documentation
```
http://localhost:8000/docs
```

---

## 📁 Project Structure

```
VORq/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project metadata
├── .gitignore                         # Git ignore rules
│
├── vorq/
│   ├── __init__.py
│   ├── api/
│   │   ├── main.py                   # FastAPI routes & endpoints
│   │   └── vorq_knowledge_graph.json  # Sector + company data
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── event_extractor.py        # NLP event classification
│   │   ├── bayesian_scenarios.py     # Scenario tree generation
│   │   ├── causal_model.py           # Causal graph propagation ⭐
│   │   ├── validation.py             # Brier score tracking
│   │   └── __pycache__/
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fred_client.py            # Macro regime detection
│   │   ├── fred_cache.json           # FRED API cache
│   │   ├── prediction_log.json       # Prediction history
│   │   ├── kb/
│   │   │   └── seed_knowledge_base.json
│   │   └── __pycache__/
│   │
│   ├── ui/
│   │   ├── app.py                    # Streamlit dashboard
│   │   └── assets/
│   └── __pycache__/
│
├── notebooks/                         # Jupyter notebooks (training)
│   ├── 01_knowledge_base.ipynb
│   ├── 02_engine_core.ipynb
│   ├── 03_api_server.ipynb
│   └── 04_ui_launcher.ipynb
│
├── tests/                             # Test suite
│   ├── test_api.py
│   ├── test_bayesian.py
│   ├── test_causal_model.py
│   ├── test_event_extractor.py
│   └── __init__.py
│
├── scripts/
│   └── download_models.sh
│
└── colab/                             # Google Colab notebooks
    └── *.ipynb
```

---

## 📊 Key Metrics & Validation

### Calibration Metrics
- **Confidence Range:** 30-95% (realistic bounds)
- **Impact Bounds:** -80% to +30% (realistic limits)
- **Brier Score:** Tracked for prediction accuracy
- **Scenario Coverage:** 5 branches covering 90%+ of outcomes

### Historical Anchors
```
Event Type              Max Impact    Reference
────────────────────────────────────────────────
War                     -60%          Iraq War
Pandemic                -70%          COVID-19
Economic Crisis         -80%          2008 Financial Crisis
Political Crisis        -30%          JFK assassination
Sanctions               -45%          Russia sanctions
Supply Shock            -40%          1973 Oil Crisis
```

---

## 🔄 Workflow

### Development
```bash
# Activate environment
source .venv/bin/activate

# Run tests
python3 -m pytest tests/ -v

# Start with auto-reload
python3 -m uvicorn vorq.api.main:app --reload
streamlit run vorq/ui/app.py
```

### Production
```bash
# Deploy backend
gunicorn -w 4 -k uvicorn.workers.UvicornWorker vorq.api.main:app

# Deploy frontend (Streamlit Enterprise)
streamlit run vorq/ui/app.py --server.headless true
```

---

## 📚 Documentation

- **Architecture:** See `vorq/engine/causal_model.py` for core logic
- **API Design:** See `vorq/api/main.py` for endpoint definitions
- **Data Model:** See `vorq/api/vorq_knowledge_graph.json` for sector/company data
- **Validation:** See `vorq/engine/validation.py` for Brier score metrics

---

## 🙏 Acknowledgments

- **Event Classification:** DistilBERT (HuggingFace)
- **Scenario Modeling:** pgmpy (Bayesian Networks)
- **Graph Analysis:** NetworkX
- **Causal Inference:** Structural Causal Models (Pearl)
- **Risk Metrics:** Financial risk management standards (VaR/CVaR)

---

## 📝 License

MIT License — See LICENSE file for details

---

## 🚀 Roadmap

- [ ] Add real-time news ingestion (GDELT, EventRegistry)
- [ ] Expand to 50+ event types
- [ ] Company-level sentiment analysis
- [ ] Supply chain graph visualization
- [ ] Alternative scenario forecasting
- [ ] Multi-currency impact analysis
- [ ] Regulatory impact module

---

## 📞 Support

For issues, questions, or contributions:
1. File a GitHub issue
2. Check existing documentation
3. Review example notebooks in `notebooks/`

---

**Last Updated:** March 2026 | **Status:** Production Ready ✅
# VORq--Vectorized-Ontology-for-Risk-Quantification
