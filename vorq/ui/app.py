"""VORq — Global Risk Intelligence · Dark Navy Futuristic UI"""
import random, math, time
from datetime import datetime
import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="VORq · Risk Intelligence", page_icon="◈",
                   layout="wide", initial_sidebar_state="collapsed")

import base64, os as _os
API = _os.getenv("VORQ_API_URL", "http://127.0.0.1:8000").rstrip("/")
_LOGO_PATH = _os.path.join(_os.path.dirname(__file__), "assets", "logo.png")
def _logo_b64():
    try:
        with open(_LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""
_LOGO = _logo_b64()

for k, v in [("conversation",[]),("active_id",None),("pending",None),("chip_query",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Palette ─────────────────────────────────────────────────────────────────
BG       = "#2B2623"          # --bg-main
BG_DARK  = "#26211E"          # --bg-dark
BG_LIGHT = "#302A26"          # --bg-light
SURFACE  = "#3A332F"          # --card
SURF2    = "#443C37"          # --card-hover
BORDER   = "#4A403A"          # --border
INPUT_BG = "#26211E"          # similar to bg-dark for depth
TEXT     = "#E6E1DC"          # --text-primary
MUTED    = "#B8B0AA"          # --text-secondary
DIM      = "#8A817C"          # --text-muted
ACCENT   = "#10A37F"          # teal (remains same)
ACCENT2  = "#0E8F6E"          # teal hover
RED      = "#F87171"
ORANGE   = "#FB923C"
YELLOW   = "#FBBF24"
GREEN    = "#34D399"
PURPLE   = "#A78BFA"

RISK_C   = {"CRITICAL":RED,"HIGH":ORANGE,"ELEVATED":YELLOW,"MODERATE":GREEN,"LOW":ACCENT}
PCTL     = {"CRITICAL":"94","HIGH":"83","ELEVATED":"67","MODERATE":"41","LOW":"18"}

CHIP_META = [
    ("US–China military conflict",    RED,    "war"),
    ("Semiconductor export ban",       ORANGE, "sanctions"),
    ("Middle East oil supply shock",   YELLOW, "energy"),
    ("Global pandemic resurgence",     PURPLE, "pandemic"),
    ("Chinese economic collapse",      ACCENT, "economy"),
]
CAUSAL = {
    "Technology & Semiconductors":"Primary fabs & R&D are concentrated in conflict zones — export controls freeze chip supply and IP licensing instantly.",
    "Defense & Aerospace":"Governments surge procurement; supply chains reorient as escalation unfolds.",
    "General Manufacturing":"Cross-border components snap; just-in-time buffers collapse within weeks.",
    "Automotive":"EV battery materials and micro-chip shortages cascade through production lines.",
    "Energy & Oil":"Shipping lanes and sanctions combine to disrupt global crude flows and refinery margins.",
    "Financial Services":"Capital flight, currency volatility, sovereign downgrades hit bank exposure hard.",
    "Agriculture & Food":"Fertiliser and grain export disruptions spike food inflation across import-dependent nations.",
}
ICONS = {"war":"⚔","sanctions":"🚫","pandemic":"🦠","natural_disaster":"🌪",
         "supply_shock":"◈","political_crisis":"⬡","economic_crisis":"◇",
         "cyberattack":"◉","energy_crisis":"◈","trade_agreement":"◈"}

def _rc(lv): return RISK_C.get(lv, MUTED)
def _sl(pct):
    if pct<-15: return "CRITICAL"
    if pct<-8:  return "HIGH"
    if pct<0:   return "ELEVATED"
    if pct>5:   return "LOW"
    return "MODERATE"
def _short(q): w=q.strip().split(); return " ".join(w[:6])+("…" if len(w)>6 else "")

def _simulate(query):
    try:
        r = requests.post(f"{API}/simulate",
                          json={"scenario_text":query,"mc_iterations":1000},timeout=90)
        r.raise_for_status(); return r.json()
    except requests.ConnectionError:
        return {"error":f"Cannot reach VORq engine at {API} — run: uvicorn vorq.api.main:app --port 8000"}
    except Exception as e:
        return {"error":str(e)}

# ── CSS — Apple-quality smooth transitions & animations ──────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700;800&display=swap');

/* ─── BASE RESET ─────────────────────────────────────────────────────────── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body,[class*="css"]{{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif!important;-webkit-font-smoothing:antialiased!important;-moz-osx-font-smoothing:grayscale!important;text-rendering:optimizeLegibility!important}}

/* ─── KEYFRAMES — Apple easing everywhere ────────────────────────────────── */
@keyframes fadeInUp{{
  from{{opacity:0;transform:translateY(18px)}}
  to{{opacity:1;transform:translateY(0)}}
}}
@keyframes fadeIn{{
  from{{opacity:0}}
  to{{opacity:1}}
}}
@keyframes slideInRight{{
  from{{opacity:0;transform:translateX(-12px)}}
  to{{opacity:1;transform:translateX(0)}}
}}
@keyframes cardReveal{{
  from{{opacity:0;transform:translateY(10px)}}
  to{{opacity:1;transform:translateY(0)}}
}}
@keyframes blink{{
  0%,100%{{opacity:1;transform:scale(1)}}
  50%{{opacity:.3;transform:scale(.7)}}
}}
@keyframes tdot{{
  0%,80%,100%{{opacity:.15;transform:scaleY(.6)}}
  40%{{opacity:1;transform:scaleY(1)}}
}}
@keyframes pulseGlow{{
  0%,100%{{box-shadow:0 0 0 0 rgba(16,163,127,.22)}}
  50%{{box-shadow:0 0 0 6px rgba(16,163,127,0)}}
}}
@keyframes shimmer{{
  0%{{background-position:200% center}}
  100%{{background-position:-200% center}}
}}

/* ─── APP BACKGROUND ─────────────────────────────────────────────────────── */
.stApp,.main,section[data-testid="stMain"]{{
  background:{BG}!important;
  transition:background .3s ease!important;
}}
.block-container{{
  background:{BG}!important;
  max-width:760px!important;
  margin:0 auto!important;
  padding:2rem 1.5rem 6rem!important;
}}

/* ─── GLOBAL INPUT OVERRIDE ──────────────────────────────────────────────── */
[data-baseweb="input"], [data-baseweb="input"] > div {{
  background-color:{BG_DARK}!important;
  border:1px solid {BORDER}!important;
}}
input {{
  background-color:transparent!important;
  color:{TEXT}!important;
}}

/* ─── HIDE STREAMLIT CHROME ──────────────────────────────────────────────── */
#MainMenu,footer,header,.stDeployButton,
[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],[data-testid="stSidebarCollapseButton"]{{display:none!important}}

/* ─── SCROLLBAR ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar{{width:3px;height:3px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:rgba(16,163,127,.18);border-radius:99px;transition:background .2s}}
::-webkit-scrollbar-thumb:hover{{background:rgba(16,163,127,.35)}}

/* ─── SIDEBAR ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"]{{
  background:{SURFACE}!important;
  border-right:1px solid {BORDER}!important;
  min-width:220px!important;max-width:220px!important;
  animation:slideInRight .35s cubic-bezier(.22,1,.36,1) forwards;
}}
[data-testid="stSidebar"]>div:first-child,[data-testid="stSidebarContent"]{{padding:0!important}}
[data-testid="stSidebar"] .stButton>button{{
  background:transparent!important;border:none!important;border-radius:6px!important;
  color:{MUTED}!important;font-size:.73rem!important;font-weight:500!important;
  padding:.35rem .7rem!important;width:100%!important;text-align:left!important;
  transition:background .15s ease,color .15s ease!important;
}}
[data-testid="stSidebar"] .stButton>button:hover{{
  background:{BG}!important;color:{TEXT}!important;
  transform:translateX(1px)!important;
}}

/* ─── SIDEBAR ELEMENTS ───────────────────────────────────────────────────── */
.sb-logo{{display:flex;align-items:center;gap:.4rem;padding:.9rem 1rem .6rem}}
.sb-mark{{
  width:28px;height:28px;border-radius:6px;overflow:hidden;
  display:flex;align-items:center;justify-content:center;
  transition:transform .2s cubic-bezier(.34,1.56,.64,1);
}}
.sb-mark img{{width:100%;height:100%;object-fit:cover}}
.sb-mark:hover{{transform:scale(1.08) rotate(-2deg)}}
.sb-name{{font-size:.7rem;font-weight:700;letter-spacing:.08em;color:{TEXT}}}
.sb-div{{height:1px;background:{BORDER};margin:.2rem 0}}
.sb-lbl{{font-size:.52rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:{DIM};padding:.6rem 1rem .2rem}}
.sb-item{{
  display:flex;align-items:flex-start;gap:.4rem;padding:.38rem 1rem;
  transition:background .15s ease,border-color .15s ease,transform .15s ease;
  cursor:pointer;border-left:2px solid transparent;
}}
.sb-item.active{{background:rgba(16,163,127,.1);border-left-color:{ACCENT}}}
.sb-item:hover{{background:{BG};transform:translateX(1px)}}
.sb-icon{{font-size:.62rem;color:{DIM};flex-shrink:0;padding-top:1px}}
.sb-title{{font-size:.7rem;color:{MUTED};line-height:1.4}}
.sb-time{{font-size:.52rem;color:{DIM};margin-top:.04rem}}

/* ─── HERO WRAP ──────────────────────────────────────────────────────────── */
.v-hero-wrap{{
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;min-height:92vh;
  padding:0 1.5rem;text-align:center;
  background:{BG};
  margin:-2rem -1.5rem 0;
}}

/* Hero content entrance animations — staggered */
.hero-badge{{
  display:inline-flex;align-items:center;gap:.35rem;
  font-size:.63rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  color:{ACCENT};background:rgba(16,163,127,.1);border:1px solid rgba(16,163,127,.3);
  border-radius:99px;padding:.28rem .72rem;margin-bottom:1.8rem;
  animation:fadeInUp .6s cubic-bezier(.22,1,.36,1) .1s both;
  will-change:transform,opacity;
}}
.hero-badge-dot{{
  width:5px;height:5px;border-radius:50%;background:{ACCENT};
  animation:blink 2.4s ease-in-out infinite;
}}
.hero-h{{
  font-family:'Space Grotesk',sans-serif;font-size:3.2rem;font-weight:800;
  letter-spacing:-.04em;color:{TEXT};line-height:1.1;
  max-width:640px;margin:0 auto .8rem;
  animation:fadeInUp .7s cubic-bezier(.22,1,.36,1) .22s both;
  will-change:transform,opacity;
}}
.hero-h em{{font-style:normal;color:{ACCENT}}}
.hero-sub{{
  font-size:1.05rem;color:{MUTED};line-height:1.7;
  max-width:460px;margin:0 auto 2.2rem;font-weight:400;
  animation:fadeInUp .7s cubic-bezier(.22,1,.36,1) .36s both;
  will-change:transform,opacity;
}}

/* ─── HERO INPUT ─────────────────────────────────────────────────────────── */
.v-hero-input-row{{
  max-width:680px;margin:0 auto;width:100%;
  animation:fadeInUp .7s cubic-bezier(.22,1,.36,1) .48s both;
  will-change:transform,opacity;
}}
.v-hero-input-row [data-baseweb="input"]{{
  background:{BG_DARK}!important;border:1.5px solid {BORDER}!important;
  border-radius:12px 0 0 12px!important;
  height:52px!important;
  box-shadow:0 2px 8px rgba(0,0,0,.15)!important;
}}
.v-hero-input-row [data-baseweb="input"] > div{{background:transparent!important}}
.v-hero-input-row [data-baseweb="input"]:focus-within{{
  border-color:{ACCENT}!important;
  box-shadow:0 0 0 3px rgba(16,163,127,.18),
             0 4px 20px rgba(16,163,127,.1),
             0 2px 8px rgba(0,0,0,.1)!important;
  animation:pulseGlow .8s ease-out!important;
}}
.v-hero-input-row input{{
  font-size:.95rem!important;color:{TEXT}!important;padding-left:1rem!important;
  letter-spacing:-.01em!important;background:transparent!important;
}}
.v-hero-input-row input::placeholder{{color:{DIM}!important}}
.v-hero-input-row .stButton button{{
  background:{ACCENT}!important;color:#ffffff!important;border:none!important;
  border-radius:0 12px 12px 0!important;font-weight:700!important;
  font-size:.82rem!important;height:52px!important;padding:0 1.3rem!important;
  transition:all .2s cubic-bezier(.4,0,.2,1)!important;
}}
.v-hero-input-row .stButton button:hover{{
  background:{ACCENT2}!important;
  box-shadow:0 0 20px rgba(16,163,127,.25)!important;
  transform:translateY(-1px)!important;
}}

/* ─── CHIP BUTTONS ──────────────────────────────────────────────────────── */
div.v-chip-rows{{
  max-width:580px;margin:1.4rem auto 0;
  animation:fadeInUp .7s cubic-bezier(.22,1,.36,1) .6s both;
  will-change:transform,opacity;
}}
div.v-chip-rows .stButton button{{
  font-size:.72rem!important;font-weight:500!important;
  padding:.4rem 1rem!important;border-radius:99px!important;
  background:{BG_DARK}!important;border:1.5px solid {BORDER}!important;
  color:{MUTED}!important;
  transition:all .2s cubic-bezier(.4,0,.2,1)!important;
}}
div.v-chip-rows .stButton button:hover{{
  background:{SURFACE}!important;border-color:{ACCENT}!important;
  color:{TEXT}!important;transform:translateY(-2px)!important;
}}

/* ─── NAV ────────────────────────────────────────────────────────────────── */
.v-nav{{
  display:flex;align-items:center;justify-content:space-between;
  padding:.55rem 1.5rem;background:rgba(43,38,35,.88);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid {BORDER};
  position:sticky;top:0;z-index:200;
  animation:fadeIn .4s ease both;
}}
.v-nav-l{{display:flex;align-items:center;gap:.4rem}}
.v-nav-mark{{
  width:26px;height:26px;border-radius:6px;overflow:hidden;
  display:flex;align-items:center;justify-content:center;
  transition:transform .2s cubic-bezier(.34,1.56,.64,1);
}}
.v-nav-mark img{{width:100%;height:100%;object-fit:cover}}
.v-nav-mark:hover{{transform:scale(1.1)}}
.v-nav-name{{font-size:.7rem;font-weight:700;letter-spacing:.08em;color:{TEXT}}}
.v-nav-badge{{
  display:inline-flex;align-items:center;gap:.3rem;font-size:.57rem;
  font-weight:600;letter-spacing:.07em;text-transform:uppercase;
  color:{GREEN};background:#F0FDF4;border:1px solid #BBF7D0;
  padding:.22rem .55rem;border-radius:99px;
  animation:fadeIn .5s ease .3s both;
}}
.v-nav-dot{{
  width:5px;height:5px;border-radius:50%;background:{GREEN};
  animation:blink 2s infinite;
}}

/* ─── RESULTS STREAM ─────────────────────────────────────────────────────── */
.v-stream{{max-width:760px;margin:0 auto;padding:1.5rem 1.5rem 9rem}}

/* ─── SCENARIO BAR ───────────────────────────────────────────────────────── */
.v-scenario{{
  display:flex;align-items:center;gap:.55rem;
  padding:.55rem .9rem;margin-bottom:1.1rem;
  background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
  animation:cardReveal .4s cubic-bezier(.22,1,.36,1) both;
  box-shadow:0 1px 3px rgba(0,0,0,.04);
}}
.v-scenario-lbl{{font-size:.54rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:{ACCENT};flex-shrink:0}}
.v-scenario-txt{{font-size:.86rem;color:{TEXT};font-weight:400}}

/* ─── CARDS ──────────────────────────────────────────────────────────────── */
.v-card{{
  background:{SURFACE};border:1.5px solid {BORDER};border-radius:12px;
  margin-bottom:.75rem;overflow:hidden;
  box-shadow:0 1px 4px rgba(0,0,0,.05),0 0 0 0 rgba(37,99,235,0);
  animation:cardReveal .5s cubic-bezier(.22,1,.36,1) both;
  transition:box-shadow .25s ease,transform .2s cubic-bezier(.34,1.56,.64,1),
             border-color .2s ease!important;
  will-change:transform;
}}
.v-card:hover{{
  box-shadow:0 4px 16px rgba(0,0,0,.08),0 1px 4px rgba(0,0,0,.04)!important;
  transform:translateY(-1px)!important;
  border-color:rgba(16,163,127,.18)!important;
}}
.v-card-hdr{{
  display:flex;align-items:center;gap:.45rem;
  padding:.48rem 1rem;border-bottom:1px solid {BORDER};background:{BG};
}}
.v-card-hdr-line{{width:14px;height:2px;border-radius:1px;flex-shrink:0}}
.v-card-title{{font-size:.56rem;font-weight:700;text-transform:uppercase;letter-spacing:.14em;color:{MUTED}}}
.v-card-body{{padding:.9rem 1rem 1rem}}

/* ─── LEAD CARD ──────────────────────────────────────────────────────────── */
.v-lead{{
  background:{SURFACE};border:1.5px solid;border-left-width:3px;
  border-radius:12px;padding:1.2rem;margin-bottom:.75rem;
  box-shadow:0 1px 4px rgba(0,0,0,.05);
  animation:cardReveal .5s cubic-bezier(.22,1,.36,1) .08s both;
  transition:transform .2s cubic-bezier(.34,1.56,.64,1),
             box-shadow .25s ease!important;
  will-change:transform;
}}
.v-lead:hover{{
  transform:translateY(-1px)!important;
  box-shadow:0 6px 20px rgba(0,0,0,.07)!important;
}}
.v-lead-lbl{{
  font-size:.54rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.14em;color:{DIM};margin-bottom:.55rem;
  display:flex;align-items:center;gap:.4rem;
}}
.v-lead-lbl::before{{
  content:'';width:14px;height:2px;border-radius:1px;
  background:{ACCENT};display:inline-block;
}}
.v-lead-txt{{font-size:1.08rem;font-weight:400;color:{TEXT};line-height:1.75;letter-spacing:-.01em}}
.v-lead-txt strong{{font-weight:600;color:{TEXT}}}
.v-lead-meta{{
  display:flex;gap:1.2rem;margin-top:.9rem;padding-top:.7rem;
  border-top:1px solid {BORDER};flex-wrap:wrap;
}}
.v-lead-mi{{font-size:.63rem;color:{DIM}}}
.v-lead-mv{{color:{MUTED};font-weight:600}}

/* ─── SCORE CARD ─────────────────────────────────────────────────────────── */
.v-score{{
  display:flex;align-items:center;gap:1.1rem;
  background:{SURFACE};border:1.5px solid {BORDER};border-radius:12px;
  padding:1rem 1.2rem;margin-bottom:.75rem;
  box-shadow:0 1px 4px rgba(0,0,0,.05);
  animation:cardReveal .5s cubic-bezier(.22,1,.36,1) .16s both;
  transition:transform .2s cubic-bezier(.34,1.56,.64,1),
             box-shadow .25s ease!important;
}}
.v-score:hover{{
  transform:translateY(-1px)!important;
  box-shadow:0 6px 20px rgba(0,0,0,.07)!important;
}}
.v-gauge{{position:relative;width:80px;height:52px;flex-shrink:0}}
.v-gauge svg{{width:80px;height:52px}}
.v-gauge-num{{
  position:absolute;bottom:0;left:50%;transform:translateX(-50%);
  font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:700;
  letter-spacing:-.04em;
  animation:fadeIn .6s ease .4s both;
}}
.v-gauge-of{{font-size:.5rem;color:{DIM};font-weight:400;margin-left:1px}}
.v-score-body{{flex:1}}
.v-score-lv{{font-size:.56rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.2rem}}
.v-score-ctx{{font-size:.74rem;color:{MUTED};margin-bottom:.35rem}}
.v-score-pills{{display:flex;gap:.25rem;flex-wrap:wrap}}
.v-spill{{
  font-size:.54rem;font-weight:500;padding:.12rem .38rem;border-radius:4px;
  background:{BG};border:1px solid {BORDER};color:{MUTED};
  transition:background .15s,border-color .15s;
}}

/* ─── WHY SECTION ────────────────────────────────────────────────────────── */
.v-reason{{
  display:flex;gap:.55rem;align-items:flex-start;
  padding:.5rem 0;border-bottom:1px solid #2E3240;
  animation:fadeInUp .45s cubic-bezier(.22,1,.36,1) both;
  transition:background .15s ease;
}}
.v-reason:last-child{{border-bottom:none}}
.v-reason:hover{{background:#2E3240}}
.v-rdot{{width:6px;height:6px;border-radius:50%;flex-shrink:0;margin-top:.46rem}}
.v-rbody{{flex:1}}
.v-rt{{font-size:.8rem;font-weight:500;color:{TEXT};display:flex;align-items:center;gap:.35rem;margin-bottom:.04rem;flex-wrap:wrap}}
.v-rd{{font-size:.7rem;color:{MUTED};line-height:1.7}}
.v-rtag{{font-size:.52rem;font-weight:600;padding:.1rem .34rem;border-radius:3px;border:1px solid;letter-spacing:.02em}}

/* ─── SECTOR CHARTS ──────────────────────────────────────────────────────── */
.v-fc-n{{font-size:.54rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:{DIM};margin-bottom:.06rem}}
.v-fc-p{{font-size:1rem;font-weight:600;letter-spacing:-.02em;line-height:1.2}}
.v-fc-s{{font-size:.58rem;color:{DIM};margin-top:.04rem}}

/* ─── COMPANY EXPOSURE ───────────────────────────────────────────────────── */
.v-co{{
  display:flex;align-items:center;gap:.6rem;padding:.45rem 0;
  border-bottom:1px solid {BORDER};
  transition:background .15s ease;
}}
.v-co:last-child{{border-bottom:none}}
.v-co:hover{{background:{SURF2}}}
.v-co-name{{flex:0 0 130px;font-size:.78rem;color:{TEXT};font-weight:500}}
.v-co-tick{{font-size:.55rem;color:{DIM}}}
.v-co-bar{{flex:1;height:3px;background:{BORDER};border-radius:99px;overflow:hidden}}
.v-co-fill{{
  height:100%;border-radius:99px;
  transition:width .7s cubic-bezier(.22,1,.36,1) .2s!important;
  will-change:width;
}}
.v-co-pct{{flex:0 0 48px;text-align:right;font-size:.78rem;font-weight:600;font-variant-numeric:tabular-nums}}

/* ─── ACTIONS ────────────────────────────────────────────────────────────── */
.v-act{{
  display:flex;gap:.55rem;padding:.45rem 0;border-bottom:1px solid {BORDER};
  align-items:flex-start;
  transition:background .15s ease;
}}
.v-act:last-child{{border-bottom:none}}
.v-act:hover{{background:{SURF2}}}
.v-act-n{{flex:0 0 18px;font-size:.65rem;color:{DIM};font-weight:700;text-align:right;padding-top:2px}}
.v-act-t{{font-size:.76rem;color:{MUTED};line-height:1.7}}

/* ─── CONFIDENCE ─────────────────────────────────────────────────────────── */
.v-conf{{display:flex;gap:1.2rem;padding:.55rem 0 0;border-top:1px solid {BORDER};margin-top:.5rem;flex-wrap:wrap}}
.v-conf-i{{font-size:.6rem;color:{DIM}}}
.v-conf-v{{color:{MUTED};font-weight:600}}
.v-conf-note{{font-size:.58rem;color:{DIM};margin-top:.35rem;line-height:1.6;font-style:italic}}

/* ─── MAP LEGEND ─────────────────────────────────────────────────────────── */
.v-legend{{display:flex;gap:.85rem;padding:.2rem 1rem .55rem;font-size:.57rem;color:{DIM}}}

/* ─── ERROR CARD ─────────────────────────────────────────────────────────── */
.v-err{{
  background:rgba(248,113,113,.1);border-left:3px solid {RED};
  padding:.7rem 1rem;border-radius:0 7px 7px 0;
  font-size:.78rem;color:{RED};margin:.8rem 0;
  animation:fadeInUp .4s ease both;
}}

/* ─── BOTTOM FIXED INPUT ─────────────────────────────────────────────────── */
.v-bottom{{
  position:fixed;bottom:0;left:220px;right:0;
  padding:.7rem 1.5rem .9rem;
  background:linear-gradient(to top,{BG} 70%,rgba(43,38,35,0));
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  z-index:300;
}}
.v-bot-inner{{max-width:760px;margin:0 auto}}
.v-bot-note{{font-size:.53rem;color:{DIM};text-align:center;margin-top:.18rem}}
.v-bottom .stTextInput [data-baseweb="input"]{{
  background:{BG_DARK}!important;border:1.5px solid {BORDER}!important;
  border-radius:9px 0 0 9px!important;height:44px!important;
}}
.v-bottom .stTextInput [data-baseweb="input"] > div{{background:transparent!important}}
.v-bottom .stTextInput [data-baseweb="input"]:focus-within{{
  border-color:{ACCENT}!important;
  box-shadow:0 0 0 3px rgba(16,163,127,.12),0 2px 10px rgba(0,0,0,.06)!important;
}}
.v-bottom .stTextInput input{{font-size:.84rem!important;color:{TEXT}!important;background:transparent!important}}
.v-bottom .stTextInput input::placeholder{{color:{DIM}!important}}
.v-bot-send button{{
  background:{ACCENT}!important;color:white!important;border:none!important;
  border-radius:0 9px 9px 0!important;font-weight:700!important;
  font-size:.76rem!important;height:44px!important;
  transition:background .2s ease,transform .15s cubic-bezier(.34,1.56,.64,1)!important;
}}
.v-bot-send button:hover{{
  background:{ACCENT2}!important;
  transform:scale(1.03)!important;
}}
.v-bot-send button:active{{transform:scale(.96)!important}}

/* ─── THINKING LOADER ────────────────────────────────────────────────────── */
.thinking{{
  display:flex;align-items:center;gap:.55rem;
  padding:5rem 0;justify-content:center;
  animation:fadeIn .3s ease both;
}}
.t-dot{{
  width:5px;height:16px;border-radius:99px;
  background:{ACCENT};
  animation:tdot .85s ease-in-out infinite;
  will-change:transform,opacity;
}}
.t-dot:nth-child(2){{animation-delay:.12s;height:22px}}
.t-dot:nth-child(3){{animation-delay:.24s;height:12px}}
.t-txt{{font-size:.83rem;color:{MUTED};margin-left:.3rem;letter-spacing:-.01em}}

/* ─── GLOBAL BUTTON DEFAULTS ─────────────────────────────────────────────── */
.stButton button{{
  background-color:{BG_DARK}!important;
  color:{MUTED}!important;
  border:1.5px solid {BORDER}!important;
  border-radius:9px!important;
  font-family:'Inter',sans-serif!important;
  transition:all .2s cubic-bezier(.4,0,.2,1)!important;
}}
.stButton button:hover{{
  background-color:{SURFACE}!important;
  color:{TEXT}!important;
  border-color:{ACCENT}!important;
}}
.stButton button:focus{{box-shadow:none!important;outline:none!important}}

/* ─── SPECIFIC HERO & BOTTOM BUTTONS ─────────────────────────────────────── */
.stButton button[kind="primary"]{{
  background-color:{ACCENT}!important;
  color:#ffffff!important;
  border:none!important;
  font-weight:700!important;
}}
.stButton button[kind="primary"]:hover{{
  background-color:{ACCENT2}!important;
  box-shadow:0 0 20px rgba(16,163,127,.25)!important;
  transform:translateY(-1px)!important;
}}

/* ─── NEW SIMULATION BUTTON (sidebar) ────────────────────────────────────── */
[data-testid="stSidebar"] .stButton>button[kind="secondary"]{{
  border:1px solid {BORDER}!important;
  border-radius:8px!important;
  padding:.4rem .8rem!important;
  transition:background .18s,border-color .18s,transform .18s cubic-bezier(.34,1.56,.64,1)!important;
}}
[data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover{{
  border-color:{ACCENT}!important;
  color:{ACCENT}!important;
  background:{SURFACE}!important;
  transform:translateY(-1px)!important;
}}

/* ─── RESPONSIVE ─────────────────────────────────────────────────────────── */
@media (max-width: 900px){{
  .block-container{{max-width:100%!important;padding:1rem .8rem 7rem!important}}
  .v-stream{{max-width:100%!important;padding:1rem .8rem 8rem!important}}
  .v-bottom{{left:0!important;right:0!important;padding:.7rem .8rem 1rem!important}}
  .v-hero-wrap{{min-height:85vh;padding:0 1rem}}
  .hero-h{{font-size:2.15rem!important}}
  .hero-sub{{font-size:.95rem!important;line-height:1.6!important}}
  .v-score{{flex-direction:column;align-items:flex-start;gap:.6rem}}
  .v-gauge{{width:70px}}
  [data-testid="stSidebar"]{{display:none!important}}
}}
</style>
""", unsafe_allow_html=True)

# ── Chip click ───────────────────────────────────────────────────────────────
if st.session_state.chip_query:
    st.session_state.pending = st.session_state.chip_query
    st.session_state.chip_query = None
    st.rerun()

# ── Run simulation ───────────────────────────────────────────────────────────
if st.session_state.pending:
    query = st.session_state.pending
    st.session_state.pending = None

    ph = st.empty()
    steps = [
        "Parsing geopolitical entities and signals",
        "Running causal propagation across 14 data layers",
        "Generating intelligence assessment",
    ]
    for txt, dl in zip(steps, [0.5, 0.8, 0.5]):
        ph.markdown(f"""
        <div class='thinking'>
          <div class='t-dot'></div><div class='t-dot'></div><div class='t-dot'></div>
          <span class='t-txt'>{txt}…</span>
        </div>""", unsafe_allow_html=True)
        time.sleep(dl)
    ph.empty()

    result = _simulate(query)
    eid = f"q{len(st.session_state.conversation)}"
    st.session_state.conversation.append(
        {"id":eid,"query":query,"result":result,"ts":datetime.now().strftime("%H:%M")})
    st.session_state.active_id = eid
    st.rerun()

has_results = bool(st.session_state.conversation)

# ════════════════════════════════════════════════════════════════════════════
# STATE 1 — LANDING
# ════════════════════════════════════════════════════════════════════════════
if not has_results:
    # Hero text — pure HTML, no widgets inside
    st.markdown(f"""
    <div class='v-hero-wrap'>
      <div class='hero-badge'><span class='hero-badge-dot'></span>Geopolitical Intelligence Platform</div>
      <div class='hero-h'>See the risk <em>before</em> the world does.</div>
      <div class='hero-sub'>Type any global scenario. VORq simulates sector impacts, causal chains, who gets hit, what to do — instantly.</div>
    </div>""", unsafe_allow_html=True)

    # Streamlit input row — outside hero div, below hero text
    # CSS class applied via wrapper
    st.markdown("<div class='v-hero-input-row'>", unsafe_allow_html=True)
    c1, c2 = st.columns([7, 2], gap="small")
    with c1:
        q = st.text_input("q", "", placeholder="What happens if China invades Taiwan in 2025?",
                          label_visibility="collapsed", key="hero_input")
    with c2:
        send = st.button("Simulate →", key="hero_send", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    # Scenario chips — 3+2 layout (centered, wrapping)
    # Row 1: 3 chips
    chip_row1 = CHIP_META[:3]
    chip_row2 = CHIP_META[3:]
    st.markdown("<div class='v-chip-rows'>", unsafe_allow_html=True)
    for row in [chip_row1, chip_row2]:
        cols = st.columns(len(row))
        for i, (lbl, col, _) in enumerate(row):
            with cols[i]:
                if st.button(lbl, key=f"chip_{CHIP_META.index((lbl, col, _))}", use_container_width=True):
                    st.session_state.pending = lbl; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if send and q.strip():
        st.session_state.pending = q.strip(); st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STATE 2 — RESULTS
# ════════════════════════════════════════════════════════════════════════════
else:
    # SIDEBAR
    with st.sidebar:
        st.markdown(f"""
        <div class='sb-logo'><div class='sb-mark'><img src='data:image/png;base64,{_LOGO}'/></div><span class='sb-name'>VORq</span></div>
        <div class='sb-div'></div>""", unsafe_allow_html=True)
        if st.button("+ New simulation",key="new_btn",use_container_width=True):
            st.session_state.conversation=[]; st.session_state.active_id=None; st.rerun()
        conv = st.session_state.conversation
        if conv:
            st.markdown("<div class='sb-lbl'>Recents</div>", unsafe_allow_html=True)
            for item in reversed(conv[-8:]):
                ac = " active" if item["id"]==st.session_state.active_id else ""
                r  = item.get("result",{})
                et = r.get("event",{}).get("type","supply_shock") if "error" not in r else "supply_shock"
                st.markdown(f"""
                <div class='sb-item{ac}'>
                  <span class='sb-icon'>{ICONS.get(et,"◈")}</span>
                  <div><div class='sb-title'>{_short(item["query"])}</div>
                  <div class='sb-time'>{item.get("ts","")}</div></div>
                </div>""", unsafe_allow_html=True)

    # NAVBAR
    st.markdown(f"""
    <div class='v-nav'>
      <div class='v-nav-l'><div class='v-nav-mark'><img src='data:image/png;base64,{_LOGO}'/></div><span class='v-nav-name'>VORq</span></div>
      <span class='v-nav-badge'><span class='v-nav-dot'></span>Intelligence Active</span>
    </div>""", unsafe_allow_html=True)

    active = next((x for x in st.session_state.conversation if x["id"]==st.session_state.active_id),
                  st.session_state.conversation[-1])
    result = active["result"]

    st.markdown("<div class='v-stream'>", unsafe_allow_html=True)

    # Scenario header
    st.markdown(f"""
    <div class='v-scenario'>
      <span class='v-scenario-lbl'>Scenario</span>
      <span class='v-scenario-txt'>{active["query"]}</span>
    </div>""", unsafe_allow_html=True)

    if "error" in result:
        st.markdown(f"<div class='v-err'>{result['error']}</div>", unsafe_allow_html=True)
    else:
        evt  = result.get("event",{})
        sim  = result.get("simulation",{})
        expl = result.get("explanation",{})
        v3   = result.get("v3", {})
        tree = v3.get("scenario_tree", {})
        risk = v3.get("risk_analytics", {})
        macro= v3.get("macro_context", {})
        ci   = sim.get("company_impacts",[])
        ii   = sim.get("industry_impacts",[])
        tl   = sim.get("timeline",{})
        rl   = sim.get("risk_level","MODERATE")
        rs   = sim.get("overall_risk_score",0)
        sa   = expl.get("sector_analysis",[])
        mits = expl.get("mitigation_suggestions",[])
        conf = expl.get("confidence_assessment",{})
        tree_stats = tree.get("summary_stats", {})
        tree_branches = tree.get("branches", [])
        dist_stats = risk.get("distribution", {})
        pdf_bins = risk.get("pdf_bins", [])
        ctrs = evt.get("entities",{}).get("countries",[])
        inds = evt.get("entities",{}).get("industries",[])
        sev  = evt.get("severity",0)
        etype= evt.get("type","supply_shock")
        pk_d = tl.get("time_to_peak_days","?")
        rec_m= tl.get("recovery_months","?")
        rc   = _rc(rl)
        pctl = PCTL.get(rl,"50")
        top_s= sa[0] if sa else None

        # ── INTELLIGENCE BRIEF ────────────────────────────────────────────
        parts=[]
        if top_s: parts.append(f"<strong>{top_s['sector']}</strong> faces a <strong>{abs(top_s['impact_pct']):.0f}% {top_s['direction']}</strong> impact")
        if ctrs:  parts.append(f"with primary exposure in <strong>{' and '.join(c['name'] for c in ctrs[:2])}</strong>")
        parts.append(f"Peak disruption in <strong>{pk_d} days</strong>, recovery ~<strong>{rec_m} months</strong>.")
        lead = " — ".join(parts[:2])+". "+parts[-1] if len(parts)>2 else ". ".join(parts)

        st.markdown(f"""
        <div class='v-lead' style='border-color:{rc}30;border-left-color:{rc}'>
          <div class='v-lead-lbl'>Intelligence Brief</div>
          <div class='v-lead-txt'>{lead}</div>
          <div class='v-lead-meta'>
            <span class='v-lead-mi'>Event <span class='v-lead-mv'>{etype.replace("_"," ").title()}</span></span>
            <span class='v-lead-mi'>Severity <span class='v-lead-mv'>{sev:.0%}</span></span>
            <span class='v-lead-mi'>Sectors <span class='v-lead-mv'>{len(ii)}</span></span>
            <span class='v-lead-mi'>Companies <span class='v-lead-mv'>{len(ci)}</span></span>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── RISK SCORE + GAUGE ────────────────────────────────────────────
        angle=min(rs/100,1.0)*180; rg=32; cxg=40; cyg=44
        x1=cxg-rg; x2t=cxg+rg
        er=math.radians(180-angle)
        x2f=cxg+rg*math.cos(er); y2f=cyg-rg*math.sin(er)
        lf=1 if angle>90 else 0
        gauge_svg=f"""<svg viewBox="0 0 80 48" style="width:80px;height:48px">
          <path d="M{x1},{cyg} A{rg},{rg} 0 1,1 {x2t},{cyg}" fill="none" stroke="#E5E7EB" stroke-width="4.5" stroke-linecap="round"/>
          <path d="M{x1},{cyg} A{rg},{rg} 0 {lf},1 {x2f:.1f},{y2f:.1f}" fill="none" stroke="{rc}" stroke-width="4.5" stroke-linecap="round"/></svg>"""

        pills="".join(f"<span class='v-spill'>{c['name']}</span>" for c in ctrs[:3])
        pills+="".join(f"<span class='v-spill'>{i['label']}</span>" for i in inds[:2])

        st.markdown(f"""
        <div class='v-score'>
          <div class='v-gauge'>{gauge_svg}<div class='v-gauge-num' style='color:{rc}'>{rs:.0f}<span class='v-gauge-of'>/ 100</span></div></div>
          <div class='v-score-body'>
            <div class='v-score-lv' style='color:{rc}'>{rl}</div>
            <div class='v-score-ctx'>Higher than {pctl}% of simulated scenarios</div>
            <div class='v-score-pills'>{pills}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── PROBABILISTIC OUTLOOK ────────────────────────────────────────
        exp_sev = float(tree_stats.get("expected_severity", 0.0))
        p_panic = float(tree_stats.get("p_panic", 0.0))
        p_global = float(tree_stats.get("p_global_contagion", 0.0))
        p_severe = float(tree_stats.get("p_severe_or_worse", 0.0))
        var_95 = float(risk.get("var_95", rs))
        cvar_95 = float(risk.get("cvar_95", var_95))
        regime_name = str(macro.get("regime", "unknown")).replace("_", " ").title()
        regime_conf = float(macro.get("regime_confidence", 0.0))

        st.markdown(f"""
        <div class='v-card'>
          <div class='v-card-hdr'><div class='v-card-hdr-line' style='background:{ACCENT}'></div><span class='v-card-title'>Probabilistic Outlook</span></div>
          <div class='v-card-body'>
            <div class='v-conf'>
              <span class='v-conf-i'>Expected severity <span class='v-conf-v'>{exp_sev:.0%}</span></span>
              <span class='v-conf-i'>P(Panic regime) <span class='v-conf-v'>{p_panic:.0%}</span></span>
              <span class='v-conf-i'>P(Global contagion) <span class='v-conf-v'>{p_global:.0%}</span></span>
              <span class='v-conf-i'>P(Severe+) <span class='v-conf-v'>{p_severe:.0%}</span></span>
              <span class='v-conf-i'>VaR 95 <span class='v-conf-v'>{var_95:.1f}</span></span>
              <span class='v-conf-i'>CVaR 95 <span class='v-conf-v'>{cvar_95:.1f}</span></span>
              <span class='v-conf-i'>Macro regime <span class='v-conf-v'>{regime_name} ({regime_conf:.0%})</span></span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        if tree_branches:
            branch_rows = ""
            for b in tree_branches[:5]:
                prob_pct = max(0.0, min(100.0, float(b.get("probability", 0.0)) * 100.0))
                sev_score = float(b.get("severity_score", 0.0))
                tone = RED if sev_score > 0.8 else ORANGE if sev_score > 0.65 else YELLOW if sev_score > 0.45 else GREEN
                scenario = str(b.get("scenario", "")).replace("_", " ").title()
                branch_rows += f"""
                <div style='padding:.42rem 0;border-bottom:1px solid {BORDER};'>
                  <div style='display:flex;justify-content:space-between;gap:.8rem;align-items:center;margin-bottom:.2rem;'>
                    <span style='font-size:.74rem;color:{TEXT};font-weight:600;'>{scenario}</span>
                    <span style='font-size:.66rem;color:{MUTED};'>{prob_pct:.1f}%</span>
                  </div>
                  <div style='height:4px;background:{BORDER};border-radius:99px;overflow:hidden;'>
                    <div style='height:100%;width:{prob_pct:.1f}%;background:{tone};border-radius:99px;'></div>
                  </div>
                </div>"""
            st.markdown(f"""
            <div class='v-card'>
              <div class='v-card-hdr'><div class='v-card-hdr-line' style='background:{ACCENT}'></div><span class='v-card-title'>Top Scenario Branches</span></div>
              <div class='v-card-body'>{branch_rows}</div>
            </div>""", unsafe_allow_html=True)

        if pdf_bins:
            x_vals = [float(b.get("x", 0)) for b in pdf_bins]
            y_vals = [float(b.get("density", 0)) for b in pdf_bins]
            fig_dist = go.Figure(go.Bar(
                x=x_vals,
                y=y_vals,
                marker=dict(color=ACCENT, opacity=0.8),
                hovertemplate="Risk %{x:.1f}: %{y:.3f}<extra></extra>",
            ))
            if dist_stats.get("mean") is not None:
                fig_dist.add_vline(
                    x=float(dist_stats.get("mean", rs)),
                    line_width=1.4,
                    line_color=ORANGE,
                    line_dash="dash",
                )
            fig_dist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=180,
                margin=dict(l=0, r=0, t=6, b=0),
                showlegend=False,
                xaxis=dict(showgrid=False, tickfont=dict(color=MUTED, size=10), title=""),
                yaxis=dict(showgrid=False, showticklabels=False, title=""),
            )
            st.markdown(f"""
            <div class='v-card'>
              <div class='v-card-hdr'><div class='v-card-hdr-line' style='background:{ACCENT}'></div><span class='v-card-title'>Risk Distribution</span></div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── WHY THIS HAPPENS ──────────────────────────────────────────────
        rows=""
        for s in sa[:5]:
            sl=_sl(s["impact_pct"]); sc=_rc(sl)
            desc=CAUSAL.get(s["sector"],f"{s['impact_pct']:+.1f}% — cascading disruption across interconnected trade and supply networks.")
            rows+=f"""<div class='v-reason'>
              <div class='v-rdot' style='background:{sc}'></div>
              <div class='v-rbody'>
                <div class='v-rt'>{s['sector']} <span class='v-rtag' style='color:{sc};border-color:{sc}33;background:{sc}0D'>{sl}</span> <span style='color:{sc};font-weight:600;font-size:.76rem'>{s['impact_pct']:+.1f}%</span></div>
                <div class='v-rd'>{desc}</div>
              </div></div>"""
        st.markdown(f"""
        <div class='v-card'>
          <div class='v-card-hdr'><div class='v-card-hdr-line' style='background:{ACCENT}'></div><span class='v-card-title'>Why This Happens</span></div>
          <div class='v-card-body'>{rows}</div>
        </div>""", unsafe_allow_html=True)

        # ── SECTOR OUTLOOK ────────────────────────────────────────────────
        top_sec=sorted(ii,key=lambda x:abs(x["impact_pct"]),reverse=True)[:3]
        if top_sec:
            st.markdown("""<div class='v-card'><div class='v-card-hdr'><div class='v-card-hdr-line' style='background:"""+ACCENT+"""'></div><span class='v-card-title'>Who Gets Hit · Sector Outlook</span></div>""",unsafe_allow_html=True)
            cols3=st.columns(len(top_sec))
            for idx,sector in enumerate(top_sec):
                with cols3[idx]:
                    base=sector["impact_pct"]
                    pk=max(2,tl.get("time_to_peak_days",60)//30)
                    rm=max(1,tl.get("recovery_months",12))
                    vals=[]
                    for m in range(1,25):
                        if m<=pk:
                            v=base*((m/pk)**0.7)
                        else:
                            v=base*(1-min(1.0,(m-pk)/(rm*1.5))*0.2)
                        vals.append(round(v+random.uniform(-0.4,0.4),2))
                    sl=_sl(base); ch=_rc(sl)
                    ri,gi,bi=int(ch[1:3],16),int(ch[3:5],16),int(ch[5:7],16)
                    fig=go.Figure(go.Scatter(
                        x=list(range(1,25)),y=vals,mode="lines",
                        line=dict(color=ch,width=1.5,shape="spline"),
                        fill="tozeroy",fillcolor=f"rgba({ri},{gi},{bi},.08)",
                        hovertemplate="Month %{x}: %{y:+.1f}%<extra></extra>"))
                    fig.add_hline(y=0,line_color="#E5E7EB",line_width=1)
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                        height=90,margin=dict(l=0,r=0,t=2,b=0),showlegend=False,
                        xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
                        yaxis=dict(showgrid=False,showticklabels=False,zeroline=False))
                    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
                    ll={"CRITICAL":"Severe","HIGH":"High Risk","ELEVATED":"Elevated","MODERATE":"Stable","LOW":"Positive"}.get(sl,sl.title())
                    st.markdown(f"""<div style='padding:0 .1rem .7rem'>
                      <div class='v-fc-n'>{sector["label"]}</div>
                      <div class='v-fc-p' style='color:{ch}'>{base:+.1f}%</div>
                      <div class='v-fc-s'>24-mo · {ll}</div></div>""",unsafe_allow_html=True)
            st.markdown("</div>",unsafe_allow_html=True)

        # ── MAP ───────────────────────────────────────────────────────────
        country_impacts=sim.get("country_impacts",[])
        LATS={"US":37.09,"CN":35.86,"IN":20.59,"JP":36.2,"DE":51.16,"GB":55.37,"FR":46.2,
              "RU":61.5,"BR":-14.2,"AU":-25.7,"KR":35.9,"TW":23.7,"SA":23.9,"CA":56.1,
              "MX":23.6,"NL":52.1,"PL":51.9,"AE":23.4}
        LONS={"US":-95.7,"CN":104.2,"IN":78.9,"JP":138.25,"DE":10.45,"GB":-3.44,"FR":2.21,
              "RU":105.3,"BR":-51.9,"AU":133.8,"KR":127.8,"TW":121.0,"SA":45.0,"CA":-106.3,
              "MX":-102.5,"NL":5.29,"PL":19.14,"AE":53.8}
        if country_impacts:
            lats,lons,texts,mcols,sizes=[],[],[],[],[]
            for c in country_impacts:
                cid=c["id"]
                if cid in LATS:
                    imp=c["impact_pct"]
                    col=RED if imp<-10 else ORANGE if imp<-5 else YELLOW if imp<0 else GREEN
                    lats.append(LATS[cid]);lons.append(LONS[cid])
                    texts.append(f"<b>{c['name']}</b><br>{imp:+.1f}% impact")
                    mcols.append(col);sizes.append(max(9,min(22,abs(imp)*1.5)))
            st.markdown(f"""<div class='v-card'>
              <div class='v-card-hdr'><div class='v-card-hdr-line' style='background:{ACCENT}'></div>
              <span class='v-card-title'>What to Watch · Geographic Exposure</span></div>""",unsafe_allow_html=True)
            fig_map=go.Figure(go.Scattergeo(lat=lats,lon=lons,text=texts,mode="markers",
                marker=dict(size=sizes,color=mcols,opacity=.85,line=dict(width=.5,color="white")),
                hovertemplate="%{text}<extra></extra>"))
            fig_map.update_geos(showland=True,landcolor="#F8F9FA",showocean=True,oceancolor="#EFF6FF",
                showcoastlines=True,coastlinecolor="#E4E7EC",showcountries=True,countrycolor="#E4E7EC",
                showframe=False,bgcolor="rgba(0,0,0,0)",projection_type="natural earth")
            fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=260,
                margin=dict(l=0,r=0,t=5,b=0),showlegend=False,geo=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_map,use_container_width=True,config={"displayModeBar":False})
            st.markdown(f"""<div class='v-legend'>
              <span style='color:{RED}'>● Severe</span><span style='color:{ORANGE}'>● High</span>
              <span style='color:{YELLOW}'>● Moderate</span><span style='color:{GREEN}'>● Stable</span>
            </div></div>""",unsafe_allow_html=True)

        # ── COMPANY EXPOSURE ──────────────────────────────────────────────
        if ci:
            mx=max(abs(c["expected_impact_pct"]) for c in ci) or 1
            co_rows="".join(f"""<div class='v-co'>
              <div class='v-co-name'>{c["name"]}<div class='v-co-tick'>{c.get("ticker","")}</div></div>
              <div class='v-co-bar'><div class='v-co-fill' style='width:{round(abs(c["expected_impact_pct"])/mx*100)}%;background:{RED if c["expected_impact_pct"]<0 else GREEN}'></div></div>
              <span class='v-co-pct' style='color:{RED if c["expected_impact_pct"]<0 else GREEN}'>{c["expected_impact_pct"]:+.1f}%</span>
            </div>""" for c in ci[:8])
            st.markdown(f"""<div class='v-card'>
              <div class='v-card-hdr'><div class='v-card-hdr-line' style='background:{ACCENT}'></div><span class='v-card-title'>Company Exposure</span></div>
              <div class='v-card-body'>{co_rows}</div></div>""",unsafe_allow_html=True)

        # ── WHAT TO DO ────────────────────────────────────────────────────
        if mits:
            acts="".join(f"""<div class='v-act'>
              <span class='v-act-n'>{j}</span><span class='v-act-t'>{m}</span></div>""" for j,m in enumerate(mits[:5],1))
            ov=max(0.0,min(float(conf.get("overall",0.0)),1.0))*100
            ep=max(0.0,min(float(conf.get("event_parsing",0.0)),1.0))*100
            nsrc=max(1,len(ii)+len(ci))
            lv="high" if ov>70 else "moderate" if ov>45 else "baseline"
            note=f"This is a {lv}-confidence assessment derived from causal propagation across {nsrc} data sources. Confidence reflects scenario match to historical patterns — not the severity of the event itself."
            st.markdown(f"""<div class='v-card'>
              <div class='v-card-hdr'><div class='v-card-hdr-line' style='background:{ACCENT}'></div><span class='v-card-title'>What to Do · Recommended Actions</span></div>
              <div class='v-card-body'>{acts}
                <div class='v-conf'><span class='v-conf-i'>Confidence <span class='v-conf-v'>{ov:.0f}%</span></span>
                <span class='v-conf-i'>Event parsing <span class='v-conf-v'>{ep:.0f}%</span></span>
                <span class='v-conf-i'>Sources <span class='v-conf-v'>{nsrc}</span></span></div>
                <div class='v-conf-note'>{note}</div>
              </div></div>""",unsafe_allow_html=True)

    st.markdown("</div>",unsafe_allow_html=True)

    # BOTTOM INPUT
    st.markdown("<div class='v-bottom'><div class='v-bot-inner'>",unsafe_allow_html=True)
    bc1,bc2=st.columns([8,2])
    with bc1:
        q2=st.text_input("q2","",placeholder="Try another scenario or ask a follow-up…",
                         label_visibility="collapsed",key="follow_input")
    with bc2:
        st.markdown("<div class='v-bot-send'>",unsafe_allow_html=True)
        send2=st.button("Simulate →",key="follow_send",use_container_width=True,type="primary")
        st.markdown("</div>",unsafe_allow_html=True)
    st.markdown(f"""</div><div class='v-bot-note'>Causal simulation engine · Multi-sector propagation · Probabilistic modeling</div></div>""",unsafe_allow_html=True)

    if send2 and q2.strip():
        st.session_state.pending=q2.strip(); st.rerun()
