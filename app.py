import json
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from data_fetcher import fetch_stock_data_live, get_company_info
from predictor import StockPredictor
from utils import calculate_technical_indicators, format_large_number

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI – Live",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #060a14; color: #e2e8f0; }

.main-header {
    background: linear-gradient(135deg,#0d1b3e 0%,#1a0a2e 50%,#0d1b3e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.2rem 2rem;
    margin-bottom: 0.8rem;
}
.main-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg,#00d4ff,#7c3aed,#00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header p {
    color: #94a3b8;
    margin: 0.15rem 0 0;
    font-size: 0.85rem;
}

.datetime-strip {
    background: #0d1224;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 0.45rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    flex-wrap: wrap;
}
.datetime-strip .dt-label { color: #334155; }
.datetime-strip .dt-val   { color: #94a3b8; font-weight: 600; }
.datetime-strip .dt-live  { color: #10b981; animation: pulse 1.4s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }

.metric-card {
    background: linear-gradient(135deg,#111827,#1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
}
.metric-card .label {
    color: #64748b;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-card .value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
}
.positive { color: #10b981; }
.negative { color: #ef4444; }

.info-chip {
    display:inline-block;
    background:#1e293b;
    border:1px solid #334155;
    border-radius:20px;
    padding:0.15rem 0.6rem;
    font-size:0.7rem;
    color:#94a3b8;
    margin:0.12rem;
}

.prediction-banner {
    background:linear-gradient(135deg,#0d2d0d,#0a1a2e);
    border:1px solid #10b981;
    border-radius:12px;
    padding:1.2rem;
    margin:0.7rem 0;
}
.prediction-banner.bearish {
    background:linear-gradient(135deg,#2d0d0d,#1a0a0a);
    border-color:#ef4444;
}

.market-badge-us {
    display:inline-block;
    background: linear-gradient(135deg,#1d4ed8,#1e40af);
    border-radius:6px;
    padding:0.12rem 0.5rem;
    font-size:0.7rem;
    color:#fff;
    font-weight:700;
    letter-spacing:0.5px;
}
.market-badge-in {
    display:inline-block;
    background: linear-gradient(135deg,#f97316,#ea580c);
    border-radius:6px;
    padding:0.12rem 0.5rem;
    font-size:0.7rem;
    color:#fff;
    font-weight:700;
    letter-spacing:0.5px;
}

[data-testid="stSidebar"] { background:#060a14 !important; }

.stButton>button {
    background:linear-gradient(135deg,#1d4ed8,#7c3aed);
    color:#fff;
    border:none;
    border-radius:8px;
    font-weight:600;
    padding:0.5rem 1rem;
    width:100%;
}
.stButton>button:hover {
    transform:translateY(-1px);
    box-shadow:0 4px 16px rgba(124,58,237,.4);
}

.stTabs [aria-selected="true"] {
    color:#00d4ff !important;
    border-bottom-color:#00d4ff !important;
}

div[data-testid="column"] { padding:0 0.2rem; }

.pred-warning {
    background: linear-gradient(135deg,#1a1a0a,#2a1a00);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #fbbf24;
    font-size: 0.82rem;
    margin: 0.5rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def infer_tradingview_symbol(yf_ticker: str) -> str:
    t = (yf_ticker or "").upper().strip()
    if ":" in t:
        return t
    if t.endswith(".NS"):
        return f"NSE:{t[:-3]}"
    if t.endswith(".BO"):
        return f"BSE:{t[:-3]}"
    return f"NSE:{t}"


# ─────────────────────────────────────────────────────────────────────────────
# STOCK LISTS
# ─────────────────────────────────────────────────────────────────────────────
INDIA_STOCKS = {
    # ── Banking ──────────────────────────────────────────────────────────
    "🏦 HDFC Bank (HDFCBANK.NS)":         ("HDFCBANK.NS",   "NSE:HDFCBANK"),
    "🏛️ ICICI Bank (ICICIBANK.NS)":       ("ICICIBANK.NS",  "NSE:ICICIBANK"),
    "🏢 SBI (SBIN.NS)":                   ("SBIN.NS",       "NSE:SBIN"),
    "🟠 Kotak Bank (KOTAKBANK.NS)":       ("KOTAKBANK.NS",  "NSE:KOTAKBANK"),
    "🔵 Axis Bank (AXISBANK.NS)":         ("AXISBANK.NS",   "NSE:AXISBANK"),
    "🟢 IndusInd Bank (INDUSINDBK.NS)":   ("INDUSINDBK.NS", "NSE:INDUSINDBK"),
    "🏦 Bank of Baroda (BANKBARODA.NS)":  ("BANKBARODA.NS", "NSE:BANKBARODA"),
    "🏦 PNB (PNB.NS)":                    ("PNB.NS",        "NSE:PNB"),
    "🏦 Canara Bank (CANBK.NS)":          ("CANBK.NS",      "NSE:CANBK"),
    "🏦 IDFC First Bank (IDFCFIRSTB.NS)": ("IDFCFIRSTB.NS", "NSE:IDFCFIRSTB"),
    "🏦 AU Small Finance (AUBANK.NS)":    ("AUBANK.NS",     "NSE:AUBANK"),
    "🏦 Federal Bank (FEDERALBNK.NS)":    ("FEDERALBNK.NS", "NSE:FEDERALBNK"),

    # ── NBFC / Insurance ─────────────────────────────────────────────────
    "🟡 Bajaj Finance (BAJFINANCE.NS)":   ("BAJFINANCE.NS", "NSE:BAJFINANCE"),
    "💰 Bajaj Finserv (BAJAJFINSV.NS)":   ("BAJAJFINSV.NS", "NSE:BAJAJFINSV"),
    "📊 SBI Life (SBILIFE.NS)":           ("SBILIFE.NS",    "NSE:SBILIFE"),
    "📊 HDFC Life (HDFCLIFE.NS)":         ("HDFCLIFE.NS",   "NSE:HDFCLIFE"),
    "📊 ICICI Pru Life (ICICIPRULI.NS)":  ("ICICIPRULI.NS", "NSE:ICICIPRULI"),
    "📊 ICICI Lombard (ICICIGI.NS)":      ("ICICIGI.NS",    "NSE:ICICIGI"),
    "💳 Shriram Finance (SHRIRAMFIN.NS)": ("SHRIRAMFIN.NS", "NSE:SHRIRAMFIN"),
    "💳 Muthoot Finance (MUTHOOTFIN.NS)": ("MUTHOOTFIN.NS", "NSE:MUTHOOTFIN"),

    # ── IT ───────────────────────────────────────────────────────────────
    "💻 TCS (TCS.NS)":                    ("TCS.NS",        "NSE:TCS"),
    "🔵 Infosys (INFY.NS)":               ("INFY.NS",       "NSE:INFY"),
    "🟡 Wipro (WIPRO.NS)":                ("WIPRO.NS",      "NSE:WIPRO"),
    "🟠 HCL Technologies (HCLTECH.NS)":   ("HCLTECH.NS",    "NSE:HCLTECH"),
    "🔴 Tech Mahindra (TECHM.NS)":        ("TECHM.NS",      "NSE:TECHM"),
    "🟢 LTIMindtree (LTIM.NS)":           ("LTIM.NS",       "NSE:LTIM"),
    "🔷 Mphasis (MPHASIS.NS)":            ("MPHASIS.NS",    "NSE:MPHASIS"),
    "🌐 Persistent Sys (PERSISTENT.NS)":  ("PERSISTENT.NS", "NSE:PERSISTENT"),

    # ── Energy / Power / Oil & Gas ──────────────────────────────────────
    "⚡ Reliance (RELIANCE.NS)":          ("RELIANCE.NS",   "NSE:RELIANCE"),
    "🛢️ ONGC (ONGC.NS)":                  ("ONGC.NS",       "NSE:ONGC"),
    "⛽ BPCL (BPCL.NS)":                  ("BPCL.NS",       "NSE:BPCL"),
    "⛽ IOC (IOC.NS)":                    ("IOC.NS",        "NSE:IOC"),
    "🔥 GAIL (GAIL.NS)":                  ("GAIL.NS",       "NSE:GAIL"),
    "⚡ NTPC (NTPC.NS)":                  ("NTPC.NS",       "NSE:NTPC"),
    "🔌 Power Grid (POWERGRID.NS)":       ("POWERGRID.NS",  "NSE:POWERGRID"),
    "⛏️ Coal India (COALINDIA.NS)":       ("COALINDIA.NS",  "NSE:COALINDIA"),
    "⚡ Tata Power (TATAPOWER.NS)":       ("TATAPOWER.NS",  "NSE:TATAPOWER"),
    "💧 NHPC (NHPC.NS)":                  ("NHPC.NS",       "NSE:NHPC"),

    # ── Metals ───────────────────────────────────────────────────────────
    "🔩 Tata Steel (TATASTEEL.NS)":       ("TATASTEEL.NS",  "NSE:TATASTEEL"),
    "🔴 JSW Steel (JSWSTEEL.NS)":         ("JSWSTEEL.NS",   "NSE:JSWSTEEL"),
    "⛏️ Hindalco (HINDALCO.NS)":          ("HINDALCO.NS",   "NSE:HINDALCO"),
    "⛏️ Vedanta (VEDL.NS)":               ("VEDL.NS",       "NSE:VEDL"),
    "🔩 Jindal Steel (JINDALSTEL.NS)":    ("JINDALSTEL.NS", "NSE:JINDALSTEL"),
    "🔩 SAIL (SAIL.NS)":                  ("SAIL.NS",       "NSE:SAIL"),

    # ── Auto ─────────────────────────────────────────────────────────────
    "🚗 Tata Motors (TATAMOTORS.NS)":     ("TATAMOTORS.NS", "NSE:TATAMOTORS"),
    "🚗 Maruti Suzuki (MARUTI.NS)":       ("MARUTI.NS",     "NSE:MARUTI"),
    "🚙 M&M (M&M.NS)":                    ("M&M.NS",        "NSE:MM"),
    "🔴 Bajaj Auto (BAJAJ-AUTO.NS)":      ("BAJAJ-AUTO.NS", "NSE:BAJAJ_AUTO"),
    "🏍️ Hero MotoCorp (HEROMOTOCO.NS)":   ("HEROMOTOCO.NS", "NSE:HEROMOTOCO"),
    "🏍️ Eicher Motors (EICHERMOT.NS)":    ("EICHERMOT.NS",  "NSE:EICHERMOT"),
    "🏍️ TVS Motor (TVSMOTOR.NS)":         ("TVSMOTOR.NS",   "NSE:TVSMOTOR"),
    "🚛 Ashok Leyland (ASHOKLEY.NS)":     ("ASHOKLEY.NS",   "NSE:ASHOKLEY"),

    # ── FMCG ─────────────────────────────────────────────────────────────
    "🧴 HUL (HINDUNILVR.NS)":             ("HINDUNILVR.NS", "NSE:HINDUNILVR"),
    "🥤 ITC (ITC.NS)":                    ("ITC.NS",        "NSE:ITC"),
    "🥛 Nestle India (NESTLEIND.NS)":     ("NESTLEIND.NS",  "NSE:NESTLEIND"),
    "🍫 Britannia (BRITANNIA.NS)":        ("BRITANNIA.NS",  "NSE:BRITANNIA"),
    "🟠 Dabur (DABUR.NS)":                ("DABUR.NS",      "NSE:DABUR"),
    "🧼 Marico (MARICO.NS)":              ("MARICO.NS",     "NSE:MARICO"),
    "🧴 Godrej Consumer (GODREJCP.NS)":   ("GODREJCP.NS",   "NSE:GODREJCP"),
    "☕ Tata Consumer (TATACONSUM.NS)":   ("TATACONSUM.NS", "NSE:TATACONSUM"),

    # ── Pharma ───────────────────────────────────────────────────────────
    "💊 Sun Pharma (SUNPHARMA.NS)":       ("SUNPHARMA.NS",  "NSE:SUNPHARMA"),
    "🧬 Dr Reddy's (DRREDDY.NS)":         ("DRREDDY.NS",    "NSE:DRREDDY"),
    "💉 Cipla (CIPLA.NS)":                ("CIPLA.NS",      "NSE:CIPLA"),
    "🔵 Divi's Lab (DIVISLAB.NS)":        ("DIVISLAB.NS",   "NSE:DIVISLAB"),
    "🩺 Apollo Hospitals (APOLLOHOSP.NS)":("APOLLOHOSP.NS", "NSE:APOLLOHOSP"),
    "💊 Lupin (LUPIN.NS)":                ("LUPIN.NS",      "NSE:LUPIN"),
    "💊 Aurobindo Pharma (AUROPHARMA.NS)":("AUROPHARMA.NS", "NSE:AUROPHARMA"),

    # ── Cement ───────────────────────────────────────────────────────────
    "🏗️ UltraTech Cement (ULTRACEMCO.NS)":("ULTRACEMCO.NS","NSE:ULTRACEMCO"),
    "🏗️ Grasim (GRASIM.NS)":              ("GRASIM.NS",     "NSE:GRASIM"),
    "🏗️ Shree Cement (SHREECEM.NS)":      ("SHREECEM.NS",   "NSE:SHREECEM"),
    "🏗️ Ambuja Cement (AMBUJACEM.NS)":    ("AMBUJACEM.NS",  "NSE:AMBUJACEM"),

    # ── Telecom ──────────────────────────────────────────────────────────
    "📡 Bharti Airtel (BHARTIARTL.NS)":   ("BHARTIARTL.NS", "NSE:BHARTIARTL"),
    "📱 Vodafone Idea (IDEA.NS)":         ("IDEA.NS",       "NSE:IDEA"),

    # ── Industrial / Conglomerate ───────────────────────────────────────
    "🔵 L&T (LT.NS)":                     ("LT.NS",         "NSE:LT"),
    "🏗️ Adani Enterprises (ADANIENT.NS)":("ADANIENT.NS",   "NSE:ADANIENT"),
    "🟡 Adani Ports (ADANIPORTS.NS)":     ("ADANIPORTS.NS", "NSE:ADANIPORTS"),
    "⚡ Adani Green (ADANIGREEN.NS)":     ("ADANIGREEN.NS", "NSE:ADANIGREEN"),
    "⚙️ Siemens (SIEMENS.NS)":            ("SIEMENS.NS",    "NSE:SIEMENS"),
    "⚙️ ABB India (ABB.NS)":              ("ABB.NS",        "NSE:ABB"),
    "💡 Havells (HAVELLS.NS)":            ("HAVELLS.NS",    "NSE:HAVELLS"),
    "🛡️ Bharat Electronics (BEL.NS)":     ("BEL.NS",        "NSE:BEL"),

    # ── Consumer / Retail / Paints ──────────────────────────────────────
    "🛒 Avenue Supermarts (DMART.NS)":    ("DMART.NS",      "NSE:DMART"),
    "💎 Titan (TITAN.NS)":                ("TITAN.NS",      "NSE:TITAN"),
    "🛍️ Trent (TRENT.NS)":                ("TRENT.NS",      "NSE:TRENT"),
    "🎨 Asian Paints (ASIANPAINT.NS)":    ("ASIANPAINT.NS", "NSE:ASIANPAINT"),
    "🧪 Pidilite (PIDILITIND.NS)":        ("PIDILITIND.NS", "NSE:PIDILITIND"),
    "🦷 Colgate India (COLPAL.NS)":       ("COLPAL.NS",     "NSE:COLPAL"),

    # ── Real Estate ──────────────────────────────────────────────────────
    "🏗️ DLF (DLF.NS)":                    ("DLF.NS",        "NSE:DLF"),
    "🏢 Godrej Properties (GODREJPROP.NS)":("GODREJPROP.NS","NSE:GODREJPROP"),

    # ── New-Age / Tech ───────────────────────────────────────────────────
    "🍔 Zomato (ZOMATO.NS)":              ("ZOMATO.NS",     "NSE:ZOMATO"),
    "💼 Naukri / Info Edge (NAUKRI.NS)":  ("NAUKRI.NS",     "NSE:NAUKRI"),
    "🚆 IRCTC (IRCTC.NS)":                ("IRCTC.NS",      "NSE:IRCTC"),
    "📱 Paytm (PAYTM.NS)":                ("PAYTM.NS",      "NSE:PAYTM"),
    "📊 PB Fintech / Policybazaar (POLICYBZR.NS)": ("POLICYBZR.NS", "NSE:POLICYBZR"),

    # ── Aviation / Media ─────────────────────────────────────────────────
    "✈️ InterGlobe / IndiGo (INDIGO.NS)": ("INDIGO.NS",     "NSE:INDIGO"),
    "📺 Zee Entertainment (ZEEL.NS)":     ("ZEEL.NS",       "NSE:ZEEL"),
    "🎬 PVR INOX (PVRINOX.NS)":           ("PVRINOX.NS",    "NSE:PVRINOX"),

    # ── Auto Ancillary ───────────────────────────────────────────────────
    "⚙️ Bharat Forge (BHARATFORG.NS)":    ("BHARATFORG.NS", "NSE:BHARATFORG"),
    "🚗 Samvardhana Motherson (MOTHERSON.NS)": ("MOTHERSON.NS", "NSE:MOTHERSON"),

    # ── Custom ───────────────────────────────────────────────────────────
    "🔑 Custom India Ticker":             ("CUSTOM_IN", "CUSTOM"),
}


# ─────────────────────────────────────────────────────────────────────────────
# PERIOD COMPATIBILITY — prevent yfinance errors
# ─────────────────────────────────────────────────────────────────────────────
# yfinance rules: intraday intervals only work with limited periods
INTERVAL_MAX_PERIOD = {
    "1m":  "7d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "1h":  "730d",
    "1d":  "10y",
    "1wk": "10y",
    "1mo": "10y",
}

# For AI prediction, minimum data rows needed
MIN_ROWS_FOR_PREDICTION = 60


def get_prediction_period(interval: str, user_period: str) -> str:
    """
    For AI prediction, always fetch enough data regardless of chart period.
    Returns the period string to use for prediction fetch.
    """
    enough_periods = {
        "1m":  "7d",     # max for 1m
        "5m":  "60d",
        "15m": "60d",
        "30m": "60d",
        "1h":  "6mo",
        "1d":  "2y",
        "1wk": "5y",
        "1mo": "5y",
    }
    return enough_periods.get(interval, "1y")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [
    ("live_mode", False),
    ("refresh_count", 0),
    ("last_refresh", 0.0),
    ("prev_price", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:0.6rem 0 0.2rem">
            <span style="font-size:1.6rem">📈</span>
            <h2 style="color:#00d4ff;margin:0.15rem 0 0;font-size:1.1rem">StockSense AI</h2>
            <p style="color:#475569;font-size:0.7rem;margin:0">Live Prediction Engine</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**🏢 India Company (NSE)**")
    sel = st.selectbox("", list(INDIA_STOCKS.keys()), label_visibility="collapsed")
    yf_ticker, tv_symbol = INDIA_STOCKS[sel]
    is_india = True

    if yf_ticker == "CUSTOM_IN":
        yf_ticker = st.text_input("Yahoo Finance Ticker (e.g. RELIANCE.NS)", "RELIANCE.NS").upper().strip()
        tv_symbol = st.text_input("TradingView Symbol (e.g. NSE:RELIANCE)", infer_tradingview_symbol(yf_ticker)).upper().strip()
    else:
        tv_symbol = infer_tradingview_symbol(tv_symbol)

    st.markdown("**⏱️ Interval**")
    interval_map = {
        "1 Min": "1m",
        "5 Min": "5m",
        "15 Min": "15m",
        "30 Min": "30m",
        "1 Hour": "1h",
        "1 Day": "1d",
        "1 Week": "1wk",
        "1 Month": "1mo",
    }
    int_label = st.selectbox("", list(interval_map.keys()), index=5, label_visibility="collapsed")  # default: 1 Day
    interval = interval_map[int_label]

    st.markdown("**📅 History Period (Chart)**")
    # Period options filtered by interval compatibility
    all_periods = {
        "7 Days": "7d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
    }
    # Filter: intraday intervals can't use very long periods
    if interval in ["1m"]:
        valid_periods = {"7 Days": "7d"}
    elif interval in ["5m", "15m", "30m"]:
        valid_periods = {k: v for k, v in all_periods.items() if v in ["7d", "1mo", "3mo"]}
    elif interval in ["1h"]:
        valid_periods = {k: v for k, v in all_periods.items() if v not in ["5y"]}
    else:
        valid_periods = all_periods

    per_label = st.selectbox("", list(valid_periods.keys()),
                             index=min(2, len(valid_periods)-1),
                             label_visibility="collapsed")
    period = valid_periods[per_label]

    st.markdown("**🔮 Prediction Days**")
    pred_days = st.slider("", 1, 30, 7, label_visibility="collapsed")

    st.markdown("**🧠 ML Model**")
    model_choice = st.selectbox(
        "",
        ["Linear Regression", "Random Forest", "LSTM Neural Network", "All Models (Ensemble)"],
        label_visibility="collapsed",
    )

    show_ind = st.checkbox("Technical Indicators", value=True)
    show_vol = st.checkbox("Show Volume", value=True)

    st.divider()

    st.markdown("**⚡ Live Mode**")
    refresh_sec = st.selectbox(
        "Refresh every",
        ["Off", "30 sec", "1 min", "2 min", "5 min"],
        index=2,
        label_visibility="collapsed",
    )

    c1, c2 = st.columns(2)
    with c1:
        load_btn = st.button("🚀 Load")
    with c2:
        live_lbl = "🔴 Stop" if st.session_state["live_mode"] else "🟢 Go Live"
        live_btn = st.button(live_lbl)

    if load_btn:
        st.session_state["live_mode"] = False
        fetch_stock_data_live.clear()

    if live_btn:
        st.session_state["live_mode"] = not st.session_state["live_mode"]
        st.session_state["last_refresh"] = 0.0
        fetch_stock_data_live.clear()

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#334155;font-size:0.65rem'>Data: Yahoo Finance • ⚠️ Not financial advice</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────────────────────────────────────
rsec_map = {"Off": 0, "30 sec": 30, "1 min": 60, "2 min": 120, "5 min": 300}
refresh_every = rsec_map[refresh_sec]

if st.session_state["live_mode"] and refresh_every > 0 and HAS_AUTOREFRESH:
    st_autorefresh(interval=refresh_every * 1000, key="live_refresh")


# ─────────────────────────────────────────────────────────────────────────────
# DATE/TIME
# ─────────────────────────────────────────────────────────────────────────────
now_dt = datetime.now()
now_date = now_dt.strftime("%A, %d %B %Y")
now_time_12 = now_dt.strftime("%I:%M:%S %p")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
live_status = "🟢 LIVE" if st.session_state["live_mode"] else "⏸ PAUSED"
live_color = "#10b981" if st.session_state["live_mode"] else "#475569"
market_flag = "🇮🇳 India (NSE/BSE)" if is_india else "🇺🇸 US (NASDAQ/NYSE)"

st.markdown(
    f"""
<div class="main-header">
    <h1>📈 StockSense AI</h1>
    <p>Real-Time Market Data • ML Predictions • Technical Analysis &nbsp;
       <span style="color:{live_color};font-weight:700">{live_status}</span>
       {"&nbsp;| Refresh #" + str(st.session_state["refresh_count"]) if st.session_state["refresh_count"] else ""}
       &nbsp;|&nbsp; {market_flag}
    </p>
</div>
""",
    unsafe_allow_html=True,
)

elapsed_since = time.time() - st.session_state["last_refresh"]
next_in = max(0, int(refresh_every - elapsed_since)) if (st.session_state["live_mode"] and refresh_every > 0) else 0

st.markdown(
    f"""
<div class="datetime-strip">
    <span class="dt-live">● LIVE</span>
    <span><span class="dt-label">📅 Date&nbsp;</span><span class="dt-val">{now_date}</span></span>
    <span><span class="dt-label">🕐 Time&nbsp;</span><span class="dt-val">{now_time_12}</span></span>
    <span><span class="dt-label">📍 Market&nbsp;</span>
          <span class="dt-val" style="color:{'#f97316' if is_india else '#3b82f6'}">
          {'🇮🇳 NSE/BSE' if is_india else '🇺🇸 NYSE/NASDAQ'}
          </span></span>
    <span><span class="dt-label">⟳ Next refresh&nbsp;</span>
          <span class="dt-val" style="color:{'#10b981' if st.session_state['live_mode'] else '#334155'}">
          {'in ' + str(next_in) + 's' if st.session_state['live_mode'] and refresh_every > 0 else 'Off'}
          </span></span>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# FETCH DATA — Chart data (user's chosen period)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"⚡ Fetching {yf_ticker} ({int_label}, {per_label}) …"):
    df = fetch_stock_data_live(yf_ticker, period, interval)

if df is None or df.empty:
    st.error(f"❌ No data for **{yf_ticker}**. Check ticker or try a different period/interval.")
    if is_india:
        st.info("💡 India stocks use `.NS` suffix. E.g. `RELIANCE.NS`, `TCS.NS`")
    st.stop()

last_price = float(df["Close"].iloc[-1])

_intraday = ["1m", "5m", "15m", "30m", "1h"]
if interval in _intraday and len(df) > 1:
    today = df.index[-1].date()
    today_mask = df.index.date == today
    today_df = df[today_mask]
    if len(today_df) > 1:
        prev_close = float(today_df["Open"].iloc[0])
    elif len(df) > 1:
        prev_session = df[~today_mask]
        prev_close = float(prev_session["Close"].iloc[-1]) if len(prev_session) > 0 else float(df["Close"].iloc[-2])
    else:
        prev_close = last_price
else:
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_price

price_chg = last_price - prev_close
pct_chg = (price_chg / prev_close * 100) if prev_close else 0
is_up = price_chg >= 0
chg_icon = "▲" if is_up else "▼"
chg_color = "#10b981" if is_up else "#ef4444"
info = get_company_info(yf_ticker)
currency_symbol = "₹" if is_india else "$"

try:
    last_ts = df.index[-1].strftime("%d %b %Y  %H:%M:%S")
except Exception:
    last_ts = str(df.index[-1])

prev_sess = st.session_state.get("prev_price")
flash_text = ""
flash_col = "#10b981"
if prev_sess and prev_sess != last_price:
    diff = last_price - prev_sess
    flash_text = f"{'▲' if diff > 0 else '▼'} {abs(diff):.2f} from last refresh"
    flash_col = "#10b981" if diff > 0 else "#ef4444"
st.session_state["prev_price"] = last_price

if st.session_state["live_mode"] and refresh_every > 0:
    st.session_state["last_refresh"] = time.time()
    st.session_state["refresh_count"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# FETCH PREDICTION DATA — always fetch enough history for ML
# ─────────────────────────────────────────────────────────────────────────────
pred_period = get_prediction_period(interval, period)
if pred_period != period:
    with st.spinner(f"🧠 Fetching extended history for AI model ({pred_period}) …"):
        df_pred = fetch_stock_data_live(yf_ticker, pred_period, interval)
    if df_pred is None or len(df_pred) < MIN_ROWS_FOR_PREDICTION:
        # Fallback to daily if intraday doesn't have enough
        with st.spinner("🧠 Fetching daily data for AI model …"):
            df_pred = fetch_stock_data_live(yf_ticker, "2y", "1d")
else:
    df_pred = df.copy()

if df_pred is None or len(df_pred) < MIN_ROWS_FOR_PREDICTION:
    df_pred = df.copy()


# ─────────────────────────────────────────────────────────────────────────────
# TOP TICKER BAR
# ─────────────────────────────────────────────────────────────────────────────
tb1, tb2, tb3, tb4, tb5 = st.columns([1.2, 1.5, 1.5, 2, 2])

with tb1:
    dot = "🟢" if st.session_state["live_mode"] else "⚪"
    badge_html = f'<span class="market-badge-{"in" if is_india else "us"}">{"🇮🇳 NSE" if is_india else "🇺🇸 US"}</span>'
    st.markdown(
        f"<div style='padding:0.4rem 0;font-family:JetBrains Mono,monospace;font-size:0.85rem;color:#94a3b8'>{dot} <b>{yf_ticker}</b><br>{badge_html}</div>",
        unsafe_allow_html=True,
    )

with tb2:
    st.markdown(
        f"<div style='padding:0.3rem 0;font-family:JetBrains Mono,monospace;font-size:1.5rem;font-weight:700;color:#f1f5f9'>{currency_symbol}{last_price:.2f}</div>",
        unsafe_allow_html=True,
    )

with tb3:
    st.markdown(
        f"<div style='padding:0.4rem 0;font-size:1rem;font-weight:700;color:{chg_color}'>{chg_icon} {abs(price_chg):.2f} ({abs(pct_chg):.2f}%)</div>",
        unsafe_allow_html=True,
    )

with tb4:
    st.markdown(
        f"<div style='padding:0.4rem 0;font-size:0.75rem;color:#475569'>🕐 Last bar: {last_ts}</div>",
        unsafe_allow_html=True,
    )

with tb5:
    if st.session_state["live_mode"] and refresh_every > 0:
        st.markdown(
            f"<div style='padding:0.4rem 0;font-size:0.75rem;color:#334155'>⟳ Next refresh in <b style='color:#475569'>{next_in}s</b></div>",
            unsafe_allow_html=True,
        )
    elif flash_text:
        st.markdown(
            f"<div style='padding:0.4rem 0;font-size:0.75rem;color:{flash_col}'>{flash_text}</div>",
            unsafe_allow_html=True,
        )

st.markdown("<hr style='border-color:#1e3a5f;margin:0.4rem 0 0.8rem'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────
high_p = df["High"].max()
low_p  = df["Low"].min()
avg_vol = df["Volume"].mean() if "Volume" in df.columns else 0
vol_std = df["Close"].pct_change().std() * 100

mc0, mc1, mc2, mc3, mc4 = st.columns([2, 1, 1, 1, 1])

with mc0:
    mkt_badge = f'<span class="market-badge-{"in" if is_india else "us"}">{"🇮🇳 India" if is_india else "🇺🇸 US"}</span>'
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:1.3rem;font-weight:700;color:#f1f5f9">{info.get('shortName', yf_ticker)}</div>
            <div style="color:#64748b;font-size:0.78rem;margin:0.15rem 0">{info.get('sector', '—')} • {info.get('exchange', '—')}</div>
            <span class="info-chip">{yf_ticker}</span>
            <span class="info-chip">{currency_symbol}</span>
            <span class="info-chip">{int_label}</span>
            &nbsp;{mkt_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )

with mc1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Price</div>
            <div class="value">{currency_symbol}{last_price:.2f}</div>
            <div style="color:{chg_color};font-size:0.82rem;font-weight:600">{chg_icon} {abs(pct_chg):.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mc2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Period H / L</div>
            <div class="value" style="font-size:1.1rem">{high_p:.2f}</div>
            <div style="color:#64748b;font-size:0.73rem">Low: {low_p:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mc3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Avg Volume</div>
            <div class="value" style="font-size:1.1rem">{format_large_number(avg_vol)}</div>
            <div style="color:#64748b;font-size:0.73rem">Shares/bar</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mc4:
    vc = "#f59e0b" if vol_std > 2 else "#10b981"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">Volatility</div>
            <div class="value" style="font-size:1.1rem;color:{vc}">{vol_std:.2f}%</div>
            <div style="color:#64748b;font-size:0.73rem">Daily σ</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

if show_ind:
    df = calculate_technical_indicators(df)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Chart", "🔮 AI Prediction", "📉 Indicators", "📋 Data"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE CHART
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        f"""
        <div style="background:#0d1224;border:1px solid #1a2540;border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.8rem">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
                <div>
                    <div style="color:#94a3b8;font-size:0.75rem">LIVE MARKET CHART — Yahoo Finance Data &nbsp;
                        <span class="market-badge-{'in' if is_india else 'us'}">
                        {'🇮🇳 NSE/BSE' if is_india else '🇺🇸 NYSE/NASDAQ'}
                        </span>
                    </div>
                    <div style="color:#f1f5f9;font-size:1.05rem;font-weight:700">{yf_ticker} &nbsp;
                        <span style="color:#64748b;font-size:0.8rem">• {int_label} • {per_label}</span>
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="color:#94a3b8;font-size:0.72rem">Last Price</div>
                    <div style="color:#f1f5f9;font-size:1.1rem;font-family:JetBrains Mono,monospace;font-weight:700">
                        {currency_symbol}{last_price:.2f}
                        <span style="font-size:0.85rem;color:{chg_color};margin-left:6px">{chg_icon} {abs(pct_chg):.2f}%</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_df = df.copy()

    c_opt1, c_opt2, c_opt3 = st.columns([2, 2, 3])
    with c_opt1:
        show_ma = st.checkbox("📈 Moving Averages", value=True, key="chart_ma")
    with c_opt2:
        show_bb_chart = st.checkbox("📊 Bollinger Bands", value=False, key="chart_bb")
    with c_opt3:
        chart_type = st.radio("Chart Type", ["Candlestick", "Line", "OHLC"], horizontal=True, key="chart_type")

    fig_rows = [0.75, 0.25] if show_vol else [1.0]
    n_rows = 2 if show_vol else 1

    fc = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=fig_rows,
    )

    if chart_type == "Candlestick":
        fc.add_trace(
            go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open"], high=chart_df["High"],
                low=chart_df["Low"],   close=chart_df["Close"],
                name=yf_ticker,
                increasing_line_color="#10b981", decreasing_line_color="#ef4444",
                increasing_fillcolor="#10b981",  decreasing_fillcolor="#ef4444",
            ), row=1, col=1,
        )
    elif chart_type == "Line":
        fc.add_trace(
            go.Scatter(
                x=chart_df.index, y=chart_df["Close"],
                name=yf_ticker,
                line=dict(color="#00d4ff", width=1.8),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
            ), row=1, col=1,
        )
    else:
        fc.add_trace(
            go.Ohlc(
                x=chart_df.index,
                open=chart_df["Open"], high=chart_df["High"],
                low=chart_df["Low"],   close=chart_df["Close"],
                name=yf_ticker,
                increasing_line_color="#10b981", decreasing_line_color="#ef4444",
            ), row=1, col=1,
        )

    if show_ma and show_ind:
        ma_colors = {"MA5": "#f59e0b", "MA10": "#8b5cf6", "MA20": "#00d4ff", "MA50": "#f97316", "MA200": "#ec4899"}
        for ma, color in ma_colors.items():
            if ma in chart_df.columns and chart_df[ma].notna().sum() > 3:
                fc.add_trace(
                    go.Scatter(
                        x=chart_df.index, y=chart_df[ma],
                        name=ma, line=dict(color=color, width=1.2), opacity=0.85,
                    ), row=1, col=1,
                )

    if show_bb_chart and show_ind and "BB_Upper" in chart_df.columns:
        fc.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["BB_Upper"],
                       name="BB Upper", line=dict(color="rgba(124,58,237,0.7)", width=1, dash="dot")),
            row=1, col=1,
        )
        fc.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["BB_Lower"],
                       name="BB Lower", line=dict(color="rgba(124,58,237,0.7)", width=1, dash="dot"),
                       fill="tonexty", fillcolor="rgba(124,58,237,0.04)"),
            row=1, col=1,
        )

    if show_vol and "Volume" in chart_df.columns and n_rows == 2:
        vol_colors = [
            "#10b981" if chart_df["Close"].iloc[i] >= chart_df["Open"].iloc[i] else "#ef4444"
            for i in range(len(chart_df))
        ]
        fc.add_trace(
            go.Bar(x=chart_df.index, y=chart_df["Volume"],
                   name="Volume", marker_color=vol_colors, opacity=0.6),
            row=2, col=1,
        )

    fc.update_layout(
        height=620,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,10,20,0.98)",
        font=dict(color="#94a3b8", family="JetBrains Mono, monospace"),
        legend=dict(bgcolor="rgba(17,24,39,0.92)", bordercolor="#1e3a5f", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(t=10, b=10, l=10, r=10),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    fc.update_xaxes(gridcolor="rgba(30,58,95,0.25)", showspikes=True, spikecolor="#334155", spikethickness=1)
    fc.update_yaxes(gridcolor="rgba(30,58,95,0.25)", showspikes=True, spikecolor="#334155")

    st.plotly_chart(fc, use_container_width=True)
    st.success(f"✅ Showing real **{yf_ticker}** data from Yahoo Finance.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — AI PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    # Show info about data used for prediction
    pred_rows = len(df_pred)
    pred_interval_label = int_label if pred_period == period else f"{int_label} ({pred_period} history)"

    st.markdown(
        f"""
        <div style="background:#0d1224;border:1px solid #1a2540;border-radius:10px;
             padding:0.6rem 1rem;margin-bottom:0.8rem;font-size:0.78rem;color:#64748b">
            🧠 <b style="color:#94a3b8">AI Training Data:</b>
            {pred_rows} bars of {pred_interval_label} data &nbsp;|&nbsp;
            Model: <b style="color:#00d4ff">{model_choice}</b> &nbsp;|&nbsp;
            Predicting: <b style="color:#7c3aed">{pred_days} future bars</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pred_rows < MIN_ROWS_FOR_PREDICTION:
        st.markdown(
            f"""<div class="pred-warning">
            ⚠️ Only {pred_rows} data rows available. AI prediction needs at least {MIN_ROWS_FOR_PREDICTION} rows.
            Try selecting <b>1 Day interval + 1 Year period</b> or <b>1 Hour interval + 6 Month period</b> for best results.
            </div>""",
            unsafe_allow_html=True,
        )

    with st.spinner("🧠 Training AI model on live data …"):
        predictor = StockPredictor(model_choice)
        predictions, confidence, metrics = predictor.predict(df_pred, pred_days)

    if predictions is None:
        st.warning(
            f"⚠️ Not enough data for prediction ({pred_rows} rows). "
            "Please use **1 Day interval** with **3 Months or more** history period for best AI results."
        )
    else:
        # Future dates: use business days for daily, else just add intervals
        if interval in ["1d", "1wk", "1mo"]:
            future_dates = pd.date_range(df_pred.index[-1] + timedelta(days=1), periods=pred_days, freq="B")
        else:
            # For intraday, just add the interval delta
            interval_delta_map = {
                "1m": timedelta(minutes=1), "5m": timedelta(minutes=5),
                "15m": timedelta(minutes=15), "30m": timedelta(minutes=30),
                "1h": timedelta(hours=1),
            }
            delta = interval_delta_map.get(interval, timedelta(days=1))
            future_dates = pd.date_range(df_pred.index[-1] + delta, periods=pred_days, freq=delta)

        pred_chg = predictions[-1] - last_price
        pred_pct = pred_chg / last_price * 100
        is_bull = pred_chg >= 0

        st.markdown(
            f"""
            <div class="prediction-banner {'bullish' if is_bull else 'bearish'}">
                <div style="display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap">
                    <div>
                        <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px">
                            {'🟢 BULLISH' if is_bull else '🔴 BEARISH'} — {model_choice}
                        </div>
                        <div style="font-size:1.5rem;font-weight:700;color:{'#10b981' if is_bull else '#ef4444'}">
                            {'+' if is_bull else ''}{pred_pct:.2f}% over {pred_days} bars
                        </div>
                        <div style="color:#94a3b8;font-size:0.85rem">
                            {currency_symbol}{last_price:.2f} → <b style="color:#f1f5f9">{currency_symbol}{predictions[-1]:.2f}</b>
                        </div>
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:0.7rem;color:#64748b">CONFIDENCE</div>
                        <div style="font-size:1.9rem;font-weight:700;color:#00d4ff">{confidence:.0f}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Prediction chart
        fp = go.Figure()
        hw = min(120, len(df_pred))
        fp.add_trace(
            go.Scatter(
                x=df_pred.index[-hw:], y=df_pred["Close"].iloc[-hw:],
                name="Historical", line=dict(color="#00d4ff", width=1.8),
            )
        )

        # Confidence band (±% based on volatility)
        hist_vol = df_pred["Close"].pct_change().std()
        band_pct = max(0.02, hist_vol * np.sqrt(pred_days))
        upper = [p * (1 + band_pct) for p in predictions]
        lower = [p * (1 - band_pct) for p in predictions]

        fp.add_trace(
            go.Scatter(
                x=list(future_dates) + list(future_dates)[::-1],
                y=upper + lower[::-1],
                fill="toself", fillcolor="rgba(124,58,237,0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"±{band_pct*100:.1f}% Band",
            )
        )
        fp.add_trace(
            go.Scatter(
                x=future_dates, y=predictions,
                name="AI Forecast",
                mode="lines+markers",
                line=dict(color="#7c3aed", width=2.5, dash="dash"),
                marker=dict(size=7, color="#7c3aed", symbol="circle"),
            )
        )
        # Bridge line from last historical to first prediction
        fp.add_trace(
            go.Scatter(
                x=[df_pred.index[-1], future_dates[0]],
                y=[last_price, predictions[0]],
                line=dict(color="#7c3aed", width=1.5, dash="dot"),
                showlegend=False, mode="lines",
            )
        )

        fp.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,10,20,0.95)",
            font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(17,24,39,0.9)", bordercolor="#1e3a5f", borderwidth=1),
            margin=dict(t=10, b=10, l=10, r=20),
            hovermode="x unified",
        )
        fp.update_xaxes(gridcolor="rgba(30,58,95,0.2)")
        fp.update_yaxes(gridcolor="rgba(30,58,95,0.2)")
        st.plotly_chart(fp, use_container_width=True)

        # Metrics
        st.markdown("**📊 Model Metrics**")
        mcols = st.columns(4)
        metric_items = [
            ("RMSE", f"{metrics.get('rmse', 0):.4f}", "Root Mean Square Error — lower is better"),
            ("MAE",  f"{metrics.get('mae', 0):.4f}",  "Mean Absolute Error — lower is better"),
            ("R²",   f"{metrics.get('r2', 0):.4f}",   "R-squared — closer to 1 is better"),
            ("MAPE", f"{metrics.get('mape', 0):.2f}%", "Mean Absolute % Error — lower is better"),
        ]
        for col, (lbl, val, tip) in zip(mcols, metric_items):
            with col:
                r2_val = metrics.get("r2", 0)
                # Color code R² — red if suspiciously perfect
                val_color = "#ef4444" if lbl == "R²" and r2_val > 0.999 else "#f1f5f9"
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align:center" title="{tip}">
                        <div class="label">{lbl}</div>
                        <div class="value" style="font-size:1rem;color:{val_color}">{val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Forecast table
        st.markdown("<br>**📅 Forecast Table**", unsafe_allow_html=True)
        pred_df = pd.DataFrame({
            "Date": [d.strftime("%d %b %Y %H:%M") if interval in _intraday else d.strftime("%d %b %Y")
                     for d in future_dates],
            f"Predicted Price ({currency_symbol})": [f"{p:.2f}" for p in predictions],
            "Change": [f"{'+' if p-last_price>=0 else ''}{p-last_price:.2f}" for p in predictions],
            "Change %": [f"{'+' if (p-last_price)/last_price*100>=0 else ''}{(p-last_price)/last_price*100:.2f}%"
                         for p in predictions],
            "Signal": ["🟢 BUY" if p > last_price * 1.001 else
                       ("🔴 SELL" if p < last_price * 0.999 else "🟡 HOLD")
                       for p in predictions],
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        st.markdown(
            "<div style='color:#475569;font-size:0.72rem;margin-top:0.5rem'>⚠️ AI predictions are for educational purposes only. Not financial advice.</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if not show_ind:
        st.info("Enable **Technical Indicators** in sidebar.")
    else:
        fi = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.37, 0.32, 0.31],
            subplot_titles=["RSI (14)", "MACD", "Stochastic"],
        )

        if "RSI" in df.columns:
            fi.add_trace(
                go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#f59e0b", width=1.4)),
                row=1, col=1,
            )
            fi.add_hline(y=70, line_color="rgba(239,68,68,0.5)", line_dash="dash", row=1, col=1)
            fi.add_hline(y=30, line_color="rgba(16,185,129,0.5)", line_dash="dash", row=1, col=1)

        if "MACD" in df.columns:
            fi.add_trace(
                go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#00d4ff", width=1.4)),
                row=2, col=1,
            )
            fi.add_trace(
                go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#f97316", width=1.4)),
                row=2, col=1,
            )
            if "MACD_Hist" in df.columns:
                hc = ["#10b981" if v >= 0 else "#ef4444" for v in df["MACD_Hist"]]
                fi.add_trace(
                    go.Bar(x=df.index, y=df["MACD_Hist"], name="Hist", marker_color=hc, opacity=0.6),
                    row=2, col=1,
                )

        if "Stoch_K" in df.columns:
            fi.add_trace(
                go.Scatter(x=df.index, y=df["Stoch_K"], name="%K", line=dict(color="#8b5cf6", width=1.4)),
                row=3, col=1,
            )
            fi.add_trace(
                go.Scatter(x=df.index, y=df["Stoch_D"], name="%D", line=dict(color="#ec4899", width=1.4)),
                row=3, col=1,
            )
            fi.add_hline(y=80, line_color="rgba(239,68,68,0.4)", line_dash="dash", row=3, col=1)
            fi.add_hline(y=20, line_color="rgba(16,185,129,0.4)", line_dash="dash", row=3, col=1)

        fi.update_layout(
            height=540,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,10,20,0.95)",
            font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(17,24,39,0.9)", bordercolor="#1e3a5f", borderwidth=1),
            margin=dict(t=30, b=10, l=10, r=10),
        )
        fi.update_xaxes(gridcolor="rgba(30,58,95,0.2)")
        fi.update_yaxes(gridcolor="rgba(30,58,95,0.2)")
        st.plotly_chart(fi, use_container_width=True)

        if "RSI" in df.columns:
            rsi_v  = float(df["RSI"].iloc[-1])
            macd_v = float(df["MACD"].iloc[-1]) if "MACD" in df.columns else 0
            macd_s = float(df["MACD_Signal"].iloc[-1]) if "MACD_Signal" in df.columns else 0

            s1, s2, s3 = st.columns(3)
            with s1:
                sig = "🔴 Overbought" if rsi_v > 70 else ("🟢 Oversold" if rsi_v < 30 else "🟡 Neutral")
                st.markdown(
                    f"""<div class="metric-card" style="text-align:center">
                        <div class="label">RSI</div>
                        <div style="font-size:0.95rem;font-weight:600;margin:0.25rem 0">{sig}</div>
                        <div class="value" style="font-size:1.15rem">{rsi_v:.1f}</div>
                    </div>""", unsafe_allow_html=True,
                )
            with s2:
                msig = "🟢 Bullish" if macd_v > macd_s else "🔴 Bearish"
                st.markdown(
                    f"""<div class="metric-card" style="text-align:center">
                        <div class="label">MACD</div>
                        <div style="font-size:0.95rem;font-weight:600;margin:0.25rem 0">{msig}</div>
                        <div class="value" style="font-size:1.15rem">{macd_v:.4f}</div>
                    </div>""", unsafe_allow_html=True,
                )
            with s3:
                if "BB_Upper" in df.columns and pd.notna(df["BB_Upper"].iloc[-1]):
                    bb_p = (last_price - df["BB_Lower"].iloc[-1]) / (
                        df["BB_Upper"].iloc[-1] - df["BB_Lower"].iloc[-1] + 1e-9) * 100
                    bsig = "🔴 Upper" if bb_p > 80 else ("🟢 Lower" if bb_p < 20 else "🟡 Mid Band")
                    st.markdown(
                        f"""<div class="metric-card" style="text-align:center">
                            <div class="label">Bollinger</div>
                            <div style="font-size:0.95rem;font-weight:600;margin:0.25rem 0">{bsig}</div>
                            <div class="value" style="font-size:1.15rem">{bb_p:.0f}%</div>
                        </div>""", unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — DATA
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    dcols = ["Open", "High", "Low", "Close", "Volume"]
    if show_ind:
        for c in ["MA5", "MA10", "MA20", "MA50", "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "Stoch_K", "Stoch_D"]:
            if c in df.columns:
                dcols.append(c)

    ddf = df[dcols].copy().round(4).iloc[::-1]
    try:
        ddf.index = ddf.index.strftime("%d %b %Y  %H:%M:%S")
    except Exception:
        pass

    st.dataframe(ddf, use_container_width=True, height=500)
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(),
        f"{yf_ticker}_{period}_{interval}.csv",
        "text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["live_mode"] and refresh_every > 0 and not HAS_AUTOREFRESH:
    elapsed = time.time() - st.session_state["last_refresh"]
    wait_left = max(1, refresh_every - elapsed)

    prog_placeholder = st.empty()
    for remaining in range(int(wait_left), 0, -1):
        frac = 1 - (remaining / refresh_every)
        prog_placeholder.markdown(
            f"""
            <div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;
                padding:0.6rem 1.2rem;display:flex;align-items:center;gap:12px;margin-top:8px">
                <span style="color:#10b981;font-size:0.8rem">🟢 LIVE</span>
                <div style="flex:1;background:#1e293b;border-radius:4px;height:5px">
                    <div style="background:linear-gradient(90deg,#10b981,#00d4ff);
                        width:{int(frac*100)}%;height:5px;border-radius:4px"></div>
                </div>
                <span style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#475569">
                    refreshing in {remaining}s &nbsp;|&nbsp; {datetime.now().strftime('%H:%M:%S')}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(1)

    prog_placeholder.empty()
    fetch_stock_data_live.clear()
    st.session_state["last_refresh"] = time.time()
    st.session_state["refresh_count"] += 1
    st.rerun()
