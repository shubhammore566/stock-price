import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import fetch_stock_data_live, get_company_info
from predictor import StockPredictor
from utils import calculate_technical_indicators, format_large_number

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI – Live",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Auto-refresh component (streamlit-autorefresh) ───────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
* { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #060a14; color: #e2e8f0; }

.main-header {
    background: linear-gradient(135deg,#0d1b3e 0%,#1a0a2e 50%,#0d1b3e 100%);
    border: 1px solid #1e3a5f; border-radius: 14px;
    padding: 1.2rem 2rem; margin-bottom: 0.8rem;
}
.main-header h1 {
    font-size: 1.8rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg,#00d4ff,#7c3aed,#00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.main-header p { color: #94a3b8; margin: 0.15rem 0 0; font-size: 0.85rem; }

.datetime-strip {
    background: #0d1224; border: 1px solid #1a2540;
    border-radius: 10px; padding: 0.45rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 2rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
}
.datetime-strip .dt-label { color: #334155; }
.datetime-strip .dt-val   { color: #94a3b8; font-weight: 600; }
.datetime-strip .dt-live  { color: #10b981; animation: pulse 1.4s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }

.metric-card {
    background: linear-gradient(135deg,#111827,#1e293b);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 0.9rem 1.1rem;
}
.metric-card .label { color: #64748b; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { font-size: 1.4rem; font-weight: 700; color: #f1f5f9; font-family: 'JetBrains Mono', monospace; }
.positive { color: #10b981; } .negative { color: #ef4444; }
.info-chip {
    display:inline-block; background:#1e293b; border:1px solid #334155;
    border-radius:20px; padding:0.15rem 0.6rem; font-size:0.7rem; color:#94a3b8; margin:0.12rem;
}
.prediction-banner {
    background:linear-gradient(135deg,#0d2d0d,#0a1a2e);
    border:1px solid #10b981; border-radius:12px; padding:1.2rem; margin:0.7rem 0;
}
.prediction-banner.bearish { background:linear-gradient(135deg,#2d0d0d,#1a0a0a); border-color:#ef4444; }
[data-testid="stSidebar"] { background:#060a14 !important; }
.stButton>button {
    background:linear-gradient(135deg,#1d4ed8,#7c3aed); color:#fff;
    border:none; border-radius:8px; font-weight:600; padding:0.5rem 1rem; width:100%;
}
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 4px 16px rgba(124,58,237,.4); }
.stTabs [aria-selected="true"] { color:#00d4ff !important; border-bottom-color:#00d4ff !important; }
div[data-testid="column"] { padding:0 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
for k, v in [('live_mode', False), ('refresh_count', 0),
             ('last_refresh', 0.0), ('prev_price', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.6rem 0 0.2rem">
        <span style="font-size:1.6rem">📈</span>
        <h2 style="color:#00d4ff;margin:0.15rem 0 0;font-size:1.1rem">StockSense AI</h2>
        <p style="color:#475569;font-size:0.7rem;margin:0">Live Prediction Engine</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    STOCKS = {
        "🍎 Apple (AAPL)": "AAPL",        "🪟 Microsoft (MSFT)": "MSFT",
        "🔍 Google (GOOGL)": "GOOGL",      "⚡ Tesla (TSLA)": "TSLA",
        "📦 Amazon (AMZN)": "AMZN",        "🤖 NVIDIA (NVDA)": "NVDA",
        "📘 Meta (META)": "META",           "🎬 Netflix (NFLX)": "NFLX",
        "💳 Visa (V)": "V",                 "🏦 JPMorgan (JPM)": "JPM",
        "☕ Starbucks (SBUX)": "SBUX",      "🛒 Walmart (WMT)": "WMT",
        "🔑 Custom Ticker": "CUSTOM"
    }
    st.markdown("**🏢 Company**")
    sel    = st.selectbox("", list(STOCKS.keys()), label_visibility="collapsed")
    ticker = STOCKS[sel]
    if ticker == "CUSTOM":
        ticker = st.text_input("Enter ticker (e.g. RELIANCE.NS)", "RELIANCE.NS").upper().strip()

    st.markdown("**⏱️ Interval**")
    interval_map = {"1 Min":"1m","5 Min":"5m","15 Min":"15m","30 Min":"30m",
                    "1 Hour":"1h","1 Day":"1d","1 Week":"1wk","1 Month":"1mo"}
    int_label = st.selectbox("", list(interval_map.keys()), index=0, label_visibility="collapsed")
    interval  = interval_map[int_label]

    st.markdown("**📅 History Period**")
    period_map = {"1 Day":"1d","5 Days":"5d","1 Month":"1mo","3 Months":"3mo",
                  "6 Months":"6mo","1 Year":"1y","2 Years":"2y"}
    per_label = st.selectbox("", list(period_map.keys()), index=0, label_visibility="collapsed")
    period    = period_map[per_label]

    st.markdown("**🔮 Prediction Days**")
    pred_days    = st.slider("", 1, 30, 5, label_visibility="collapsed")
    st.markdown("**🧠 ML Model**")
    model_choice = st.selectbox("",
        ["Linear Regression","Random Forest","LSTM Neural Network","All Models (Ensemble)"],
        label_visibility="collapsed")

    show_ind = st.checkbox("Technical Indicators", value=True)
    show_vol = st.checkbox("Show Volume",           value=True)
    st.divider()

    st.markdown("**⚡ Live Mode**")
    refresh_sec = st.selectbox("Refresh every",
        ["Off","30 sec","1 min","2 min","5 min"], index=2,
        label_visibility="collapsed")

    c1, c2 = st.columns(2)
    with c1:
        load_btn = st.button("🚀 Load")
    with c2:
        live_lbl = "🔴 Stop" if st.session_state['live_mode'] else "🟢 Go Live"
        live_btn = st.button(live_lbl)

    if load_btn:
        st.session_state['live_mode'] = False
        fetch_stock_data_live.clear()

    if live_btn:
        st.session_state['live_mode'] = not st.session_state['live_mode']
        st.session_state['last_refresh'] = 0.0
        fetch_stock_data_live.clear()

    st.divider()
    st.markdown("""<div style="text-align:center;color:#334155;font-size:0.65rem">
        Data: Yahoo Finance • ⚠️ Not financial advice</div>""", unsafe_allow_html=True)

# ─── Refresh interval ─────────────────────────────────────────────────────────
rsec_map = {"Off":0,"30 sec":30,"1 min":60,"2 min":120,"5 min":300}
refresh_every = rsec_map[refresh_sec]

# ─── streamlit-autorefresh: keep clock ticking every second when live ─────────
if st.session_state['live_mode'] and refresh_every > 0 and HAS_AUTOREFRESH:
    st_autorefresh(interval=refresh_every * 1000, key="live_refresh")

# ─── Real-time date & time ────────────────────────────────────────────────────
now_dt      = datetime.now()
now_date    = now_dt.strftime("%A, %d %B %Y")          # e.g. Monday, 08 June 2026
now_time    = now_dt.strftime("%H:%M:%S")               # 24-hr
now_time_12 = now_dt.strftime("%I:%M:%S %p")            # 12-hr AM/PM
weekday     = now_dt.strftime("%A")

# ─── HEADER ───────────────────────────────────────────────────────────────────
live_status = "🟢 LIVE" if st.session_state['live_mode'] else "⏸ PAUSED"
live_color  = "#10b981"  if st.session_state['live_mode'] else "#475569"
st.markdown(f"""
<div class="main-header">
    <h1>📈 StockSense AI</h1>
    <p>Real-Time Market Data • ML Predictions • Technical Analysis &nbsp;
       <span style="color:{live_color};font-weight:700">{live_status}</span>
       {"&nbsp;| Refresh #" + str(st.session_state['refresh_count']) if st.session_state['refresh_count'] else ""}
    </p>
</div>""", unsafe_allow_html=True)

# ─── LIVE DATE & TIME STRIP ───────────────────────────────────────────────────
elapsed_since = time.time() - st.session_state['last_refresh']
next_in       = max(0, int(refresh_every - elapsed_since)) if (st.session_state['live_mode'] and refresh_every > 0) else 0

st.markdown(f"""
<div class="datetime-strip">
    <span class="dt-live">● LIVE</span>
    <span><span class="dt-label">📅 Date&nbsp;</span><span class="dt-val">{now_date}</span></span>
    <span><span class="dt-label">🕐 Time&nbsp;</span><span class="dt-val">{now_time_12}</span></span>
    <span><span class="dt-label">⟳ Next refresh&nbsp;</span>
          <span class="dt-val" style="color:{'#10b981' if st.session_state['live_mode'] else '#334155'}">
          {'in ' + str(next_in) + 's' if st.session_state['live_mode'] and refresh_every>0 else 'Off'}
          </span></span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FETCH DATA
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner(f"⚡ Fetching {ticker} ({int_label}, {per_label}) …"):
    df = fetch_stock_data_live(ticker, period, interval)

if df is None or df.empty:
    st.error(f"❌ No data for **{ticker}**. Check ticker symbol or try a different period/interval combo.")
    st.stop()

# ─── Compute key values ───────────────────────────────────────────────────────
last_price = float(df['Close'].iloc[-1])
prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else last_price
price_chg  = last_price - prev_close
pct_chg    = (price_chg / prev_close * 100) if prev_close else 0
is_up      = price_chg >= 0
chg_icon   = "▲" if is_up else "▼"
chg_color  = "#10b981" if is_up else "#ef4444"
info       = get_company_info(ticker)
currency   = info.get('currency', '$')

try:
    last_ts = df.index[-1].strftime("%d %b %Y  %H:%M:%S")
except:
    last_ts = str(df.index[-1])

# Flash vs previous session
prev_sess   = st.session_state.get('prev_price')
flash_text  = ""
flash_col   = "#10b981"
if prev_sess and prev_sess != last_price:
    diff = last_price - prev_sess
    flash_text = f"{'▲' if diff>0 else '▼'} {abs(diff):.2f} from last refresh"
    flash_col  = "#10b981" if diff>0 else "#ef4444"
st.session_state['prev_price'] = last_price

# ─── TICKER BAR ───────────────────────────────────────────────────────────────
tb1, tb2, tb3, tb4, tb5 = st.columns([1.2, 1.5, 1.5, 2, 2])
with tb1:
    dot = "🟢" if st.session_state['live_mode'] else "⚪"
    st.markdown(f"<div style='padding:0.4rem 0;font-family:JetBrains Mono,monospace;font-size:0.85rem;color:#94a3b8'>{dot} <b>{ticker}</b></div>", unsafe_allow_html=True)
with tb2:
    st.markdown(f"<div style='padding:0.3rem 0;font-family:JetBrains Mono,monospace;font-size:1.5rem;font-weight:700;color:#f1f5f9'>{currency}{last_price:.2f}</div>", unsafe_allow_html=True)
with tb3:
    st.markdown(f"<div style='padding:0.4rem 0;font-size:1rem;font-weight:700;color:{chg_color}'>{chg_icon} {abs(price_chg):.2f} ({abs(pct_chg):.2f}%)</div>", unsafe_allow_html=True)
with tb4:
    st.markdown(f"<div style='padding:0.4rem 0;font-size:0.75rem;color:#475569'>🕐 Last bar: {last_ts}</div>", unsafe_allow_html=True)
with tb5:
    if st.session_state['live_mode'] and refresh_every > 0:
        st.markdown(f"<div style='padding:0.4rem 0;font-size:0.75rem;color:#334155'>⟳ Next refresh in <b style='color:#475569'>{next_in}s</b></div>", unsafe_allow_html=True)
    elif flash_text:
        st.markdown(f"<div style='padding:0.4rem 0;font-size:0.75rem;color:{flash_col}'>{flash_text}</div>", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e3a5f;margin:0.4rem 0 0.8rem'>", unsafe_allow_html=True)

# ─── METRIC CARDS ─────────────────────────────────────────────────────────────
high_p  = df['High'].max()
low_p   = df['Low'].min()
avg_vol = df['Volume'].mean() if 'Volume' in df.columns else 0
vol_std = df['Close'].pct_change().std() * 100

mc0, mc1, mc2, mc3, mc4 = st.columns([2,1,1,1,1])
with mc0:
    st.markdown(f"""<div class="metric-card">
        <div style="font-size:1.3rem;font-weight:700;color:#f1f5f9">{info.get('shortName', ticker)}</div>
        <div style="color:#64748b;font-size:0.78rem;margin:0.15rem 0">{info.get('sector','—')} • {info.get('exchange','—')}</div>
        <span class="info-chip">{ticker}</span>
        <span class="info-chip">{currency}</span>
        <span class="info-chip">{int_label}</span>
    </div>""", unsafe_allow_html=True)
with mc1:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Price</div>
        <div class="value">{currency}{last_price:.2f}</div>
        <div style="color:{chg_color};font-size:0.82rem;font-weight:600">{chg_icon} {abs(pct_chg):.2f}%</div>
    </div>""", unsafe_allow_html=True)
with mc2:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Period H / L</div>
        <div class="value" style="font-size:1.1rem">{high_p:.2f}</div>
        <div style="color:#64748b;font-size:0.73rem">Low: {low_p:.2f}</div>
    </div>""", unsafe_allow_html=True)
with mc3:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Avg Volume</div>
        <div class="value" style="font-size:1.1rem">{format_large_number(avg_vol)}</div>
        <div style="color:#64748b;font-size:0.73rem">Shares/bar</div>
    </div>""", unsafe_allow_html=True)
with mc4:
    vc = '#f59e0b' if vol_std > 2 else '#10b981'
    st.markdown(f"""<div class="metric-card">
        <div class="label">Volatility</div>
        <div class="value" style="font-size:1.1rem;color:{vc}">{vol_std:.2f}%</div>
        <div style="color:#64748b;font-size:0.73rem">Daily σ</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ─── Technical Indicators ─────────────────────────────────────────────────────
if show_ind:
    df = calculate_technical_indicators(df)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Chart", "🔮 AI Prediction", "📉 Indicators", "📋 Data"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    rows = 2 if show_vol else 1
    rh   = [0.73, 0.27] if show_vol else [1]
    fig  = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                         vertical_spacing=0.03, row_heights=rh)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="OHLC",
        increasing_line_color='#10b981', decreasing_line_color='#ef4444',
        increasing_fillcolor='rgba(16,185,129,0.22)',
        decreasing_fillcolor='rgba(239,68,68,0.22)'
    ), row=1, col=1)

    if show_ind:
        for ma, clr in [('MA20','#f59e0b'), ('MA50','#8b5cf6')]:
            if ma in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma,
                    line=dict(color=clr, width=1.3, dash='dot')), row=1, col=1)
        if 'BB_Upper' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper",
                line=dict(color='rgba(0,212,255,0.3)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower",
                line=dict(color='rgba(0,212,255,0.3)', width=1),
                fill='tonexty', fillcolor='rgba(0,212,255,0.04)'), row=1, col=1)

    # Live price line
    fig.add_hline(y=last_price,
        line_color='rgba(0,212,255,0.55)', line_dash='dash', line_width=1.2,
        annotation_text=f"  ₹{last_price:.2f}" if currency=='INR' else f"  {currency}{last_price:.2f}",
        annotation_position="right",
        annotation_font_color="#00d4ff", row=1, col=1)

    if show_vol and 'Volume' in df.columns:
        vcols = ['#10b981' if c >= o else '#ef4444'
                 for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume",
            marker_color=vcols, opacity=0.5), row=2, col=1)

    # ── Chart title shows REAL date + time ────────────────────────────────────
    chart_title = f"  {ticker}  |  {int_label}  |  {now_date}  {now_time}"

    fig.update_layout(
        height=560,
        title=dict(text=chart_title,
                   font=dict(color='#64748b', size=12)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(6,10,20,0.95)',
        font=dict(color='#94a3b8', size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor='rgba(17,24,39,0.9)', bordercolor='#1e3a5f',
                    borderwidth=1, font=dict(size=10)),
        margin=dict(t=35, b=10, l=10, r=70),
        hovermode='x unified'
    )
    fig.update_xaxes(gridcolor='rgba(30,58,95,0.2)', showgrid=True)
    fig.update_yaxes(gridcolor='rgba(30,58,95,0.2)', showgrid=True)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🕯️ Recent 10 Candles"):
        rc = df[['Open','High','Low','Close','Volume']].tail(10).iloc[::-1].round(3)
        try:
            rc.index = rc.index.strftime('%d %b %Y  %H:%M:%S')
        except:
            pass
        st.dataframe(rc, use_container_width=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    with st.spinner("🧠 Training model …"):
        predictor = StockPredictor(model_choice)
        predictions, confidence, metrics = predictor.predict(df, pred_days)

    if predictions is not None:
        future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=pred_days, freq='B')
        pred_chg = predictions[-1] - last_price
        pred_pct = pred_chg / last_price * 100
        is_bull  = pred_chg >= 0

        st.markdown(f"""
        <div class="prediction-banner {'bullish' if is_bull else 'bearish'}">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px">
                        {'🟢 BULLISH' if is_bull else '🔴 BEARISH'} — {model_choice}
                    </div>
                    <div style="font-size:1.5rem;font-weight:700;color:{'#10b981' if is_bull else '#ef4444'}">
                        {'+' if is_bull else ''}{pred_pct:.2f}% over {pred_days} days
                    </div>
                    <div style="color:#94a3b8;font-size:0.85rem">
                        {currency}{last_price:.2f} → <b style="color:#f1f5f9">{currency}{predictions[-1]:.2f}</b>
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:0.7rem;color:#64748b">CONFIDENCE</div>
                    <div style="font-size:1.9rem;font-weight:700;color:#00d4ff">{confidence:.0f}%</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        fp = go.Figure()
        hw = min(90, len(df))
        fp.add_trace(go.Scatter(x=df.index[-hw:], y=df['Close'].iloc[-hw:],
            name="Historical", line=dict(color='#00d4ff', width=1.8)))
        upper = [p*1.03 for p in predictions]
        lower = [p*0.97 for p in predictions]
        fp.add_trace(go.Scatter(
            x=list(future_dates)+list(future_dates)[::-1],
            y=upper+lower[::-1], fill='toself',
            fillcolor='rgba(124,58,237,0.1)',
            line=dict(color='rgba(0,0,0,0)'), name='±3% Band'))
        fp.add_trace(go.Scatter(x=future_dates, y=predictions,
            name="Forecast", mode='lines+markers',
            line=dict(color='#7c3aed', width=2.2, dash='dash'),
            marker=dict(size=6, color='#7c3aed')))
        fp.add_trace(go.Scatter(
            x=[df.index[-1], future_dates[0]], y=[last_price, predictions[0]],
            line=dict(color='#7c3aed', width=1.1, dash='dot'),
            showlegend=False, mode='lines'))

        fp.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(6,10,20,0.95)', font=dict(color='#94a3b8'),
            legend=dict(bgcolor='rgba(17,24,39,0.9)',bordercolor='#1e3a5f',borderwidth=1),
            margin=dict(t=10,b=10,l=10,r=20), hovermode='x unified')
        fp.update_xaxes(gridcolor='rgba(30,58,95,0.2)')
        fp.update_yaxes(gridcolor='rgba(30,58,95,0.2)')
        st.plotly_chart(fp, use_container_width=True)

        st.markdown("**📊 Model Metrics**")
        mcols = st.columns(4)
        for col, (lbl, val) in zip(mcols, [
            ("RMSE", f"{metrics.get('rmse',0):.4f}"),
            ("MAE",  f"{metrics.get('mae',0):.4f}"),
            ("R²",   f"{metrics.get('r2',0):.4f}"),
            ("MAPE", f"{metrics.get('mape',0):.2f}%"),
        ]):
            with col:
                st.markdown(f"""<div class="metric-card" style="text-align:center">
                    <div class="label">{lbl}</div>
                    <div class="value" style="font-size:1rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>**📅 Forecast Table**", unsafe_allow_html=True)
        pred_df = pd.DataFrame({
            'Date': future_dates.strftime('%d %b %Y'),
            f'Price ({currency})': [f"{p:.2f}" for p in predictions],
            'Change': [f"{'+' if p-last_price>=0 else ''}{p-last_price:.2f}" for p in predictions],
            'Change %': [f"{'+' if (p-last_price)/last_price*100>=0 else ''}{(p-last_price)/last_price*100:.2f}%" for p in predictions],
            'Signal': ['🟢 BUY' if p > last_price else '🔴 SELL' for p in predictions]
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    if not show_ind:
        st.info("Enable **Technical Indicators** in sidebar.")
    else:
        fi = make_subplots(rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.37,0.32,0.31],
            subplot_titles=["RSI (14)", "MACD", "Stochastic"])
        if 'RSI' in df.columns:
            fi.add_trace(go.Scatter(x=df.index,y=df['RSI'],name="RSI",
                line=dict(color='#f59e0b',width=1.4)),row=1,col=1)
            fi.add_hline(y=70,line_color='rgba(239,68,68,0.5)',line_dash='dash',row=1,col=1)
            fi.add_hline(y=30,line_color='rgba(16,185,129,0.5)',line_dash='dash',row=1,col=1)
        if 'MACD' in df.columns:
            fi.add_trace(go.Scatter(x=df.index,y=df['MACD'],name="MACD",
                line=dict(color='#00d4ff',width=1.4)),row=2,col=1)
            fi.add_trace(go.Scatter(x=df.index,y=df['MACD_Signal'],name="Signal",
                line=dict(color='#f97316',width=1.4)),row=2,col=1)
            if 'MACD_Hist' in df.columns:
                hc=['#10b981' if v>=0 else '#ef4444' for v in df['MACD_Hist']]
                fi.add_trace(go.Bar(x=df.index,y=df['MACD_Hist'],name="Hist",
                    marker_color=hc,opacity=0.6),row=2,col=1)
        if 'Stoch_K' in df.columns:
            fi.add_trace(go.Scatter(x=df.index,y=df['Stoch_K'],name="%K",
                line=dict(color='#8b5cf6',width=1.4)),row=3,col=1)
            fi.add_trace(go.Scatter(x=df.index,y=df['Stoch_D'],name="%D",
                line=dict(color='#ec4899',width=1.4)),row=3,col=1)
            fi.add_hline(y=80,line_color='rgba(239,68,68,0.4)',line_dash='dash',row=3,col=1)
            fi.add_hline(y=20,line_color='rgba(16,185,129,0.4)',line_dash='dash',row=3,col=1)
        fi.update_layout(height=540,paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(6,10,20,0.95)',font=dict(color='#94a3b8'),
            legend=dict(bgcolor='rgba(17,24,39,0.9)',bordercolor='#1e3a5f',borderwidth=1),
            margin=dict(t=30,b=10,l=10,r=10))
        fi.update_xaxes(gridcolor='rgba(30,58,95,0.2)')
        fi.update_yaxes(gridcolor='rgba(30,58,95,0.2)')
        st.plotly_chart(fi, use_container_width=True)

        if 'RSI' in df.columns:
            rsi_v = df['RSI'].iloc[-1]
            macd_v = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
            macd_s = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns else 0
            s1,s2,s3 = st.columns(3)
            with s1:
                sig = "🔴 Overbought" if rsi_v>70 else ("🟢 Oversold" if rsi_v<30 else "🟡 Neutral")
                st.markdown(f"""<div class="metric-card" style="text-align:center">
                    <div class="label">RSI</div><div style="font-size:0.95rem;font-weight:600;margin:0.25rem 0">{sig}</div>
                    <div class="value" style="font-size:1.15rem">{rsi_v:.1f}</div></div>""", unsafe_allow_html=True)
            with s2:
                msig = "🟢 Bullish" if macd_v>macd_s else "🔴 Bearish"
                st.markdown(f"""<div class="metric-card" style="text-align:center">
                    <div class="label">MACD</div><div style="font-size:0.95rem;font-weight:600;margin:0.25rem 0">{msig}</div>
                    <div class="value" style="font-size:1.15rem">{macd_v:.4f}</div></div>""", unsafe_allow_html=True)
            with s3:
                if 'BB_Upper' in df.columns:
                    bb_p  = (last_price-df['BB_Lower'].iloc[-1])/(df['BB_Upper'].iloc[-1]-df['BB_Lower'].iloc[-1]+1e-9)*100
                    bsig  = "🔴 Upper" if bb_p>80 else ("🟢 Lower" if bb_p<20 else "🟡 Mid Band")
                    st.markdown(f"""<div class="metric-card" style="text-align:center">
                        <div class="label">Bollinger</div><div style="font-size:0.95rem;font-weight:600;margin:0.25rem 0">{bsig}</div>
                        <div class="value" style="font-size:1.15rem">{bb_p:.0f}%</div></div>""", unsafe_allow_html=True)

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    dcols = ['Open','High','Low','Close','Volume']
    if show_ind:
        for c in ['MA20','MA50','RSI','MACD']:
            if c in df.columns: dcols.append(c)
    ddf = df[dcols].copy().round(4).iloc[::-1]
    try:
        ddf.index = ddf.index.strftime('%d %b %Y  %H:%M:%S')
    except:
        pass
    st.dataframe(ddf, use_container_width=True, height=500)
    st.download_button("⬇️ Download CSV", df.to_csv(),
        f"{ticker}_{period}_{interval}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# LIVE AUTO-REFRESH  (fallback if streamlit-autorefresh not installed)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state['live_mode'] and refresh_every > 0 and not HAS_AUTOREFRESH:
    elapsed   = time.time() - st.session_state['last_refresh']
    wait_left = max(1, refresh_every - elapsed)

    prog_placeholder = st.empty()
    for remaining in range(int(wait_left), 0, -1):
        frac = 1 - (remaining / refresh_every)
        prog_placeholder.markdown(
            f"""<div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;
                padding:0.6rem 1.2rem;display:flex;align-items:center;gap:12px;margin-top:8px">
                <span style="color:#10b981;font-size:0.8rem">🟢 LIVE</span>
                <div style="flex:1;background:#1e293b;border-radius:4px;height:5px">
                    <div style="background:linear-gradient(90deg,#10b981,#00d4ff);
                        width:{int(frac*100)}%;height:5px;border-radius:4px"></div>
                </div>
                <span style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#475569">
                    refreshing in {remaining}s &nbsp;|&nbsp; {datetime.now().strftime('%H:%M:%S')}
                </span>
            </div>""",
            unsafe_allow_html=True
        )
        time.sleep(1)

    prog_placeholder.empty()
    fetch_stock_data_live.clear()
    st.session_state['last_refresh']  = time.time()
    st.session_state['refresh_count'] += 1
    st.rerun()

elif st.session_state['live_mode'] and refresh_every > 0 and HAS_AUTOREFRESH:
    # autorefresh handles the rerun; just update session counters on each run
    st.session_state['last_refresh']  = time.time()
    st.session_state['refresh_count'] += 1