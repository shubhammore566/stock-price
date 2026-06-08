import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=30)   # 30-second cache for live data
def fetch_stock_data_live(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data with a short TTL so live mode always gets fresh data.
    ttl=30s means data is at most 30 seconds stale.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, prepost=False)
        if df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Data fetch error for {ticker}: {e}")
        return None


# Keep backward compat alias
fetch_stock_data = fetch_stock_data_live


@st.cache_data(ttl=3600)
def get_company_info(ticker: str) -> dict:
    """Fetch company metadata (cached 1 hour)."""
    try:
        info = yf.Ticker(ticker).info
        return {
            'shortName':  info.get('shortName', ticker),
            'sector':     info.get('sector', 'N/A'),
            'industry':   info.get('industry', 'N/A'),
            'exchange':   info.get('exchange', 'N/A'),
            'currency':   info.get('currency', '$'),
            'marketCap':  info.get('marketCap', 0),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
            'fiftyTwoWeekLow':  info.get('fiftyTwoWeekLow', 0),
            'trailingPE': info.get('trailingPE', 0),
            'beta':       info.get('beta', 0),
        }
    except Exception:
        return {'shortName': ticker, 'sector': 'N/A',
                'exchange': 'N/A', 'currency': '$'}
