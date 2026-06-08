import pandas as pd
import numpy as np


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI, MACD, Bollinger Bands, Stochastic, Moving Averages."""
    df = df.copy()

    # ── Moving Averages ──────────────────────────────────────────────────────
    for window in [5, 10, 20, 50, 200]:
        if len(df) >= window:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        else:
            df[f'MA{window}'] = np.nan

    # Exponential MAs
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # ── RSI ──────────────────────────────────────────────────────────────────
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────────────────────
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # ── Bollinger Bands (20, 2σ) ─────────────────────────────────────────────
    bb_period = 20
    if len(df) >= bb_period:
        bb_ma = df['Close'].rolling(window=bb_period).mean()
        bb_std = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = bb_ma + 2 * bb_std
        df['BB_Lower'] = bb_ma - 2 * bb_std
        df['BB_Mid'] = bb_ma
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    else:
        df['BB_Upper'] = df['BB_Lower'] = df['BB_Mid'] = df['BB_Width'] = np.nan

    # ── Stochastic Oscillator ────────────────────────────────────────────────
    stoch_period = 14
    if len(df) >= stoch_period:
        low_min = df['Low'].rolling(stoch_period).min()
        high_max = df['High'].rolling(stoch_period).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-9)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    else:
        df['Stoch_K'] = df['Stoch_D'] = np.nan

    # ── ATR (Average True Range) ─────────────────────────────────────────────
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # ── OBV (On-Balance Volume) ──────────────────────────────────────────────
    if 'Volume' in df.columns:
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv

    return df


def format_large_number(num: float) -> str:
    """Format numbers with K/M/B suffixes."""
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return "N/A"
    num = float(num)
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return f"{num:.0f}"


def get_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """Identify key support and resistance levels."""
    highs = df['High'].rolling(window, center=True).max()
    lows = df['Low'].rolling(window, center=True).min()

    resistance_levels = df['High'][df['High'] == highs].dropna().tail(3).values.tolist()
    support_levels = df['Low'][df['Low'] == lows].dropna().tail(3).values.tolist()

    return {
        'resistance': sorted(set(round(r, 2) for r in resistance_levels), reverse=True),
        'support': sorted(set(round(s, 2) for s in support_levels), reverse=True)
    }
