# pip install yfinance pandas numpy
import numpy as np, pandas as pd

PERIOD, MULT = 10, 3.0   # supertrend settings
FEE = 0.0003             # 0.03% per side
TP = 0.30                # take profit at +30% from entry (None = off)

STOCKS = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE
BAJAJFINSV BEL BHARTIARTL CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH
HDFCBANK HDFCLIFE HINDALCO HINDUNILVR ICICIBANK INDUSINDBK INFY ITC JSWSTEEL JIOFIN
KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE SBIN
SHRIRAMFIN SUNPHARMA TATACONSUM TMPV TATASTEEL TCS TECHM TITAN TRENT
ULTRACEMCO WIPRO""".split()

def supertrend(df, n, m):
    h, l, c = df.High.values, df.Low.values, df.Close.values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values
    hl2 = (h + l) / 2
    ub, lb = hl2 + m * atr, hl2 - m * atr
    fub, flb = ub.copy(), lb.copy()
    d = np.ones(len(c))  # 1 = green, -1 = red
    for i in range(1, len(c)):
        fub[i] = ub[i] if ub[i] < fub[i-1] or c[i-1] > fub[i-1] else fub[i-1]
        flb[i] = lb[i] if lb[i] > flb[i-1] or c[i-1] < flb[i-1] else flb[i-1]
        if d[i-1] == 1:
            d[i] = -1 if c[i] < flb[i] else 1
        else:
            d[i] = 1 if c[i] > fub[i] else -1
    return d

def backtest(df):
    d = supertrend(df, PERIOD, MULT)
    o, h, c = df.Open.values, df.High.values, df.Close.values
    cap, pos, entry, blocked = 1.0, False, 0.0, False
    eq, trades, tp_hits = [1.0], [], 0
    cap_in = 1.0
    for i in range(1, len(c)):
        sig = d[i-1]                      # signal from previous closed candle
        if sig == -1:
            blocked = False               # after a take-profit, wait for a fresh green
